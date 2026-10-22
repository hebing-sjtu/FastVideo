"""Score phase-0 takes: candidate rewards and, separately, metrics against the real target.

Rewards are what a reward-driven fine-tune could optimise; the independent metrics are what
best-of-N selection by a reward is judged on, so a reward is only useful if picking its best
seed also raises them.

Rewards:
    depth      - Depth Anything V2 disparity of the take, aligned per frame (scale and shift) to
                 the target's, as negative relative error
    motion     - correlation of the per-frame change curves of take and target
    vlm_event  - Qwen3-VL (H3 text encoder): do the captioned actions happen (expected 0-9 score)
    vlm_identity - ... does the protagonist match the OC sheet (OC samples only)
    vlm_quality  - ... visual quality, artifacts
Independent:
    psnr_late, ssim_late - ffmpeg, frames >= 24, as on the eval board
    dino       - DINOv2 CLS cosine similarity to the target, mean over sampled frames

    python phase0_score.py --spool /workspace/rl0/spool --eval-dir <R5-B@600 correct dir> \
        --out /workspace/rl0/scores.jsonl --watch
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SERVE = Path(__file__).resolve().parents[1] / "serve" / "proxy_service.py"
SKIP_FRAMES = 24
DEPTH_FRAMES = 16
VLM_FRAMES = 8

JUDGE = {
    "event": "Do the actions and interactions described in the caption actually happen in the frames, in order "
             "(for example getting into or out of a vehicle, driving, a punch landing and the other person "
             "falling)? 0 = none of them, 9 = all of them, clearly.",
    "identity": "Does the protagonist (the character the camera follows) match the identity sheet: outfit, colours, "
                "hair, build? 0 = a different person, 9 = clearly the same character in every frame.",
    "quality": "Rate the visual quality: sharp and coherent bodies, faces and vehicles, no melting, smearing, "
               "flicker, duplicated limbs or objects popping in and out. 0 = broken, 9 = flawless.",
}


def load_serve():
    spec = importlib.util.spec_from_file_location("proxy_service", SERVE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_service"] = module
    spec.loader.exec_module(module)
    return module


def read_video(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def per_frame(generated: Path, target: Path, metric: str) -> list[float]:
    with tempfile.NamedTemporaryFile(suffix=".log") as stats:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(generated), "-i", str(target), "-lavfi",
                        f"[0:v][1:v]{metric}=stats_file={stats.name}", "-f", "null", "-"], check=True)
        text = Path(stats.name).read_text()
    pattern = r"psnr_avg:(\S+)" if metric == "psnr" else r"All:(\S+)"
    return [100.0 if match == "inf" else float(match) for match in re.findall(pattern, text)]


def picks(count: int, frames: int) -> np.ndarray:
    return np.linspace(0, frames - 1, num=count).round().astype(int)


def aligned_error(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean relative error after a per-frame least-squares scale and shift of ``pred`` onto ``ref``."""
    errors = []
    for p, r in zip(pred.flatten(1), ref.flatten(1)):
        design = torch.stack([p, torch.ones_like(p)], 1)
        solution = torch.linalg.lstsq(design, r[:, None]).solution
        errors.append(((design @ solution)[:, 0] - r).abs().mean() / r.abs().mean().clamp_min(1e-6))
    return float(torch.stack(errors).mean())


def motion_correlation(take: np.ndarray, target: np.ndarray) -> float:
    def curve(frames: np.ndarray) -> np.ndarray:
        grey = frames[:, ::8, ::8].astype(np.float32).mean(-1)
        return np.abs(np.diff(grey, axis=0)).mean((1, 2))

    a, b = curve(take), curve(target[: len(take)])
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class Scorer:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation

        self.serve = load_serve()
        self.device = torch.device(args.device)
        self.vlm = self.serve.Vlm(Path(args.model), args.device, Path(args.work))
        tokenizer = self.vlm.processor.tokenizer
        self.digits = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
        self.depth_processor = AutoImageProcessor.from_pretrained(args.depth_model)
        self.depth = AutoModelForDepthEstimation.from_pretrained(args.depth_model).to(self.device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)
        self.dino = AutoModel.from_pretrained(args.dino_model).to(self.device).eval()

    @torch.no_grad()
    def disparity(self, frames: np.ndarray) -> torch.Tensor:
        inputs = self.depth_processor(images=[Image.fromarray(f) for f in frames], return_tensors="pt").to(self.device)
        return self.depth(**inputs).predicted_depth.float()

    @torch.no_grad()
    def embed(self, frames: np.ndarray) -> torch.Tensor:
        inputs = self.dino_processor(images=[Image.fromarray(f) for f in frames], return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.dino(**inputs).last_hidden_state[:, 0].float(), dim=-1)

    @torch.no_grad()
    def expected_digit(self, images: list[Image.Image], text: str) -> float:
        vlm = self.vlm
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        prompt = vlm.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                   add_generation_prompt=True)
        inputs = vlm.processor(text=[prompt], images=images, return_tensors="pt").to(vlm.device)
        logits = vlm.model(**inputs).logits[0, -1].float()
        self.digit_mass = min(self.digit_mass, float(torch.softmax(logits, dim=0)[self.digits].sum()))
        probs = torch.softmax(logits[self.digits], dim=0)
        return float((probs * torch.arange(10, device=probs.device)).sum())

    def judge(self, take: np.ndarray, caption: str, oc: Image.Image | None) -> dict[str, float | None]:
        frames = self.serve.sample_frames(take, VLM_FRAMES)
        self.digit_mass = 1.0
        scores: dict[str, float | None] = {}
        for name, question in JUDGE.items():
            if name == "identity" and oc is None:
                scores[f"vlm_{name}"] = None
                continue
            images = ([self.serve.thumbnail(oc, 448)] if name == "identity" else []) + frames
            lead = ("The first image is the protagonist identity sheet; the next " if name == "identity" else
                    "The ") + f"{VLM_FRAMES} images are frames of a generated video, in order."
            text = (f"{lead}\nCaption of the video:\n{caption}\n\n{question}\n"
                    "Answer with a single digit from 0 to 9 and nothing else.")
            scores[f"vlm_{name}"] = self.expected_digit(images, text)
        scores["vlm_digit_mass"] = self.digit_mass
        return scores

    def score(self, take_path: Path, target_path: Path, caption: str, oc: Image.Image | None) -> dict[str, Any]:
        take, target = read_video(take_path), read_video(target_path)
        n = min(len(take), len(target))
        psnr, ssim = per_frame(take_path, target_path, "psnr"), per_frame(take_path, target_path, "ssim")
        index = picks(DEPTH_FRAMES, n)
        dino = (self.embed(take[index]) * self.embed(target[index])).sum(-1).mean()
        row = {
            "frames": n,
            "psnr_late": statistics.mean(psnr[SKIP_FRAMES:]),
            "ssim_late": statistics.mean(ssim[SKIP_FRAMES:]),
            "dino": float(dino),
            "depth": -aligned_error(self.disparity(take[index]), self.disparity(target[index])),
            "motion": motion_correlation(take[:n], target[:n]),
        }
        row.update(self.judge(take[:n], caption, oc))
        return row


def eval_slots(eval_dir: Path) -> dict[str, Path]:
    slots = {}
    for manifest in eval_dir.glob("proxy-inference/wave-*/dp-rank-*/slot-*/manifest.json"):
        slots[json.loads(manifest.read_text())["case"]["sample_id"]] = manifest.parent
    return slots


def sample_info(data: Path) -> dict[str, dict[str, Any]]:
    info = {}
    for line in data.read_text().splitlines():
        if line.strip():
            path = Path(line.split()[0])
            info[path.stem] = torch.load(path, map_location="cpu", weights_only=False)["info"]
    return info


def parse_job(job_id: str) -> tuple[int, str]:
    seed, _, sample = job_id.split("-", 2)
    return int(seed[1:]), sample


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spool", required=True)
    p.add_argument("--eval-dir", required=True, help="the model's seed-42 eval task dir (holds target.mp4)")
    p.add_argument("--data", default="/data/binghe/h3_proxy/manifests/gta_record_0930_omni_704p_ref640_heldout32_oc.txt")
    p.add_argument("--dataset-root", default="/data/binghe/datasets/gta_record_0930-v0003", help="root of info['oc']")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="/data/models/MiniMax-H3")
    p.add_argument("--work", default="/workspace/rl0/work")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Base-hf")
    p.add_argument("--dino-model", default="facebook/dinov2-base")
    p.add_argument("--also-eval", action="store_true", help="also score the eval's own seed-42 generated.mp4")
    p.add_argument("--watch", action="store_true", help="keep scoring until the spool has STOP and nothing is left")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    spool, out = Path(args.spool), Path(args.out)
    slots, infos = eval_slots(Path(args.eval_dir)), sample_info(Path(args.data))
    seen = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.is_file() else set()
    scorer = Scorer(args)
    print(f"[score] ready: {len(slots)} eval slots, {len(infos)} samples, {len(seen)} already scored", flush=True)

    def todo() -> list[tuple[str, Path, str, int]]:
        rows = []
        for done in sorted(spool.glob("done/*.json")):
            if done.stem in seen:
                continue
            payload = json.loads(done.read_text())
            if payload.get("error"):
                continue
            seed, sample = parse_job(done.stem)
            rows.append((done.stem, Path(payload["take"]), sample, seed))
        if args.also_eval:
            rows += [(f"eval-s0042-{s}", slot / "generated.mp4", s, 42) for s, slot in sorted(slots.items())
                     if f"eval-s0042-{s}" not in seen]
        return rows

    scored = 0
    while True:
        rows = todo()
        for job_id, take, sample, seed in rows:
            info = infos[sample]
            oc_path = Path(args.dataset_root) / info["oc"] if info.get("oc") else None
            oc = Image.open(oc_path).convert("RGB") if oc_path and oc_path.is_file() else None
            started = time.time()
            row = scorer.score(take, slots[sample] / "target.mp4", info["prompt"], oc)
            row.update({"id": job_id, "sample": sample, "seed": seed, "modality": info.get("proxy_modality"),
                        "target_kind": info.get("target_kind"), "oc": oc is not None,
                        "source": "eval" if job_id.startswith("eval-") else "serve"})
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            seen.add(job_id)
            scored += 1
            print(f"[score] {job_id} {time.time() - started:.1f}s psnr_late={row['psnr_late']:.2f} "
                  f"depth={row['depth']:.3f} event={row['vlm_event']:.2f} quality={row['vlm_quality']:.2f}",
                  flush=True)
            if args.limit and scored >= args.limit:
                return
        if not args.watch or ((spool / "STOP").exists() and not todo()):
            break
        time.sleep(60)
    print(f"[score] finished: {scored} scored", flush=True)


if __name__ == "__main__":
    main()
