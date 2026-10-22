"""Pairwise VLM judge for phase 0: which of two seeds of one sample is better.

Absolute 0-9 VLM scores saturate near 9 and barely separate seeds, so this asks for a choice
instead. Every pair of seeds is shown as 8 side-by-side frames (A left, B right), in both orders
to cancel the position bias; P(A) comes from the next-token logits of "A" and "B". A seed's
score is its mean win probability against the other seeds of its sample.

    python phase0_pairwise.py scores.jsonl --takes /workspace/rl0/spool/out --out pairwise-0.jsonl --shard 0/8
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_score import load_serve, read_video, sample_info  # noqa: E402

FRAMES = 8
QUESTIONS = {
    "event": "Which video, A or B, shows the actions and interactions described in the caption more completely "
             "and convincingly (for example getting into or out of a vehicle, driving, a punch landing and the "
             "other person falling)?",
    "quality": "Which video, A or B, has better visual quality: sharper and more coherent bodies, faces and "
               "vehicles, and fewer artifacts such as melting, smearing, flicker or objects popping in and out?",
    "identity": "Which video, A or B, shows a protagonist (the character the camera follows) that matches the "
                "identity sheet better: outfit, colours, hair and build?",
}


def side_by_side(a: np.ndarray, b: np.ndarray, index: np.ndarray, serve) -> list[Image.Image]:
    images = []
    for i in index:
        left, right = serve.thumbnail(Image.fromarray(a[i]), 288), serve.thumbnail(Image.fromarray(b[i]), 288)
        canvas = Image.new("RGB", (left.width + right.width + 8, left.height), "white")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + 8, 0))
        images.append(canvas)
    return images


class Judge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.serve = load_serve()
        self.vlm = self.serve.Vlm(Path(args.model), args.device, Path(args.work))
        tokenizer = self.vlm.processor.tokenizer
        self.choices = [tokenizer.convert_tokens_to_ids(t) for t in ("A", "B")]

    @torch.no_grad()
    def prefer_a(self, images: list[Image.Image], text: str) -> tuple[float, float]:
        vlm = self.vlm
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        prompt = vlm.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                   add_generation_prompt=True)
        inputs = vlm.processor(text=[prompt], images=images, return_tensors="pt").to(vlm.device)
        logits = vlm.model(**inputs).logits[0, -1].float()
        mass = float(torch.softmax(logits, 0)[self.choices].sum())
        return float(torch.softmax(logits[self.choices], 0)[0]), mass

    def compare(self, a: np.ndarray, b: np.ndarray, caption: str, oc: Image.Image | None) -> dict[str, tuple[float, float]]:
        index = np.linspace(0, min(len(a), len(b)) - 1, num=FRAMES).round().astype(int)
        frames = side_by_side(a, b, index, self.serve)
        out = {}
        for name, question in QUESTIONS.items():
            if name == "identity" and oc is None:
                continue
            lead = ("The first image is the protagonist identity sheet. The next " if name == "identity" else "The ")
            text = (f"{lead}{FRAMES} images each show the same moment of two generated videos side by side: video A "
                    f"on the left, video B on the right, in time order. Both were generated from this caption:\n"
                    f"{caption}\n\n{question}\nAnswer with a single letter, A or B.")
            images = ([self.serve.thumbnail(oc, 448)] if name == "identity" else []) + frames
            out[name] = self.prefer_a(images, text)
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scores", nargs="+", help="phase0_score rows: which seeds exist for each sample")
    p.add_argument("--takes", nargs="+", required=True, help="directories holding <job id>/take.mp4")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="/data/binghe/h3_proxy/manifests/gta_record_0930_omni_704p_ref640_heldout32_oc.txt")
    p.add_argument("--dataset-root", default="/data/binghe/datasets/gta_record_0930-v0003")
    p.add_argument("--ignore-oc", action="store_true")
    p.add_argument("--seeds", help="comma-separated seeds to compare (default: all)")
    p.add_argument("--model", default="/data/models/MiniMax-H3")
    p.add_argument("--work", default="/workspace/rl0/work")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", default="0/1")
    args = p.parse_args()
    shard, shards = (int(x) for x in args.shard.split("/"))
    keep = {int(s) for s in args.seeds.split(",")} if args.seeds else None

    rows = [json.loads(line) for path in args.scores for line in Path(path).read_text().splitlines() if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source", "serve") == "serve" and (keep is None or row["seed"] in keep):
            groups[row["sample"]].append(row)
    infos = sample_info(Path(args.data))
    order = [s for s in infos if s in groups]
    mine = [s for i, s in enumerate(order) if i % shards == shard]

    def take(job_id: str) -> Path:
        for root in args.takes:
            path = Path(root) / job_id / "take.mp4"
            if path.is_file():
                return path
        raise FileNotFoundError(f"no take.mp4 for {job_id} under {args.takes}")

    judge = Judge(args)
    out = Path(args.out)
    for sample in mine:
        started = time.time()
        members = sorted(groups[sample], key=lambda r: r["seed"])
        videos = {r["seed"]: read_video(take(r["id"])) for r in members}
        info = infos[sample]
        oc_path = Path(args.dataset_root) / info["oc"] if info.get("oc") and not args.ignore_oc else None
        oc = Image.open(oc_path).convert("RGB") if oc_path and oc_path.is_file() else None
        wins: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        masses = []
        for x, y in itertools.combinations(sorted(videos), 2):
            forward = judge.compare(videos[x], videos[y], info["prompt"], oc)
            backward = judge.compare(videos[y], videos[x], info["prompt"], oc)
            for name in forward:
                p_x = (forward[name][0] + 1.0 - backward[name][0]) / 2.0
                wins[name][x].append(p_x)
                wins[name][y].append(1.0 - p_x)
                masses += [forward[name][1], backward[name][1]]
        with out.open("a", encoding="utf-8") as handle:
            for seed in videos:
                row = {"kind": "pairwise", "sample": sample, "seed": seed, "source": "serve",
                       "pair_mass_min": min(masses)}
                for name in QUESTIONS:
                    row[f"pair_{name}"] = float(np.mean(wins[name][seed])) if wins[name].get(seed) else None
                handle.write(json.dumps(row) + "\n")
        print(f"[pairwise] {sample}: {len(videos)} seeds in {time.time() - started:.0f}s, min mass {min(masses):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
