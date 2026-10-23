"""Score chained long takes per segment window, and test GT-free rewards against late-segment fidelity.

A take of three 124-frame clips with 85-frame stride is 294 frames; its windows are the frames each
clip adds: [0, 124), [124, 209), [209, 294). Per window:

GT-referenced (only to judge the rewards): ``psnr_k``, ``dino_k`` (DINOv2 cosine to the target),
``depth_k`` (negative aligned relative disparity error, as in phase 0).
GT-free (candidate rewards): ``anchor_k`` (DINOv2 cosine to the take's first frame: does the scene and
character persist), ``sharp_k`` (variance of the Laplacian), ``sat_k`` / ``luma_k`` (mean saturation
and brightness), and per take ``seam`` (frame change at the two clip joins over the nearby median).

    python long0_score.py score --spool /workspace/long0/spool --takes /workspace/long0/takes \\
        --out /workspace/long0/scores.jsonl --shard 0/8
    python long0_score.py report /workspace/long0/scores.jsonl --out long0.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

WINDOWS = ((0, 124), (124, 209), (209, 294))
STRIDE, OVERLAP = 85, 39
SAMPLES = 8


def stitched_target(take_dir: Path, segments: int, height: int, work: Path) -> Path:
    """The real take under the generated one: clip 0, then each later clip past its overlap."""
    from phase0_score import read_video

    out = work / "targets" / f"{take_dir.name}.mp4"
    if out.is_file():
        return out
    parts = [read_video(take_dir / "clips" / f"s{k:02d}" / "target.mp4")[(0 if k == 0 else OVERLAP):]
             for k in range(segments)]
    frames = np.concatenate(parts)
    top = (frames.shape[1] - height) // 2
    frames = np.ascontiguousarray(frames[:, top:top + height])
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                             f"{w}x{h}", "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt",
                             "yuv420p", str(out)], stdin=subprocess.PIPE)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed writing {out}")
    return out


def sharpness(frames: np.ndarray) -> float:
    grey = frames.astype(np.float32).mean(-1)
    lap = (-4 * grey[:, 1:-1, 1:-1] + grey[:, :-2, 1:-1] + grey[:, 2:, 1:-1] + grey[:, 1:-1, :-2]
           + grey[:, 1:-1, 2:])
    return float(lap.var(axis=(1, 2)).mean())


def saturation_luma(frames: np.ndarray) -> tuple[float, float]:
    f = frames.astype(np.float32) / 255.0
    high, low = f.max(-1), f.min(-1)
    return float(((high - low) / np.maximum(high, 1e-6)).mean()), float(f.mean())


def seam_jump(frames: np.ndarray, joins: list[int]) -> float:
    grey = frames[:, ::4, ::4].astype(np.float32).mean(-1)
    change = np.abs(np.diff(grey, axis=0)).mean((1, 2))
    ratios = []
    for join in joins:
        near = [change[t] for t in range(join - 7, join + 6) if 0 <= t < len(change) and t != join - 1]
        ratios.append(change[join - 1] / max(float(np.median(near)), 1e-6))
    return float(np.mean(ratios))


def score(args: argparse.Namespace) -> None:
    import torch
    from phase0_score import Scorer, aligned_error, per_frame, picks, read_video

    args.no_vlm = True
    scorer = Scorer(args)
    spool, takes, out, work = Path(args.spool), Path(args.takes), Path(args.out), Path(args.work)
    shard, shards = (int(x) for x in args.shard.split("/"))
    seen = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.is_file() else set()
    for index, done in enumerate(sorted(spool.glob("done/*.json"))):
        if index % shards != shard or done.stem in seen:
            continue
        payload = json.loads(done.read_text())
        if payload.get("error"):
            continue
        seed_text, _, take_name = done.stem.split("-", 2)
        take = read_video(Path(payload["take"]))
        target_path = stitched_target(takes / take_name, len(payload["segments"]), take.shape[1], work)
        target = read_video(target_path)
        n = min(len(take), len(target))
        psnr = per_frame(Path(payload["take"]), target_path, "psnr")
        first = scorer.embed(take[:1])
        row = {"id": done.stem, "take": take_name, "seed": int(seed_text[1:]), "frames": n}
        for k, (start, stop) in enumerate(WINDOWS):
            stop = min(stop, n)
            index_k = start + picks(SAMPLES, stop - start)
            gen_embed, tgt_embed = scorer.embed(take[index_k]), scorer.embed(target[index_k])
            row[f"psnr_{k}"] = statistics.mean(psnr[start:stop])
            row[f"dino_{k}"] = float((gen_embed * tgt_embed).sum(-1).mean())
            row[f"depth_{k}"] = -aligned_error(scorer.disparity(take[index_k]), scorer.disparity(target[index_k]))
            row[f"anchor_{k}"] = float((gen_embed @ first.T).mean())
            row[f"sharp_{k}"] = sharpness(take[start:stop:4])
            row[f"sat_{k}"], row[f"luma_{k}"] = saturation_luma(take[start:stop:4])
        row["seam"] = seam_jump(take[:n], [119, 204])
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"[long0] {done.stem} psnr {row['psnr_0']:.2f}/{row['psnr_1']:.2f}/{row['psnr_2']:.2f} "
              f"anchor {row['anchor_0']:.3f}/{row['anchor_2']:.3f}", flush=True)
        torch.cuda.empty_cache()


REWARDS = {
    "anchor_keep": lambda r: r["anchor_2"] - r["anchor_0"],
    "anchor_late": lambda r: r["anchor_2"],
    "sharp_keep": lambda r: np.log(r["sharp_2"] / r["sharp_0"]),
    "color_keep": lambda r: -abs(r["sat_2"] - r["sat_0"]) - abs(r["luma_2"] - r["luma_0"]),
    "seam": lambda r: -r["seam"],
}
TARGETS = ("psnr_2", "dino_2", "depth_2")


def ranks(values: list[float]) -> np.ndarray:
    return np.argsort(np.argsort(values)).astype(float)


def report(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for path in args.scores for line in Path(path).read_text().splitlines()]
    takes: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        takes[row["take"]].append(row)
    takes = {k: v for k, v in takes.items() if len(v) >= 3}
    lines = [f"# Long-take check: {len(takes)} takes, {len(rows)} rollouts\n", "## Drift by window (mean over all)\n",
             "| metric | clip 1 [0,124) | clip 2 [124,209) | clip 3 [209,294) |", "|---|---|---|---|"]
    for name in ("psnr", "dino", "depth", "anchor", "sharp", "sat", "luma"):
        lines.append(f"| {name} | " + " | ".join(f"{np.mean([r[f'{name}_{k}'] for r in rows]):.4g}" for k in range(3))
                     + " |")
    lines += ["", "## Seed headroom on clip 3 (within-take std, mean over takes)\n"]
    for name in TARGETS:
        lines.append(f"- {name}: {np.mean([np.std([r[name] for r in v]) for v in takes.values()]):.4f}; "
                     f"best seed - mean {np.mean([max(r[name] for r in v) - np.mean([r[name] for r in v]) for v in takes.values()]):.4f}")
    lines += ["", "## GT-free reward vs clip-3 fidelity: within-take Spearman, and best-of-N gain over a random seed\n",
              "| reward | " + " | ".join(f"rho {t}" for t in TARGETS) + " | " + " | ".join(f"gain {t}" for t in TARGETS)
              + " |", "|---" * (1 + 2 * len(TARGETS)) + "|"]
    for name, fn in REWARDS.items():
        rhos, gains = defaultdict(list), defaultdict(list)
        for v in takes.values():
            reward = [fn(r) for r in v]
            best = v[int(np.argmax(reward))]
            for t in TARGETS:
                values = [r[t] for r in v]
                if np.std(values) > 0 and np.std(reward) > 0:
                    rhos[t].append(np.corrcoef(ranks(reward), ranks(values))[0, 1])
                gains[t].append(best[t] - np.mean(values))
        lines.append(f"| {name} | " + " | ".join(f"{np.mean(rhos[t]):+.2f}" for t in TARGETS) + " | "
                     + " | ".join(f"{np.mean(gains[t]):+.4f}" for t in TARGETS) + " |")
    oracle = {t: np.mean([max(r[t] for r in v) - np.mean([r[t] for r in v]) for v in takes.values()]) for t in TARGETS}
    lines.append("| (oracle) | | | | " + " | ".join(f"{oracle[t]:+.4f}" for t in TARGETS) + " |")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--spool", required=True)
    s.add_argument("--takes", required=True, help="split_proxy_take.py output dirs, named like the jobs' takes")
    s.add_argument("--out", required=True)
    s.add_argument("--work", default="/workspace/long0/work")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--model", default="/data/models/MiniMax-H3")
    s.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Base-hf")
    s.add_argument("--dino-model", default="facebook/dinov2-base")
    s.add_argument("--shard", default="0/1")
    r = sub.add_parser("report")
    r.add_argument("scores", nargs="+")
    r.add_argument("--out", default="")
    args = p.parse_args()
    score(args) if args.command == "score" else report(args)


if __name__ == "__main__":
    main()
