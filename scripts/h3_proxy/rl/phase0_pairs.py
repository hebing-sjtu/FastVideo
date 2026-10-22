"""Blind A/B clips for a human check of a reward: its best and worst seed of every sample.

Each clip is ``proxy | A | B`` at 640x352 per panel; which side holds the reward's pick is drawn
at random and kept only in ``key-<reward>.json``, so the rater sees neither seeds nor scores.

    python phase0_pairs.py scores.jsonl --takes /workspace/rl0/spool/out --eval-dir <dir> --out pairs/ --reward depth
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_report import grouped, merge_pairwise, with_composite  # noqa: E402
from phase0_score import eval_slots  # noqa: E402


def clip(proxy: Path, left: Path, right: Path, out: Path) -> None:
    scale = "scale=640:352:flags=bicubic,setsar=1"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(proxy), "-i", str(left), "-i", str(right),
                    "-filter_complex", f"[0:v]{scale}[p];[1:v]{scale}[a];[2:v]{scale}[b];[p][a][b]hstack=3",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scores", nargs="+")
    p.add_argument("--takes", nargs="+", required=True, help="directories holding <job id>/take.mp4")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reward", default="composite")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rows = [json.loads(line) for path in args.scores for line in Path(path).read_text().splitlines() if line.strip()]
    groups = grouped(merge_pairwise(rows))
    with_composite(groups)
    slots = eval_slots(Path(args.eval_dir))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    key = {}
    for index, (sample, rs) in enumerate(sorted(groups.items())):
        if any(r.get(args.reward) is None for r in rs):
            continue
        ranked = sorted(rs, key=lambda r: r[args.reward])
        worst, best = ranked[0], ranked[-1]
        best_left = rng.random() < 0.5
        left, right = (best, worst) if best_left else (worst, best)
        def take(row: dict) -> Path:
            for root in args.takes:
                if (Path(root) / row["id"] / "take.mp4").is_file():
                    return Path(root) / row["id"] / "take.mp4"
            raise FileNotFoundError(row["id"])

        name = f"{args.reward}-{index:02d}.mp4"
        clip(slots[sample] / "proxy.mp4", take(left), take(right), out / name)
        key[name] = {"sample": sample, "reward": args.reward, "pick": "A" if best_left else "B",
                     "A_seed": left["seed"], "B_seed": right["seed"],
                     "A_score": left[args.reward], "B_score": right[args.reward]}
    (out / f"key-{args.reward}.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"{len(key)} pairs in {out}; A is the left of the two generated panels")


if __name__ == "__main__":
    main()
