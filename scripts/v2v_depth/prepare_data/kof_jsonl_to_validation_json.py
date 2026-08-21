#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert kof_1k_pre_q_060 RGB jsonl → ValidationCallback dataset json.

Emits the ``{"data": [...]}`` shape ``fastvideo.dataset.ValidationDataset``
expects, one record per clip::

    {"caption":            prompt text,
     "control_video_path": low-poly source clip  (the video-to-video condition),
     "ref_video":          teacher target clip   (ground truth for metrics),
     "num_frames":         frames to generate,
     "crop_top": ..., "crop_bottom": ..., "crop_left": ..., "crop_right": ...}

The crop fractions must match the ones ``encode_v2v_depth_samples.py`` used to
build the training cache. They travel per record because the crop is a property
of how the dataset was encoded, not of the model, and validation has to
reproduce it exactly or the control channel shifts under the checkpoint.

Paths are written absolute: ``ValidationDataset`` resolves relative paths
against the json file's own directory, which is rarely where the pack lives.

Note the deliberate absence of ``video_path``: that key makes the callback treat
the clip's first frame as an image-to-video prompt. Video-to-video conditions on
the whole source clip through ``control_video_path`` instead.

Usage::

    python scripts/v2v_depth/prepare_data/kof_jsonl_to_validation_json.py \\
      --root /data/raw/kof_1k_pre_q_060_rgb \\
      --split val --limit 8 \\
      --num-frames 81 \\
      --crop-top 0.175 --crop-bottom 0.105 \\
      --out /data/raw/kof_1k_pre_q_060_rgb/validation_val8.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kof_jsonl_to_encode_manifest import clip_id_of, jsonl_path, prompt_of, resolve_videos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Pack root containing data/ and datasets/.")
    p.add_argument("--split", choices=("train", "val", "all"), default="val")
    p.add_argument("--out", required=True, help="Output validation json.")
    p.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Keep at most this many clips. Validation runs inside the training loop, so a "
        "handful is usually the right budget; 0 keeps everything.",
    )
    p.add_argument("--num-frames", type=int, default=81, help="Must satisfy (n - 1) %% 4 == 0 for the Wan VAE.")
    p.add_argument("--crop-top", type=float, default=0.0)
    p.add_argument("--crop-bottom", type=float, default=0.0)
    p.add_argument("--crop-left", type=float, default=0.0)
    p.add_argument("--crop-right", type=float, default=0.0)
    p.add_argument(
        "--depth-subdir",
        default=None,
        help="Optional depth clip location relative to the pack root, with '{id}' standing in "
        "for the clip id, e.g. 'depth/{id}/depth.mp4'. Omit until depth renders exist.",
    )
    p.add_argument("--depth-wide-subdir", default=None, help="Same as --depth-subdir for the wide-FOV render.")
    p.add_argument("--depth-near", type=float, default=None)
    p.add_argument("--depth-far", type=float, default=None)
    args = p.parse_args()
    if (args.num_frames - 1) % 4 != 0:
        raise SystemExit(f"--num-frames must satisfy (n - 1) % 4 == 0 for the Wan VAE, got {args.num_frames}")
    for name in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 0.5:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 0.5), got {value}")
    if args.depth_wide_subdir and not args.depth_subdir:
        raise SystemExit("--depth-wide-subdir needs --depth-subdir")
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    src = jsonl_path(root, args.split)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped = 0
    with open(src, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if args.limit and len(records) >= args.limit:
                break

            row = json.loads(line)
            clip_id = clip_id_of(row)
            target, source = resolve_videos(row, root, clip_id)
            target_path, source_path = root / target, root / source
            if not (target_path.is_file() and source_path.is_file()):
                skipped += 1
                print(f"SKIP missing media: {clip_id}", flush=True)
                continue

            record = {
                "caption": prompt_of(row, root, clip_id),
                "control_video_path": str(source_path),
                "ref_video": str(target_path),
                "num_frames": int(args.num_frames),
                "crop_top": float(args.crop_top),
                "crop_bottom": float(args.crop_bottom),
                "crop_left": float(args.crop_left),
                "crop_right": float(args.crop_right),
                "id": clip_id,
            }

            for flag, key in (
                (args.depth_subdir, "depth_video_path"),
                (args.depth_wide_subdir, "depth_wide_video_path"),
            ):
                if not flag:
                    continue
                depth_path = root / flag.format(id=clip_id)
                if not depth_path.exists():
                    raise SystemExit(f"depth clip not found for {clip_id}: {depth_path}")
                record[key] = str(depth_path)
            if args.depth_near is not None:
                record["depth_near"] = float(args.depth_near)
            if args.depth_far is not None:
                record["depth_far"] = float(args.depth_far)

            records.append(record)

    if not records:
        raise SystemExit(f"No usable clips found in {src}")

    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"data": records}, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} validation records → {out} (skipped={skipped}, split={args.split})")


if __name__ == "__main__":
    main()
