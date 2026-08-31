#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert kof_1k_pre_q_060 RGB jsonl → encode_v2v_depth_samples manifest.

Does not re-split. Reads the pack's train/val jsonl and emits one encode
manifest line per clip:

    {"name", "target", "source", "prompt"}

Paths are relative to the pack root (``--root``), matching
``datasets/<id>/video_{src,target}.mp4``.

Usage::

    python scripts/v2v_depth/prepare_data/kof_jsonl_to_encode_manifest.py \\
      --root /data/raw/kof_1k_pre_q_060_rgb \\
      --split train \\
      --out /data/raw/kof_1k_pre_q_060_rgb/encode_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Pack root containing data/ and datasets/.")
    p.add_argument("--split", choices=("train", "val", "all"), default="train")
    p.add_argument("--out", required=True, help="Output encode manifest jsonl.")
    p.add_argument("--check-files", action="store_true", help="Skip rows whose mp4/prompt are missing.")
    # The two encoders name the low-poly clip differently: for Wan V2V it is the `source` the model
    # re-renders, for H3 it is the `proxy` that rides a reference slot. Same file either way.
    p.add_argument("--source-key",
                   choices=("source", "proxy"),
                   default="source",
                   help="Field name for the low-poly clip: 'source' for encode_v2v_depth_samples, "
                   "'proxy' for encode_proxy_samples.")
    return p.parse_args()


def jsonl_path(root: Path, split: str) -> Path:
    data = root / "data"
    mapping = {
        "train": data / "kof_1k_pre_q_060_train.jsonl",
        "val": data / "kof_1k_pre_q_060_val.jsonl",
        "all": data / "kof_1k_pre_q_060.jsonl",
    }
    path = mapping[split]
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    return path


def clip_id_of(row: dict) -> str:
    for key in ("id", "clip_id", "name", "sample_id"):
        value = row.get(key)
        if value:
            return str(value).strip().strip("/")
    raise KeyError(f"row has no id-like field: keys={sorted(row)}")


def prompt_of(row: dict, root: Path, clip_id: str) -> str:
    for key in ("prompt", "caption", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    prompt_file = root / "datasets" / clip_id / "prompt.txt"
    if prompt_file.is_file():
        return prompt_file.read_text(encoding="utf-8").strip()
    return ""


def resolve_videos(row: dict, root: Path, clip_id: str) -> tuple[str, str]:
    """Return (target, source) paths relative to root."""
    # Prefer explicit fields if the pack jsonl already carries them.
    target = row.get("video_target") or row.get("target") or row.get("teacher_video")
    source = row.get("video_src") or row.get("source") or row.get("src_video")
    if target and source:
        return str(target), str(source)
    base = Path("datasets") / clip_id
    return str(base / "video_target.mp4"), str(base / "video_src.mp4")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    src = jsonl_path(root, args.split)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(src, encoding="utf-8") as hin, open(out, "w", encoding="utf-8") as hout:
        for line in hin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            clip_id = clip_id_of(row)
            target, source = resolve_videos(row, root, clip_id)
            prompt = prompt_of(row, root, clip_id)
            # Flat cache name: list_dir only picks *.pt in one directory.
            name = clip_id.replace("/", "__")

            if args.check_files:
                ok = (root / target).is_file() and (root / source).is_file()
                if not ok:
                    skipped += 1
                    print(f"SKIP missing media: {clip_id}", flush=True)
                    continue

            entry = {
                "name": name,
                "target": target,
                args.source_key: source,
                "prompt": prompt,
                "id": clip_id,
            }
            hout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows → {out} (skipped={skipped}, split={args.split})")


if __name__ == "__main__":
    main()
