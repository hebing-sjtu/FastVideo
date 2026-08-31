#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scan a flat ``seg_*/`` dataset into an ``encode_proxy_samples`` manifest.

Layout consumed::

    <root>/
      seg_0000/
        video_src.mp4        low-poly render -> the Ref2VA video reference ("proxy")
        video_target.mp4     high-quality clip -> the denoising target
        prompt.txt           the only text that enters training
        metadata.json        identity; read only for cross-checking the id
        .minimax_h3/         teacher provenance, ignored here
      manifests/
        <prefix>_train.jsonl
        <prefix>_val.jsonl

The seg directories are authoritative for *what exists*; the manifests are authoritative for
*which split a clip belongs to*. So this scans the directories and, unless ``--split all``, keeps
only the ids the split manifest names. Ids are matched by the ``seg_\\d+`` pattern anywhere in each
manifest row, which avoids depending on that file's field names.

Every emitted row is checked for its three required files. A seg directory missing any of them is
reported and skipped rather than left to fail one-by-one inside the encoder.

Frame budget
------------
The encoder resamples to H3's fixed 24 fps *before* trimming, so a clip's usable length is
``floor(num_source_frames * 24 / source_fps)``, not its raw frame count. A 124-frame 30-fps clip
yields 99 frames and would fail ``--num-frames 124`` — every clip in the set, after the model is
loaded. This probes container metadata (no decoding) on a sample of clips and prints the largest
valid ``--num-frames`` the set supports.

Usage::

    python scripts/h3_proxy/prepare_data/seg_dir_to_encode_manifest.py \\
      --root /data/tmp --split train \\
      --out /workspace/h3_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

# The H3 causal VAE consumes frames in groups of 17 after a 5-frame head.
FRAME_MULTIPLE = 17
FRAME_OFFSET = 5
MINIMAX_H3_FPS = 24

SEG_PATTERN = re.compile(r"seg_\d+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root containing seg_*/ and manifests/.")
    p.add_argument("--split", choices=("train", "val", "all"), default="train")
    p.add_argument("--out", required=True, help="Output encode manifest jsonl.")
    # `proxy` for encode_proxy_samples (H3), `source` for encode_v2v_depth_samples (Wan V2V). The
    # same video_src.mp4 either way; only the field name differs.
    p.add_argument("--source-key", choices=("proxy", "source"), default="proxy")
    p.add_argument("--probe-limit",
                   type=int,
                   default=32,
                   help="Clips to probe for the frame budget; 0 probes all, which is slower on FUSE mounts.")
    return p.parse_args()


def largest_valid_num_frames(usable: int) -> int:
    """Largest ``n <= usable`` with ``n %% 17 == 5``, or 0 if none fits."""
    if usable < FRAME_OFFSET:
        return 0
    return (usable - FRAME_OFFSET) // FRAME_MULTIPLE * FRAME_MULTIPLE + FRAME_OFFSET


def read_split(root: Path, split: str) -> tuple[set[str], list[Path]]:
    """Ids mentioned by one split's manifests, and the files they came from."""
    directory = root / "manifests"
    if not directory.is_dir():
        raise SystemExit(f"--split {split} needs {directory}, which does not exist. Use --split all to take every "
                         "seg directory.")
    matches = sorted(directory.glob(f"*_{split}.jsonl"))
    ids: set[str] = set()
    for path in matches:
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Match against the raw line rather than parsed fields: the split file's schema is
                # not part of this contract, only the seg ids it mentions.
                ids.update(SEG_PATTERN.findall(line))
    return ids, matches


def split_ids(root: Path, split: str) -> set[str] | None:
    """Ids named by the split manifest, or None to accept every seg directory.

    Also cross-checks the sibling split. Two splits that share ids are not a situation any caller
    asked for, and it is invisible downstream: training would simply include the held-out clips and
    report a validation loss on data it had already fitted.
    """
    if split == "all":
        return None
    ids, matches = read_split(root, split)
    if not matches:
        directory = root / "manifests"
        listing = ", ".join(path.name for path in sorted(directory.iterdir())) or "(empty)"
        raise SystemExit(f"No '*_{split}.jsonl' in {directory}. Found: {listing}")
    if not ids:
        raise SystemExit(f"{', '.join(str(path) for path in matches)} mention no 'seg_NNNN' ids.")
    print(f"Split '{split}': {len(ids)} ids from {', '.join(path.name for path in matches)}")

    sibling = "val" if split == "train" else "train"
    other, other_matches = read_split(root, sibling)
    if other_matches:
        shared = ids & other
        if shared:
            print(f"  WARNING: {len(shared)} of these ids are also in '{sibling}' "
                  f"({', '.join(sorted(shared)[:8])}{' ...' if len(shared) > 8 else ''}).")
            print("  train and val are not disjoint, so every shared clip would be both trained on and "
                  "validated on. Fix the split files before encoding.")
    return ids


def probe_usable_frames(path: Path) -> int | None:
    """Frames this clip contributes at 24 fps, from container metadata alone."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            rate = stream.average_rate or getattr(stream, "guessed_rate", None)
            if not rate:
                return None
            count = int(stream.frames or 0)
            if count <= 0:
                # Containers written without a frame count still carry a duration.
                if stream.duration and stream.time_base:
                    count = int(float(stream.duration * stream.time_base) * float(rate))
                if count <= 0:
                    return None
            return int(count * MINIMAX_H3_FPS / float(rate))
    except (OSError, ValueError, StopIteration):
        return None


def report_frame_budget(rows: list[dict], root: Path, limit: int) -> None:
    sample = rows if limit <= 0 else rows[::max(1, len(rows) // limit)][:limit]
    budgets: list[tuple[str, int]] = []
    for row in sample:
        for key in ("target", "proxy", "source"):
            relative = row.get(key)
            if not relative:
                continue
            usable = probe_usable_frames(root / str(relative))
            if usable is not None:
                budgets.append((f"{row['name']}/{key}", usable))
    if not budgets:
        print("Frame budget: could not probe any clip (PyAV missing or metadata absent). Verify --num-frames by "
              "encoding a few clips before launching the full set.")
        return
    name, worst = min(budgets, key=lambda item: item[1])
    print(f"Frame budget over {len(budgets)} probed streams: shortest is {worst} frames at 24 fps ({name}).")
    recommended = largest_valid_num_frames(worst)
    if recommended:
        print(f"  Largest valid --num-frames for this sample: {recommended}")
    else:
        print(f"  No valid --num-frames fits {worst} frames; the shortest clip is unusable.")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root does not exist: {root}")
    keep = split_ids(root, args.split)

    seg_dirs = sorted(path for path in root.glob("seg_*") if path.is_dir())
    if not seg_dirs:
        raise SystemExit(f"No 'seg_*' directories under {root}")

    rows: list[dict] = []
    skipped_split = 0
    incomplete: list[str] = []
    for seg in seg_dirs:
        if keep is not None and seg.name not in keep:
            skipped_split += 1
            continue
        target, source, prompt_file = (seg / "video_target.mp4", seg / "video_src.mp4", seg / "prompt.txt")
        missing = [path.name for path in (target, source, prompt_file) if not path.is_file()]
        if missing:
            incomplete.append(f"{seg.name}: missing {', '.join(missing)}")
            continue
        prompt = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            incomplete.append(f"{seg.name}: prompt.txt is empty")
            continue
        rows.append({
            "name": seg.name,
            "target": str(target.relative_to(root)),
            args.source_key: str(source.relative_to(root)),
            "prompt": prompt,
            "id": seg.name,
        })

    if not rows:
        raise SystemExit(f"No complete seg directories survived (scanned {len(seg_dirs)}, "
                         f"{len(incomplete)} incomplete, {skipped_split} out of split).")

    missing_from_disk = sorted(keep - {row["name"] for row in rows}) if keep is not None else []

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {out}")
    # State the total explicitly: without it the reader has to add the lines below to notice that
    # the split files reference more clips than the directory tree holds.
    print(f"  {len(seg_dirs)} seg directories on disk, {len(rows)} of them usable in split '{args.split}'")
    if skipped_split:
        print(f"  {skipped_split} seg directories are not in split '{args.split}'")
    if incomplete:
        print(f"  {len(incomplete)} incomplete, skipped:")
        for line in incomplete[:10]:
            print(f"    {line}")
        if len(incomplete) > 10:
            print(f"    ... and {len(incomplete) - 10} more")
    if missing_from_disk:
        print(f"  {len(missing_from_disk)} ids in the split manifest have no usable seg directory: "
              f"{', '.join(missing_from_disk[:10])}{' ...' if len(missing_from_disk) > 10 else ''}")
    report_frame_budget(rows, root, args.probe_limit)


if __name__ == "__main__":
    main()
