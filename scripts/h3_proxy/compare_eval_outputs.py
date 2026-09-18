#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Did two eval runs produce different video, and were they comparable in the first place?

Byte-identical mp4s from two checkpoints mean the adapter never reached the sampling model: a
diffusion trajectory is chaotic enough that any nonzero weight delta perturbs the output, even one
far too small to see. So hashing the videos is a real test of whether a resume worked -- but only if
nothing except the checkpoint differed, and an mp4 records none of that. Different videos from runs
that read different validation sets, or conditioned at different geometry, prove nothing.

``eval_checkpoint.sh`` writes an ``eval_manifest.json`` next to the videos for this reason. This
compares those first and the bytes second, and refuses to give a verdict on runs that were not
comparable.

Usage::

    python scripts/h3_proxy/compare_eval_outputs.py \\
        /data/binghe/h3_proxy/runs/eval_gta_v2_train_step0/checkpoints \\
        /data/binghe/h3_proxy/runs/eval_gta_v2_train_step161/checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Everything a run records except the two things that are *supposed* to differ.
#
# `sp_size` belongs here for the same reason `geometry` does: sequence parallelism moves where
# attention is split, so two evals at different values are not bit-comparable -- and the caller is
# about to read a pixel difference and attribute all of it to the checkpoint.
MUST_MATCH = ("cache", "geometry", "val_json", "nproc", "nnodes", "sp_size")

VIDEO_RE = re.compile(r"validation_step_(\d+)_inference_steps_(\d+)_rank_(\d+)_video_(\d+)(_compare)?\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", help="An eval output directory (the one holding the mp4s).")
    parser.add_argument("right", help="The other one.")
    parser.add_argument("--force",
                        action="store_true",
                        help="Compare bytes even when the manifests disagree or are missing. The verdict is then "
                        "about the videos only and says nothing about the checkpoints.")
    return parser.parse_args()


def load_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / "eval_manifest.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    # Manifests written before the mesh became configurable record only `nproc`, and the script that
    # wrote them had no way to express anything else: it launched --standalone and set
    # num_gpus == sp_size == nproc. So these are recovered values, not assumed ones, and filling
    # them keeps an old eval comparable against a new one that happens to agree instead of
    # reporting `None != 8` at it.
    manifest.setdefault("nnodes", 1)
    manifest.setdefault("sp_size", manifest.get("nproc"))
    return manifest


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def index_videos(directory: Path) -> dict[tuple[str, str], Path]:
    """Keyed by what identifies the *sample*, not the step: the clip index and whether it is a panel.

    The step is in the filename and is exactly what differs between two runs, so it cannot be part
    of the key.
    """
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.glob("validation_step_*.mp4")):
        match = VIDEO_RE.search(path.name)
        if match:
            found[(match.group(4), match.group(5) or "")] = path
    return found


def compare_manifests(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        print("No eval_manifest.json in one or both directories, so what produced these videos is unrecorded.\n"
              "Runs from before the manifest existed cannot be verified as comparable; re-run with\n"
              "eval_checkpoint.sh, or pass --force to hash the bytes anyway.")
        return False

    print(f"steps: {left.get('step')} vs {right.get('step')}")
    agreed = True
    for field in MUST_MATCH:
        if left.get(field) == right.get(field):
            print(f"  ok        {field}: {left.get(field)!r}")
        else:
            agreed = False
            print(f"  MISMATCH  {field}: {left.get(field)!r} != {right.get(field)!r}")
    if left.get("step") == right.get("step"):
        print(f"  NOTE: both ran at step {left.get('step')}, so identical video is the expected result "
              f"and proves nothing either way.")
    return agreed


def main() -> None:
    args = parse_args()
    left_dir, right_dir = Path(args.left).expanduser(), Path(args.right).expanduser()
    for directory in (left_dir, right_dir):
        if not directory.is_dir():
            raise SystemExit(f"Not a directory: {directory}")

    comparable = compare_manifests(load_manifest(left_dir), load_manifest(right_dir))
    if not comparable and not args.force:
        raise SystemExit(1)

    left_videos, right_videos = index_videos(left_dir), index_videos(right_dir)
    shared = sorted(set(left_videos) & set(right_videos))
    if not shared:
        raise SystemExit(f"No video indices in common ({len(left_videos)} vs {len(right_videos)} found), so there "
                         "is nothing to compare.")

    identical, differing = [], []
    print(f"\n{len(shared)} samples in common:")
    for key in shared:
        label = f"video_{key[0]}{key[1]}"
        if digest(left_videos[key]) == digest(right_videos[key]):
            identical.append(label)
            print(f"  IDENTICAL  {label}")
        else:
            differing.append(label)
            print(f"  differs    {label}")

    print()
    if not identical:
        print(f"All {len(differing)} differ. The checkpoints sampled differently, so the adapter reached the\n"
              "model. Whether the difference is visible is a separate question -- a delta of 1e-4 relative to\n"
              "the base weights changes every byte while looking the same.")
        return
    if not differing:
        raise SystemExit(f"All {len(identical)} are byte-identical. Two checkpoints cannot sample identically "
                         "unless the weights that differ never reached the sampling model, so the resume loaded "
                         "nothing. Check the resume's 'global lora_B norm' line, and that the step in these "
                         "filenames is the step you asked for.")
    raise SystemExit(f"{len(differing)} differ but {len(identical)} are byte-identical "
                     f"({', '.join(identical)}). A partial split is not a small-delta result -- that would "
                     "change every sample. Suspect a per-sample failure instead: a record whose conditioning "
                     "could not be read, or a rank that wrote one file and skipped another.")


if __name__ == "__main__":
    main()
