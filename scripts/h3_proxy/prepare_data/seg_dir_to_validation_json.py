#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a validation JSON for ``MiniMaxH3ProxyValidationCallback`` from a flat ``seg_*/`` dataset.

Consumes the same layout as ``seg_dir_to_encode_manifest.py`` and reuses its split handling, so the
val clips here are exactly the ones held out of the encode manifest.

Unlike the encode manifest this emits **absolute** paths. ``ValidationDataset`` resolves only the
media keys it already knows about against the dataset directory, and the proxy and target arrive
through keys it does not know, so a relative path would reach the callback unresolved.

Held-out clips are generated one full diffusion trajectory at a time inside the training loop, so
``--limit`` matters: eight 124-frame clips at 50 steps cost roughly as much as a few hundred
training steps. Four to eight is usually the right size for watching a run.

Usage::

    python scripts/h3_proxy/prepare_data/seg_dir_to_validation_json.py \\
      --root /data/tmp --split val --limit 6 \\
      --out /data/binghe/h3_proxy/validation_val6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seg_dir_to_encode_manifest import (  # noqa: E402
    largest_valid_num_frames, probe_usable_frames, resolve_seg_media, split_ids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root containing seg_*/ and manifests/.")
    p.add_argument("--split", choices=("train", "val", "all"), default="val")
    p.add_argument("--out", required=True, help="Output validation json.")
    p.add_argument("--limit",
                   type=int,
                   default=6,
                   help="Keep at most this many clips, evenly spaced over the split; 0 keeps all.")
    p.add_argument("--num-frames",
                   type=int,
                   default=124,
                   help="The callback's num_frames, checked against each clip's 24-fps budget.")
    p.add_argument("--proxy-stream",
                   choices=("duv", "color", "auto"),
                   default="auto",
                   help="Must match the encode manifest. Nested GTA segs default to proxy/duv.mp4.")
    return p.parse_args()


def evenly_spaced(items: list, limit: int) -> list:
    """Sample ``limit`` items across the whole list rather than taking a prefix.

    Consecutive seg ids tend to be consecutive footage, so a prefix would validate on one scene.
    """
    if limit <= 0 or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root does not exist: {root}")
    keep = split_ids(root, args.split)

    seg_dirs = sorted(path for path in root.glob("seg_*") if path.is_dir())
    if not seg_dirs:
        raise SystemExit(f"No 'seg_*' directories under {root}")

    records: list[dict] = []
    incomplete: list[str] = []
    for seg in seg_dirs:
        if keep is not None and seg.name not in keep:
            continue
        media, reason = resolve_seg_media(seg, proxy_stream=args.proxy_stream)
        if media is None:
            incomplete.append(f"{seg.name}: {reason}")
            continue
        record = {
            "id": seg.name,
            # `caption` is the one key ValidationDataset requires; it aliases to the prompt.
            "caption": media["prompt"],
            "proxy_path": str(media["proxy"]),
            "target_path": str(media["target"]),
            # Read by the metrics evaluator when callbacks.validation.metrics is enabled. Deliberately
            # not `video_path`: the loader would decode that clip on every rank to condition an
            # image-to-video pipeline, which H3 Ref2VA does not use.
            "ref_video": str(media["target"]),
        }
        if media.get("anchor") is not None:
            record["anchor_path"] = str(media["anchor"])
        records.append(record)

    if not records:
        raise SystemExit(f"No complete seg directories survived (scanned {len(seg_dirs)}, "
                         f"{len(incomplete)} incomplete).")

    available = len(records)
    records = evenly_spaced(records, args.limit)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"data": records}, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} validation records -> {out}")
    print(f"  {available} usable clips in split '{args.split}'; kept {len(records)}")
    if incomplete:
        print(f"  {len(incomplete)} incomplete, skipped: {', '.join(line.split(':')[0] for line in incomplete[:10])}"
              f"{' ...' if len(incomplete) > 10 else ''}")

    short = []
    for record in records:
        usable = probe_usable_frames(Path(record["proxy_path"]))
        if usable is not None and usable < args.num_frames:
            short.append((record["id"], usable))
    if short:
        worst = min(item[1] for item in short)
        print(f"  WARNING: {len(short)} clips supply fewer than {args.num_frames} frames at 24 fps "
              f"(shortest {worst}, e.g. {', '.join(item[0] for item in short[:5])}).")
        print(f"  Set callbacks.validation.num_frames to {largest_valid_num_frames(worst)} or drop those clips; "
              "otherwise generation fails once per validation event.")


if __name__ == "__main__":
    main()
