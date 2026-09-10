#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a validation JSON for ``MiniMaxH3ProxyValidationCallback`` from a ``clip_*/`` dataset.

Consumes the same layout as ``clip_dir_to_encode_manifest.py`` and reuses its episode-level split,
so the clips here are exactly the ones held out of the encode manifest. Pass the same
``--val-episodes`` to both or the two disagree about what was held out.

Paths are **absolute** on purpose. ``ValidationDataset`` resolves only the media keys it already
knows about against the dataset directory, and the proxy, anchor and target arrive through keys it
does not know, so a relative path would reach the callback unresolved.

Unlike the ``seg_*`` variant this emits ``anchor_path``. These clips ship a lossless
``target/anchor.png``, and the callback otherwise falls back to decoding frame 0 out of the
lossy ``rgb.mp4`` -- which is not the frame the training cache was encoded from.

Held-out clips are generated one full diffusion trajectory at a time inside the training loop, so
``--limit`` matters: six 124-frame clips at 50 steps cost roughly as much as a few hundred training
steps. Four to eight is usually the right size for watching a run.

Usage::

    python scripts/h3_proxy/prepare_data/clip_dir_to_validation_json.py \\
      --root /data/binghe/datasets/ABot-sub-2000-clips \\
      --val-episodes 24 --limit 6 \\
      --out /data/binghe/h3_proxy/abot_validation_val6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_dir_to_encode_manifest import (  # noqa: E402
    CLIP_PATTERN, check_report, episode_of, evenly_spaced, read_prompt, report_window_scope,
)
from seg_dir_to_encode_manifest import largest_valid_num_frames, probe_usable_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root containing clip_*/ directories.")
    p.add_argument("--out", required=True, help="Output validation json.")
    p.add_argument("--split", choices=("train", "val", "all"), default="val")
    p.add_argument("--val-episodes",
                   type=int,
                   default=24,
                   help="Must match the value passed to clip_dir_to_encode_manifest.py.")
    p.add_argument("--limit",
                   type=int,
                   default=6,
                   help="Keep at most this many clips, evenly spaced over the split; 0 keeps all.")
    p.add_argument("--num-frames", type=int, default=124, help="The callback's num_frames, checked against each clip.")
    p.add_argument("--target-height", type=int, default=768)
    p.add_argument("--target-width", type=int, default=1344)
    p.add_argument("--proxy-height", type=int, default=192)
    p.add_argument("--proxy-width", type=int, default=336)
    p.add_argument("--prompt-fallback", default="", help="Prompt for clips with no prompt.txt.")
    p.add_argument("--allow-episode-caption",
                   action="store_true",
                   help="Fall back to annotations/caption.json when a clip has no prompt.txt. Off by default: it "
                   "describes the whole 60-second episode, so validating against it scores the model on events the "
                   "clip never contains.")
    p.add_argument("--no-episode-caption",
                   action="store_true",
                   help="Accepted and redundant; this is the default now. Use --allow-episode-caption to opt back in.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root does not exist: {root}")

    clip_dirs = sorted(path for path in root.glob("clip_*") if path.is_dir() and CLIP_PATTERN.match(path.name))
    if not clip_dirs:
        raise SystemExit(f"No 'clip_<episode>_<window>' directories under {root}")

    episodes = sorted({episode_of(path.name) for path in clip_dirs})
    val_episodes = set(evenly_spaced(episodes, args.val_episodes))
    if args.split == "train":
        keep_episodes = set(episodes) - val_episodes
    elif args.split == "val":
        keep_episodes = val_episodes
    else:
        keep_episodes = set(episodes)

    records: list[dict] = []
    rejected: list[str] = []
    for clip in clip_dirs:
        if episode_of(clip.name) not in keep_episodes:
            continue
        target, anchor, duv = (clip / "target" / "rgb.mp4", clip / "target" / "anchor.png", clip / "proxy" / "duv.mp4")
        report_file = clip / "clip_report.json"
        if any(not path.is_file() for path in (target, anchor, duv, report_file)):
            rejected.append(clip.name)
            continue
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rejected.append(clip.name)
            continue
        if check_report(report, args) is not None:
            rejected.append(clip.name)
            continue
        prompt, _ = read_prompt(clip, use_episode_caption=args.allow_episode_caption)
        prompt = prompt or args.prompt_fallback.strip()
        if not prompt:
            rejected.append(clip.name)
            continue
        records.append({
            "id": clip.name,
            # `caption` is the one key ValidationDataset requires; it aliases to the prompt.
            "caption": prompt,
            "proxy_path": str(duv),
            # The lossless frame the cache was encoded from, rather than frame 0 of the lossy mp4.
            "anchor_path": str(anchor),
            "target_path": str(target),
            # Read by the metrics evaluator when callbacks.validation.metrics is enabled. Deliberately
            # not `video_path`: the loader would decode that clip on every rank to condition an
            # image-to-video pipeline, which H3 Ref2VA does not use.
            "ref_video": str(target),
        })

    if not records:
        raise SystemExit(f"No usable clips in split '{args.split}' (scanned {len(clip_dirs)}, "
                         f"{len(rejected)} rejected).")

    available = len(records)
    available_episodes = {episode_of(record["id"]) for record in records}
    # Consecutive windows are consecutive footage from one episode, so a prefix would validate on
    # five views of the same minute.
    records = evenly_spaced(records, args.limit)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"data": records}, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} validation records -> {out}")
    print(f"  split '{args.split}' covers {len(keep_episodes)} episode(s); {available} usable clips over "
          f"{len(available_episodes)} of them; kept {len(records)}")
    print(f"  episodes: {', '.join(sorted({episode_of(record['id']) for record in records}))}")
    # `caption` is this file's prompt key; report_window_scope reads `prompt`.
    report_window_scope([{"prompt": record["caption"]} for record in records])
    if rejected:
        print(f"  {len(rejected)} rejected: {', '.join(rejected[:10])}{' ...' if len(rejected) > 10 else ''}")
    print(f"  set callbacks.validation.proxy_height/proxy_width to {args.proxy_height}/{args.proxy_width}, matching "
          "the encoder; otherwise the proxy is presented on the full canvas instead of its own grid")

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
