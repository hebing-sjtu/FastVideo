#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scan an ``ABot-sub-2000-clips`` style dataset into an ``encode_proxy_samples`` manifest.

Layout consumed, one directory per clip, each already trimmed to a single CWM window::

    <root>/
      clip_000000_0/                 clip_<six-digit episode>_<window 0..4>
        target/rgb.mp4               1344x768, 124 frames, 24 fps  -> the denoising target
        target/anchor.png            1344x768 lossless, frame 0    -> the appearance anchor
        proxy/duv.mp4                336x192, 124 frames, lossless -> the Ref2VA video reference
        annotations/caption.json     optional, episode-level
        clip_report.json             authoritative metadata for this clip
      clip_000000_1/
      ...

The DUV video is passed through as ``proxy_duv_video``, unconverted. Its channel convention is the
producer's, not this repo's: log depth over 0.1 m to 8000 m rising away from the camera with 255 as
a sky sentinel, and a class code in the other two channels. That is not what
:mod:`fastvideo.pipelines.basic.minimax_h3.proxy` would have written, and it does not need to be --
this stage starts from the base Ref2VA checkpoint, which has no DUV prior either way, so what the
proxy channels mean is learned here. What matters is that the same convention is used at sampling
time, which is why nothing rewrites the pixels on the way in.

Two properties are checked rather than assumed, because both are silent when wrong:

* **Every clip must agree on the convention.** ``duv_depth_inverted`` flips the depth channel's
  direction and ``taxonomy`` decides what the class codes mean. A set mixing either one asks the
  model to read one channel two ways, and no shape or count check would notice.
* **The DUV must decode to its palette exactly.** A DUV frame is integer codes wearing an RGB
  costume, so a decoder handing back BGR, or any resampling, turns class codes into colours no
  segmenter predicted. ``--preflight-limit`` clips are decoded and matched against the palette.

Episodes, not clips, are split. Five clips cut from one 60-second episode share weather, lighting
and terrain, so splitting by clip would validate on footage already trained on and report a loss
that looks much better than the model is.

Usage::

    python scripts/h3_proxy/prepare_data/clip_dir_to_encode_manifest.py \\
      --root /data/binghe/datasets/ABot-sub-2000-clips \\
      --split train --val-episodes 24 \\
      --out /workspace/h3_abot_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seg_dir_to_encode_manifest import (  # noqa: E402
    largest_valid_num_frames, probe_usable_frames,
)

CLIP_PATTERN = re.compile(r"^clip_(\d+)_(\d+)$")

# The (green, blue) pairs the delivery DUV uses for its 11-class taxonomy. Five classes share
# (0, 0) and sky shares (255, 255) with road, so this is what the encoding can express, not what
# the segmenter predicted; see the dataset's DATA_CLIPS.md.
DUV_CLASS_CODES = frozenset({(255, 255), (0, 255), (0, 128), (64, 0), (128, 0), (0, 0), (255, 0)})

# Keys a caption.json might carry, most specific first. The file is copied verbatim from the source
# episode, so its schema is the upstream dataset's rather than something this pipeline defines.
CAPTION_KEYS = ("caption", "description", "text", "prompt", "summary", "title")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root containing clip_*/ directories.")
    p.add_argument("--out", required=True, help="Output encode manifest jsonl.")
    p.add_argument("--split", choices=("train", "val", "all"), default="train")
    p.add_argument("--val-episodes",
                   type=int,
                   default=24,
                   help="Episodes held out, spread evenly over the sorted episode list. The train and val splits "
                   "are complements of one another by construction, so there is no split file to keep in sync.")
    p.add_argument("--num-frames", type=int, default=124, help="Frames per clip; checked against each report.")
    p.add_argument("--target-height", type=int, default=768)
    p.add_argument("--target-width", type=int, default=1344)
    p.add_argument("--proxy-height", type=int, default=192)
    p.add_argument("--proxy-width", type=int, default=336)
    p.add_argument("--prompt-fallback",
                   default="",
                   help="Prompt for clips with neither prompt.txt nor a usable caption.json. Empty means skip them.")
    p.add_argument("--no-episode-caption",
                   action="store_true",
                   help="Ignore annotations/caption.json. It describes the whole 60-second episode rather than this "
                   "5-second window, so a run that wants per-clip text only should not silently inherit it.")
    p.add_argument("--preflight-limit",
                   type=int,
                   default=8,
                   help="Clips whose DUV is decoded and palette-checked; 0 skips the check.")
    p.add_argument("--probe-limit", type=int, default=32, help="Clips probed for the 24-fps frame budget; 0 probes all.")
    return p.parse_args()


def episode_of(name: str) -> str:
    match = CLIP_PATTERN.match(name)
    if match is None:
        raise ValueError(f"{name!r} does not look like clip_<episode>_<window>")
    return match.group(1)


def evenly_spaced(items: list, limit: int) -> list:
    """Sample ``limit`` items across the whole list rather than taking a prefix."""
    if limit <= 0 or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def read_prompt(clip: Path, *, use_episode_caption: bool) -> tuple[str, str]:
    """Resolve this clip's prompt and say where it came from.

    ``prompt.txt`` wins so that regenerating text per clip is a matter of writing that file, with
    no change here and no re-derivation of which caption belonged to which clip.
    """
    prompt_file = clip / "prompt.txt"
    if prompt_file.is_file():
        text = prompt_file.read_text(encoding="utf-8").strip()
        if text:
            return text, "prompt.txt"

    caption_file = clip / "annotations" / "caption.json"
    if use_episode_caption and caption_file.is_file():
        try:
            payload = json.loads(caption_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        text = extract_caption(payload)
        if text:
            return text, "caption.json"
    return "", "none"


def extract_caption(payload: object) -> str:
    """Pull a caption string out of an unknown-schema payload, or return ""."""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        parts = [extract_caption(item) for item in payload]
        return " ".join(part for part in parts if part).strip()
    if isinstance(payload, dict):
        for key in CAPTION_KEYS:
            if isinstance(payload.get(key), str) and payload[key].strip():
                return payload[key].strip()
        # No known key hit. Rather than guess at nesting, take the longest string value: a caption
        # file's prose is reliably longer than the ids and paths beside it.
        strings = [value.strip() for value in payload.values() if isinstance(value, str) and value.strip()]
        if strings:
            return max(strings, key=len)
    return ""


def check_report(report: dict, args: argparse.Namespace) -> str | None:
    """Reasons this clip is not usable, or None. Shape checks only; conventions are checked later."""
    if report.get("deliverable") is not True:
        return f"deliverable={report.get('deliverable')!r} (placeholder backend, not trainable)"
    frames = report.get("frames")
    if frames is not None and int(frames) != args.num_frames:
        return f"frames={frames}, expected {args.num_frames}"
    fps = report.get("fps")
    if fps is not None and abs(float(fps) - 24.0) > 1e-6:
        return f"fps={fps}, expected 24"
    for key, expected in (("target_size", [args.target_width, args.target_height]),
                          ("duv_size", [args.proxy_width, args.proxy_height])):
        value = report.get(key)
        if value is not None and list(value) != expected:
            return f"{key}={value}, expected {expected}"
    return None


def check_conventions(conventions: dict[tuple, list[str]]) -> None:
    """Refuse a split whose clips disagree about what the DUV channels mean.

    ``duv_depth_inverted`` flips the depth channel's direction and ``taxonomy`` decides what the
    class codes stand for. A mixed set asks the model to read one channel two ways, and no shape,
    count or loss check downstream would show it.
    """
    if len(conventions) <= 1:
        return
    print(f"ERROR: {len(conventions)} different DUV conventions in this split:")
    for (taxonomy, inverted), names in sorted(conventions.items(), key=lambda item: -len(item[1])):
        print(f"  taxonomy={taxonomy!r} inverted={inverted}: {len(names)} clips, e.g. {', '.join(names[:3])}")
    raise SystemExit("Encode one convention at a time. A model cannot read a channel that means near-is-bright on "
                     "some clips and near-is-dark on others, and nothing downstream can detect the mix.")


def preflight_duv(paths: list[Path], height: int, width: int, num_frames: int) -> tuple[list[str], str | None]:
    """Decode a sample of DUV videos and confirm they arrive as the exact codes they were written as.

    Returns ``(problems, skipped_reason)``. A missing decoder is not a problem with the data, so it
    leaves ``problems`` empty and names itself instead: this script also runs on machines that only
    have the dataset, and refusing to emit a manifest there would be the wrong trade.
    """
    problems: list[str] = []
    try:
        import numpy as np

        from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_video
    except ImportError as error:
        return [], f"decoder unavailable ({error}); run this where the encoder runs to get the check"

    for path in paths:
        try:
            frames, fps, _ = decode_reference_video(path)
        except Exception as error:  # noqa: BLE001 - a container can fail in codec-specific ways
            problems.append(f"{path.parent.parent.name}: cannot decode duv.mp4: {error}")
            continue
        if abs(float(fps) - 24.0) > 1e-6:
            problems.append(f"{path.parent.parent.name}: duv.mp4 is {fps} fps, not 24; the encoder would resample it "
                            "while the target keeps its own timeline, silently desynchronising the pair")
        if frames.shape[1:3] != (height, width):
            problems.append(f"{path.parent.parent.name}: duv.mp4 is {frames.shape[2]}x{frames.shape[1]}, "
                            f"expected {width}x{height}")
            continue
        if frames.shape[0] < num_frames:
            problems.append(f"{path.parent.parent.name}: duv.mp4 has {frames.shape[0]} frames, expected {num_frames}")
        # Every distinct (green, blue) pair in the clip, which on a lossless stream is a handful.
        codes = {tuple(int(value) for value in pair) for pair in np.unique(frames[:num_frames, ..., 1:].reshape(-1, 2),
                                                                          axis=0)}
        stray = sorted(codes - DUV_CLASS_CODES)
        if stray:
            problems.append(f"{path.parent.parent.name}: duv.mp4 carries {len(stray)} (green, blue) pairs outside the "
                            f"class palette, e.g. {stray[:4]}. The stream is lossless, so this means the decode is "
                            "not returning the written codes -- a BGR channel order or a chroma-subsampled re-encode "
                            "would both look like this.")
    return problems, None


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root does not exist: {root}")

    clip_dirs = sorted(path for path in root.glob("clip_*") if path.is_dir() and CLIP_PATTERN.match(path.name))
    if not clip_dirs:
        raise SystemExit(f"No 'clip_<episode>_<window>' directories under {root}")

    # Discovered by walking, not read from a manifest: the dataset ships 112 shard manifests and no
    # total, and three episodes are short enough that they contributed no clips at all, so
    # clip_<episode>_0..4 is not a set that can be assumed complete.
    episodes = sorted({episode_of(path.name) for path in clip_dirs})
    if args.val_episodes >= len(episodes):
        raise SystemExit(f"--val-episodes {args.val_episodes} leaves no training episodes; {len(episodes)} exist")
    val_episodes = set(evenly_spaced(episodes, args.val_episodes))
    if args.split == "train":
        keep_episodes = set(episodes) - val_episodes
    elif args.split == "val":
        keep_episodes = val_episodes
    else:
        keep_episodes = set(episodes)

    rows: list[dict] = []
    rejected: list[str] = []
    out_of_split = 0
    prompt_sources: dict[str, int] = {}
    conventions: dict[tuple, list[str]] = {}
    for clip in clip_dirs:
        if episode_of(clip.name) not in keep_episodes:
            out_of_split += 1
            continue
        target, anchor, duv = (clip / "target" / "rgb.mp4", clip / "target" / "anchor.png", clip / "proxy" / "duv.mp4")
        report_file = clip / "clip_report.json"
        missing = [path.name for path in (target, anchor, duv, report_file) if not path.is_file()]
        if missing:
            rejected.append(f"{clip.name}: missing {', '.join(missing)}")
            continue
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            rejected.append(f"{clip.name}: unreadable clip_report.json: {error}")
            continue
        reason = check_report(report, args)
        if reason is not None:
            rejected.append(f"{clip.name}: {reason}")
            continue

        prompt, source = read_prompt(clip, use_episode_caption=not args.no_episode_caption)
        if not prompt:
            prompt = args.prompt_fallback.strip()
            source = "fallback" if prompt else "none"
        if not prompt:
            rejected.append(f"{clip.name}: no prompt (no prompt.txt, no usable caption.json, no --prompt-fallback)")
            continue
        prompt_sources[source] = prompt_sources.get(source, 0) + 1

        # `.get` throughout: the two-step pipeline that produced this subset writes fewer fields
        # than the one-pass `clip-episodes` route does.
        convention = (report.get("taxonomy"), bool(report.get("duv_depth_inverted", False)))
        conventions.setdefault(convention, []).append(clip.name)

        rows.append({
            "name": clip.name,
            "target": str(target.relative_to(root)),
            "proxy_duv_video": str(duv.relative_to(root)),
            "anchor": str(anchor.relative_to(root)),
            "prompt": prompt,
            "id": clip.name,
        })

    if not rows:
        raise SystemExit(f"No usable clips survived (scanned {len(clip_dirs)}, {len(rejected)} rejected, "
                         f"{out_of_split} out of split '{args.split}').")

    # Everything that can reject the set runs before the file exists. A manifest left on disk beside
    # an error message is worse than no manifest: the encode step takes a path, not this script's
    # exit code, so a stale file is indistinguishable from a good one hours later.
    check_conventions(conventions)
    sampled = evenly_spaced(rows, args.preflight_limit) if args.preflight_limit > 0 else []
    problems, skipped = preflight_duv(
        [root / row["proxy_duv_video"] for row in sampled],
        args.proxy_height,
        args.proxy_width,
        args.num_frames,
    )
    if problems:
        print(f"DUV preflight found {len(problems)} problem(s):")
        for line in problems:
            print(f"  {line}")
        raise SystemExit("Fix the DUV stream before encoding. Every one of these is invisible downstream: the cache "
                         "would be written, training would converge, and the proxy would be describing noise.")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept_episodes = {episode_of(row["name"]) for row in rows}
    print(f"Wrote {len(rows)} rows -> {out}")
    print(f"  {len(clip_dirs)} clips on disk over {len(episodes)} episodes")
    print(f"  split '{args.split}': {len(rows)} clips over {len(kept_episodes)} episodes "
          f"({len(val_episodes)} episodes held out for val)")
    if out_of_split:
        print(f"  {out_of_split} clips belong to the other split")
    if rejected:
        print(f"  {len(rejected)} rejected:")
        for line in rejected[:10]:
            print(f"    {line}")
        if len(rejected) > 10:
            print(f"    ... and {len(rejected) - 10} more")

    print(f"  prompt sources: {', '.join(f'{key}={value}' for key, value in sorted(prompt_sources.items()))}")
    if prompt_sources.get("caption.json"):
        print("    NOTE: caption.json is episode-level. It describes the whole 60-second episode, not the 5.17 "
              "seconds this clip covers, so it will name things the clip never shows.")

    taxonomy, inverted = next(iter(conventions))
    print(f"  DUV convention: taxonomy={taxonomy!r}, depth_inverted={inverted}")
    if skipped is not None:
        print(f"  DUV preflight skipped: {skipped}")
    elif sampled:
        print(f"  DUV preflight passed on {len(sampled)} clip(s): "
              f"{args.proxy_width}x{args.proxy_height}, 24 fps, all class codes in the palette")

    report_frame_budget(rows, root, args.probe_limit, args.num_frames)

    for global_batch in (8, ):
        steps = 3 * -(-len(rows) // global_batch)
        print(f"  three epochs at global batch {global_batch} is max_train_steps: {steps}")


def report_frame_budget(rows: list[dict], root: Path, limit: int, num_frames: int) -> None:
    """Report the shortest clip's 24-fps budget, the way the encoder will measure it."""
    sample = evenly_spaced(rows, limit) if limit > 0 else rows
    budgets: list[tuple[str, int]] = []
    for row in sample:
        for key in ("target", "proxy_duv_video"):
            usable = probe_usable_frames(root / str(row[key]))
            if usable is not None:
                budgets.append((f"{row['name']}/{key}", usable))
    if not budgets:
        print("  frame budget: could not probe any clip (PyAV missing or metadata absent).")
        return
    name, worst = min(budgets, key=lambda item: item[1])
    print(f"  frame budget over {len(budgets)} probed streams: shortest is {worst} frames at 24 fps ({name})")
    if worst < num_frames:
        print(f"    WARNING: below --num-frames {num_frames}. Largest that fits: {largest_valid_num_frames(worst)}")


if __name__ == "__main__":
    main()
