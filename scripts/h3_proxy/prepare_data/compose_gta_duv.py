#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compose ``proxy/duv.mp4`` from native GTA ``depth.mp4`` + ``semantic.mp4``.

``gta_web_0902`` ships the three game streams separately and never wrote a DUV.
This is the same deterministic pack ``proxy_extract.proxy.compose_proxy_frame``
uses for ABot delivery: R = forward log-z (0.1–8000 m, sky=255), G/B = 11-class
colours. Depth video is decoded as inverted log-z (near bright, 0 = invalid),
which is DATA_F.md's ``depth.mp4`` convention. If this GTA depth is a different
encoding, the first-frame probe will show it — do not encode the corpus until
that printout looks like a depth map, not noise.

Usage::

    python scripts/h3_proxy/prepare_data/compose_gta_duv.py \\
        --root /data/binghe/datasets/gta_web_0902 --limit 1

    python scripts/h3_proxy/prepare_data/compose_gta_duv.py \\
        --root /data/binghe/datasets/gta_web_0902
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

DEPTH_NEAR = 0.1
DEPTH_FAR = 256.0
PROXY_NEAR = 0.1
PROXY_FAR = 8000.0
PROXY_MAX = 254
PROXY_SKY = 255

# DATA_F.md / standard11 G,B. sky and road share (255,255); R==255 is sky.
PROXY_GB = {
    0: (255, 255),  # sky
    1: (0, 255),  # player
    2: (0, 128),  # ped
    3: (64, 0),  # vehicle
    4: (0, 0),  # building
    5: (255, 255),  # road
    6: (0, 0),  # ground
    7: (255, 0),  # vegetation
    8: (0, 0),  # terrain
    9: (0, 0),  # water
    10: (0, 0),  # prop
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--probe-only", action="store_true", help="Print first-frame stats, write nothing.")
    return p.parse_args()


def decode_depth_grey(grey: np.ndarray) -> np.ndarray:
    span = math.log(DEPTH_FAR) - math.log(DEPTH_NEAR)
    metres = np.exp(math.log(DEPTH_FAR) - (grey.astype(np.float64) / 255.0) * span)
    return np.where(grey == 0, 0.0, metres).astype(np.float32)


def log_code_forward(metres: np.ndarray) -> np.ndarray:
    span = math.log(PROXY_FAR) - math.log(PROXY_NEAR)
    clipped = np.clip(metres.astype(np.float64), PROXY_NEAR, PROXY_FAR)
    fraction = 1.0 - (math.log(PROXY_FAR) - np.log(clipped)) / span
    return np.rint(fraction * PROXY_MAX).astype(np.uint8)


def compose_frame(metres: np.ndarray, ids: np.ndarray) -> np.ndarray:
    red = log_code_forward(metres)
    sky = (metres <= 1.0e-3) | (ids == 0)
    red[sky] = PROXY_SKY
    green = np.zeros_like(red)
    blue = np.zeros_like(red)
    for class_id, (g, b) in PROXY_GB.items():
        mask = ids == class_id
        green[mask] = g
        blue[mask] = b
    return np.stack([red, green, blue], axis=-1)


def read_rgb_video(path: Path) -> list[np.ndarray]:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames


def read_grey_video(path: Path) -> list[np.ndarray]:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(rgb[..., 0])
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames


def semantic_ids(frame: np.ndarray) -> np.ndarray:
    """Prefer the blue channel when it looks like a class id map."""
    blue = frame[..., 2]
    if int(blue.max()) <= 15:
        return blue
    # Some writers put the id in red.
    red = frame[..., 0]
    if int(red.max()) <= 15:
        return red
    raise ValueError(f"semantic frame does not look like an id map: unique RGB "
                     f"R={len(np.unique(frame[..., 0]))} G={len(np.unique(frame[..., 1]))} "
                     f"B={len(np.unique(frame[..., 2]))} maxB={int(blue.max())}")


def write_duv(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    temporary = path.with_suffix(".mp4.tmp")
    command = [
        os.environ.get("FFMPEG", "ffmpeg"),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264rgb",
        "-pix_fmt",
        "rgb24",
        "-crf",
        "0",
        "-preset",
        "fast",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        err = process.stderr.read().decode(errors="replace") if process.stderr else ""
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed writing {path}: {err[-400:]}")
    temporary.replace(path)


def probe_video_fps(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or getattr(stream, "guessed_rate", None)
        return float(rate) if rate else 24.0


def write_split_manifests(root: Path, names: list[str], *, val_count: int = 24) -> None:
    """Train/val jsonl so the existing --split train/val path works."""
    directory = root / "manifests"
    directory.mkdir(exist_ok=True)
    if list(directory.glob("*_train.jsonl")) or list(directory.glob("*_val.jsonl")):
        return
    step = max(1, len(names) // val_count)
    val = {names[i] for i in list(range(0, len(names), step))[:val_count]}
    train = [name for name in names if name not in val]
    val_list = [name for name in names if name in val]
    for split, rows in (("train", train), ("val", val_list)):
        path = directory / f"gta_web_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for name in rows:
                handle.write(json.dumps({"id": name}) + "\n")
        print(f"Wrote {len(rows)} ids -> {path}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    segs = sorted(path for path in root.glob("seg_*") if path.is_dir())
    if args.limit:
        segs = segs[: args.limit]
    if not segs:
        raise SystemExit(f"No seg_* under {root}")

    written = skipped = failed = 0
    for index, seg in enumerate(segs):
        depth_path, semantic_path, duv_path = (seg / "proxy" / "depth.mp4", seg / "proxy" / "semantic.mp4",
                                               seg / "proxy" / "duv.mp4")
        if duv_path.is_file() and not args.overwrite and not args.probe_only:
            skipped += 1
            continue
        if not depth_path.is_file() or not semantic_path.is_file():
            failed += 1
            print(f"[{index + 1}/{len(segs)}] {seg.name}: missing depth or semantic")
            continue
        try:
            depth_frames = read_grey_video(depth_path)
            semantic_frames = read_rgb_video(semantic_path)
            if len(depth_frames) != len(semantic_frames):
                raise ValueError(f"depth {len(depth_frames)} frames vs semantic {len(semantic_frames)}")
            ids0 = semantic_ids(semantic_frames[0])
            metres0 = decode_depth_grey(depth_frames[0])
            if index == 0 or args.probe_only:
                valid = metres0 > 1.0e-3
                print(f"{seg.name}: {depth_frames[0].shape[1]}x{depth_frames[0].shape[0]} "
                      f"{len(depth_frames)} frames, depth grey "
                      f"min={int(depth_frames[0].min())} max={int(depth_frames[0].max())}, "
                      f"metres p50={float(np.median(metres0[valid])) if np.any(valid) else 0:.2f}, "
                      f"semantic ids {sorted(int(x) for x in np.unique(ids0)[:16])}")
            if args.probe_only:
                continue
            composed = [
                compose_frame(decode_depth_grey(depth), semantic_ids(semantic))
                for depth, semantic in zip(depth_frames, semantic_frames, strict=True)
            ]
            write_duv(duv_path, composed, probe_video_fps(depth_path))
            written += 1
            if written % 20 == 0:
                print(f"[{index + 1}/{len(segs)}] wrote {written}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"[{index + 1}/{len(segs)}] {seg.name}: {error}")

    if not args.probe_only:
        print(f"Done: {written} written, {skipped} skipped, {failed} failed")
        write_split_manifests(root, [path.name for path in sorted(root.glob("seg_*")) if path.is_dir()])


if __name__ == "__main__":
    main()
