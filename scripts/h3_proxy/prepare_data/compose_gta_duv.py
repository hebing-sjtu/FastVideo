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
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vlm_filter import add_filter_arguments, load_vlm_filter  # noqa: E402

DEPTH_NEAR = 0.1
DEPTH_FAR = 256.0
PROXY_NEAR = 0.1
PROXY_FAR = 8000.0
PROXY_MAX = 254
PROXY_SKY = 255

# DATA_F.md / standard11 G,B. Not injective: the eleven labels collapse onto six codes, because
# sky and road share (255,255) and building, ground, terrain, water, and prop all share (0,0).
# Kept as the default only because an existing cache was encoded with it.
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

# CWM's semantic grid: cwm_h3_inference.constants SEMANTIC_U x SEMANTIC_V, twelve well-separated
# (G,B) pairs. Eleven GTA classes fit, so every label gets its own code and the model can tell road
# from sky without falling back on R==255 -- which also means "depth invalid" and so cannot
# disambiguate anything.
CWM_SEMANTIC_U = (32, 96, 160, 224)
CWM_SEMANTIC_V = (43, 128, 213)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--probe-only", action="store_true", help="Print first-frame stats, write nothing.")
    p.add_argument(
        "--height",
        type=int,
        default=704,
        help="Crop the native 720p height to a multiple of 32 (default 704). 720 yields a 45-row "
        "VAE latent, which the 2x2 patch cannot tile.",
    )
    p.add_argument(
        "--out-width",
        type=int,
        default=0,
        help="Resample the streams to this width before composing, with --out-height. The encoder "
        "refuses to resize a DUV video, so the grid has to be decided here; 336x192 is the released "
        "CWM proxy geometry. Zero keeps the native width and only applies --height's crop.",
    )
    p.add_argument("--out-height", type=int, default=0, help="See --out-width.")
    p.add_argument(
        "--semantic-palette",
        choices=("standard11", "cwm12"),
        default="standard11",
        help="'cwm12' gives every class its own (G,B) from CWM's 4x3 semantic grid, which "
        "'standard11' collapses eight classes out of. Changing this changes the proxy pixels, so "
        "a cache encoded under one palette cannot be compared against a run under the other.",
    )
    add_filter_arguments(p)
    return p.parse_args()


def build_palette(kind: str, class_ids: list[int]) -> dict[int, tuple[int, int]]:
    """Class id -> (G, B). ``cwm12`` is injective; ``standard11`` is the legacy table."""
    if kind == "standard11":
        return dict(PROXY_GB)
    slots = [(u, v) for u in CWM_SEMANTIC_U for v in CWM_SEMANTIC_V]
    if len(class_ids) > len(slots):
        raise SystemExit(f"cwm12 has {len(slots)} codes but the semantic map declares {len(class_ids)} classes.")
    return {class_id: slots[index] for index, class_id in enumerate(sorted(class_ids))}


def resample_nearest(plane: np.ndarray, width: int, height: int) -> np.ndarray:
    """Index-preserving resize.

    Both planes here are code books, not images: the semantic plane holds class ids and the depth
    plane holds a quantised log-z. Any interpolating filter would average two codes into a third
    that means something else -- a road/sky boundary would grow a rim of "vehicle". Nearest is the
    only filter that keeps every output pixel a value that was actually observed.
    """
    if plane.shape[1] == width and plane.shape[0] == height:
        return plane
    rows = (np.arange(height) * plane.shape[0] // height).clip(0, plane.shape[0] - 1)
    cols = (np.arange(width) * plane.shape[1] // width).clip(0, plane.shape[1] - 1)
    return plane[rows[:, None], cols[None, :]]


def read_class_ids(seg: Path) -> list[int]:
    """Class ids the seg's ``proxy/semantic.json`` declares, or the legacy 0..10."""
    path = seg / "proxy" / "semantic.json"
    if path.is_file():
        try:
            classes = json.loads(path.read_text(encoding="utf-8")).get("classes")
        except (OSError, json.JSONDecodeError):
            classes = None
        if isinstance(classes, dict) and classes:
            return sorted(int(key) for key in classes)
    return sorted(PROXY_GB)


def decode_depth_grey(grey: np.ndarray) -> np.ndarray:
    span = math.log(DEPTH_FAR) - math.log(DEPTH_NEAR)
    metres = np.exp(math.log(DEPTH_FAR) - (grey.astype(np.float64) / 255.0) * span)
    return np.where(grey == 0, 0.0, metres).astype(np.float32)


def log_code_forward(metres: np.ndarray) -> np.ndarray:
    span = math.log(PROXY_FAR) - math.log(PROXY_NEAR)
    clipped = np.clip(metres.astype(np.float64), PROXY_NEAR, PROXY_FAR)
    fraction = 1.0 - (math.log(PROXY_FAR) - np.log(clipped)) / span
    return np.rint(fraction * PROXY_MAX).astype(np.uint8)


def compose_frame(metres: np.ndarray, ids: np.ndarray, palette: dict[int, tuple[int, int]]) -> np.ndarray:
    red = log_code_forward(metres)
    sky = (metres <= 1.0e-3) | (ids == 0)
    red[sky] = PROXY_SKY
    green = np.zeros_like(red)
    blue = np.zeros_like(red)
    for class_id, (g, b) in palette.items():
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
    """Lossless RGB mp4 via PyAV. stdin-to-ffmpeg died with EPIPE on this node."""
    import av

    height, width = frames[0].shape[:2]
    temporary = path.with_name(path.stem + ".tmp.mp4")
    temporary.unlink(missing_ok=True)
    container = av.open(str(temporary), mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": "0", "preset": "fast", "tune": "fastdecode"}
        for array in frames:
            video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"PyAV wrote no bytes to {temporary}")
    temporary.replace(path)


def probe_video_fps(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or getattr(stream, "guessed_rate", None)
        return float(rate) if rate else 24.0


def write_split_manifests(root: Path, names: list[str], *, val_count: int = 24) -> None:
    """Train/val jsonl so the existing --split train/val path works.

    ``names`` is the composed set, not everything on disk, so the split is drawn over the clips that
    actually survived the corpus gate. A split written over the full corpus would hand the manifest
    builders val ids they then filter away, shrinking the held-out set for no stated reason.
    """
    directory = root / "manifests"
    directory.mkdir(exist_ok=True)
    existing = list(directory.glob("*_train.jsonl")) + list(directory.glob("*_val.jsonl"))
    if existing:
        print(f"Keeping the split already in {directory}: {', '.join(path.name for path in sorted(existing))}. "
              "Delete them to redraw it over this run's clips.")
        return
    if len(names) < 2:
        raise SystemExit(f"A split needs at least two clips; {len(names)} survived.")
    # Capped at half the corpus. Without this a set smaller than val_count lands entirely in val
    # and train comes out empty, which reads as a successful run right up until training starts.
    val_count = max(1, min(val_count, len(names) // 2))
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
    if not segs:
        raise SystemExit(f"No seg_* under {root}")

    # Filter before composing, not after: a rejected clip's DUV is a few seconds of lossless
    # encoding that nothing will ever read.
    vlm = load_vlm_filter(root, args)
    if vlm is not None:
        vlm.report(on_disk={path.name for path in segs})
        rejected = [path.name for path in segs if not vlm.verdict(path.name)[0]]
        segs = [path for path in segs if vlm.verdict(path.name)[0]]
        print(f"  composing {len(segs)}, skipping {len(rejected)} the judge rejected")
        if not segs:
            raise SystemExit("Every clip was rejected. Lower --min-vlm-score or pass --vlm-filter none.")
    if args.limit:
        segs = segs[: args.limit]

    composed_names: list[str] = []
    written = skipped = failed = 0
    for index, seg in enumerate(segs):
        depth_path, semantic_path, duv_path = (seg / "proxy" / "depth.mp4", seg / "proxy" / "semantic.mp4",
                                               seg / "proxy" / "duv.mp4")
        if duv_path.is_file() and not args.overwrite and not args.probe_only:
            # Already composed on an earlier run, so still part of the corpus for the split.
            composed_names.append(seg.name)
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
            palette = build_palette(args.semantic_palette, read_class_ids(seg))
            ids0 = semantic_ids(semantic_frames[0])
            metres0 = decode_depth_grey(depth_frames[0])
            if index == 0 or args.probe_only:
                distinct = len(set(palette.values()))
                print(f"  palette {args.semantic_palette}: {len(palette)} classes -> {distinct} distinct (G,B)"
                      f"{'' if distinct == len(palette) else '  [classes collapse; the model cannot separate them]'}")
                valid = metres0 > 1.0e-3
                print(f"{seg.name}: {depth_frames[0].shape[1]}x{depth_frames[0].shape[0]} "
                      f"{len(depth_frames)} frames, depth grey "
                      f"min={int(depth_frames[0].min())} max={int(depth_frames[0].max())}, "
                      f"metres p50={float(np.median(metres0[valid])) if np.any(valid) else 0:.2f}, "
                      f"semantic ids {sorted(int(x) for x in np.unique(ids0)[:16])}")
            if args.probe_only:
                continue
            resize = bool(args.out_width and args.out_height)
            composed = []
            for depth, semantic in zip(depth_frames, semantic_frames, strict=True):
                grey, ids = depth, semantic_ids(semantic)
                if resize:
                    grey = resample_nearest(grey, args.out_width, args.out_height)
                    ids = resample_nearest(ids, args.out_width, args.out_height)
                composed.append(compose_frame(decode_depth_grey(grey), ids, palette))
            # --out-* already chose a tileable grid, so the 720 -> 704 crop only applies otherwise.
            if not resize and args.height and composed[0].shape[0] != args.height:
                if args.height > composed[0].shape[0]:
                    raise ValueError(f"--height {args.height} is taller than {composed[0].shape[0]}")
                trim = composed[0].shape[0] - args.height
                top = trim // 2
                bottom = composed[0].shape[0] - (trim - top)
                composed = [frame[top:bottom] for frame in composed]
            write_duv(duv_path, composed, probe_video_fps(depth_path))
            composed_names.append(seg.name)
            written += 1
            if written % 20 == 0:
                print(f"[{index + 1}/{len(segs)}] wrote {written}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"[{index + 1}/{len(segs)}] {seg.name}: {error}")

    if not args.probe_only:
        print(f"Done: {written} written, {skipped} skipped, {failed} failed")
        if composed_names and not args.limit:
            write_split_manifests(root, sorted(composed_names))
        elif args.limit:
            print(f"  --limit {args.limit} was set, so no split was written; rerun without it.")


if __name__ == "__main__":
    main()
