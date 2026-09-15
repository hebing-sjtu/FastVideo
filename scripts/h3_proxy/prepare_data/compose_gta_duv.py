#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compose ``proxy/duv.mp4`` from native GTA ``depth.mp4`` + ``semantic.mp4``.

``gta_web_0902`` ships the three game streams separately and never wrote a DUV,
so this composes one. Decoding is always DATA_F.md's ``depth.mp4`` convention --
inverted log-z over ``[near, far]``, near bright, ``gray == 0`` invalid -- with
the range taken from the seg's ``metadata.json`` when it declares one.

The output convention is a choice. ``cwm`` is the default, matching
``cwm_h3_inference.duv``:

* R is ``(ln(far) - ln(d)) / (ln(far) - ln(near))`` over **0.3-256 m**,
  quantised to uint16 then divided by 257 for the 8-bit plane. Near is bright and
  invalid is 0, so the far plane and the sky share code 0.
* G/B are ``(SEMANTIC_U[id % 4], SEMANTIC_V[id // 4])`` -- u varies fastest.
  Twelve mid-tone codes, all distinct, touching neither 0 nor 255.

The adapter is trained from scratch on base Ref2VA, which has never seen a DUV
frame, so none of this is chosen to match a pretrained association -- there is
none to match. It is chosen on information content:

* 0.3-256 m is the range the source actually reports. ABot's 0.1-8000 m spends
  30% of its codes on distances that never occur, leaving 4.55% per code against
  2.68% here.
* Sky sharing code 0 with the far plane costs nothing *because* the semantic
  channel is injective -- sky has its own (G, B). Under the collapsing palette it
  did not, which is why ABot needed R's 255 as a sky sentinel and then could not
  tell sky from depth-invalid.
* The codes have to survive the video VAE, which reconstructs the extremes
  worst. DATA_F's colours lean on 0 and 255; this grid avoids both.

``abot`` reproduces what the first cache was built with, for comparing against
that run. Every difference is silent, so a cache built under one convention
cannot be compared against a run under the other.

Usage::

    # Look before writing: prints the decoded range, the palette and the R stats.
    python scripts/h3_proxy/prepare_data/compose_gta_duv.py \\
        --root /data/binghe/datasets/gta_web_0902_v2/gta_web_0902 --probe-only --limit 1

    python scripts/h3_proxy/prepare_data/compose_gta_duv.py \\
        --root /data/binghe/datasets/gta_web_0902_v2/gta_web_0902 --overwrite
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

# DATA_F.md's depth.mp4: inverted log-z over this range, near bright, gray == 0 invalid. Used for
# decoding only, and overridden by metadata.json or --source-near/--source-far.
SOURCE_NEAR = 0.1
SOURCE_FAR = 256.0

# cwm_h3_inference.constants. The range the checkpoint was pretrained on.
CWM_NEAR = 0.3
CWM_FAR = 256.0
CWM_SEMANTIC_U = (32, 96, 160, 224)
CWM_SEMANTIC_V = (43, 128, 213)

# ABot's DUV, for reproducing the first cache. R is forward log-z with 255 reserved for sky.
ABOT_NEAR = 0.1
ABOT_FAR = 8000.0
ABOT_MAX = 254
ABOT_SKY = 255

GTA_SKY_ID = 0

# DATA_F.md's semantic *colours*. Not injective: eleven labels collapse onto six codes, because sky
# and road share (255,255) and building, ground, terrain, water and prop all share (0,0). Sky is
# then separable only by R == 255, which also means "depth invalid" and so disambiguates nothing.
ABOT_GB = {
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

# GTA label (DATA_F.md) -> CWM label (INFERENCE.md).
#
#   GTA  0 sky 1 player 2 ped 3 vehicle 4 building 5 road 6 ground 7 vegetation 8 terrain 9 water 10 prop
#   CWM  0 void_unknown 1 sky 2 water 3 terrain 4 road_paved 5 vegetation
#        6 building_structure 7 infrastructure 8 human 9 animal 10 vehicle 11 prop
#
# Which slot a label lands in does not matter, and it is worth being explicit about why, because the
# names invite an argument that has no stake in it. The adapter here is trained from scratch on base
# MiniMax-H3 Ref2VA, which has never seen a DUV frame -- the association between these (G, B) pairs
# and what they denote lives in CWM's LoRA, which is not loaded. To this model `(96, 128)` is two
# bytes, so the assignment is a relabelling and any injective one trains the same.
#
# Three properties do matter:
#
#   injective -- the collapsed table it replaces made road and sky the same symbol, and no amount of
#     training separates a symbol from itself;
#   well separated after the VAE -- the proxy reaches the DiT through the video VAE, so codes have to
#     survive a lossy round trip. CWM's 4x3 grid is mid-tone with 64 and 85 apart on the two axes and
#     touches neither 0 nor 255, which is where reconstruction is worst. DATA_F's colours lean on
#     both extremes;
#   stable across the corpus -- the one property a cache silently violates when the palette changes
#     under it.
#
# Following CWM's meanings anyway costs nothing and keeps the option of checking a clip against the
# released LoRA, which needs the codes to line up.
GTA_TO_CWM = {
    0: 1,   # sky            -> sky
    1: 8,   # player         -> human
    2: 9,   # ped            -> animal        (see above: an arbitrary free slot, not a claim)
    3: 10,  # vehicle        -> vehicle
    4: 6,   # building       -> building_structure
    5: 4,   # road           -> road_paved
    6: 7,   # ground         -> infrastructure
    7: 5,   # vegetation     -> vegetation
    8: 3,   # terrain        -> terrain
    9: 2,   # water          -> water
    10: 11,  # prop          -> prop
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--probe-only", action="store_true", help="Print first-frame stats, write nothing.")
    p.add_argument(
        "--out-name",
        default="duv.mp4",
        help="Filename written under each seg's proxy/. The default overwrites the canonical "
        "duv.mp4 in place, which is what a corpus wants. Point two runs at different names to hold "
        "two conventions side by side, which is the only way to A/B an encoding on otherwise "
        "identical clips -- composing in place makes every checkpoint trained on the old one "
        "unevaluatable.",
    )
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
        "--convention",
        choices=("cwm", "abot"),
        default="cwm",
        help="Output DUV convention. 'cwm' is what the checkpoint was pretrained on: R is 0.3-256 m "
        "inverted log-z with near bright and invalid 0, G/B are CWM's twelve injective semantic "
        "codes. 'abot' reproduces the first cache (0.1-8000 m forward, sky 255, non-injective "
        "colours). Both differences are silent, so caches under the two cannot be compared.",
    )
    p.add_argument(
        "--source-near",
        type=float,
        default=SOURCE_NEAR,
        help=f"depth.mp4's near plane in metres, used only when metadata.json declares none "
        f"(default {SOURCE_NEAR}, DATA_F.md).",
    )
    p.add_argument("--source-far", type=float, default=SOURCE_FAR, help="See --source-near.")
    add_filter_arguments(p)
    return p.parse_args()


def cwm_code(label: int) -> tuple[int, int]:
    """CWM label -> (G, B), matching ``cwm_h3_inference.duv.load_duv_frame``.

    u indexes on ``label % 4`` and v on ``label // 4``, so u varies fastest. Building the grid as
    a product in the other order gives twelve distinct pairs that are still the wrong twelve.
    """
    if not 0 <= label <= 11:
        raise ValueError(f"CWM semantic label must be in [0, 11]: {label}")
    return CWM_SEMANTIC_U[label % 4], CWM_SEMANTIC_V[label // 4]


def build_palette(kind: str, class_ids: list[int]) -> dict[int, tuple[int, int]]:
    """GTA class id -> (G, B). All eleven labels get distinct codes."""
    if kind == "abot":
        return dict(ABOT_GB)
    unknown = [class_id for class_id in class_ids if class_id not in GTA_TO_CWM]
    if unknown:
        raise SystemExit(f"semantic.json declares classes {unknown} that GTA_TO_CWM has no entry for. "
                         "Add them rather than letting them fall through to a neighbouring code.")
    return {class_id: cwm_code(GTA_TO_CWM[class_id]) for class_id in class_ids}


def read_source_depth_range(seg: Path, near: float, far: float) -> tuple[float, float, str]:
    """Decode range for ``depth.mp4``: whatever ``metadata.json`` declares, else the arguments.

    Hardcoding DATA_F's 0.1-256 m was safe for the corpus it was written against and is a silent
    error on any other. A wrong range is not a visible failure -- the depth map still looks like a
    depth map, it is just a monotone relabelling of one, so every parallax cue the proxy is supposed
    to carry is miscalibrated.
    """
    path = seg / "metadata.json"
    if not path.is_file():
        return near, far, "DATA_F default (no metadata.json)"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return near, far, f"DATA_F default (metadata.json unreadable: {error})"

    found: dict[str, float] = {}

    def walk(node: object, trail: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            label = f"{trail}.{key}" if trail else str(key)
            lowered = label.lower()
            if isinstance(value, int | float) and not isinstance(value, bool) and "depth" in lowered:
                if "near" in lowered and "near" not in found:
                    found["near"] = float(value)
                elif ("far" in lowered or "max" in lowered) and "far" not in found:
                    found["far"] = float(value)
            walk(value, label)

    walk(payload, "")
    if "near" in found and "far" in found and 0 < found["near"] < found["far"]:
        return found["near"], found["far"], f"{path.name}"
    return near, far, f"DATA_F default ({path.name} declares no depth near/far)"


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
    return sorted(ABOT_GB)


def decode_depth_grey(grey: np.ndarray, near: float, far: float) -> np.ndarray:
    """``depth.mp4`` grey -> metres. Inverted log-z, near bright, 0 invalid (DATA_F.md)."""
    span = math.log(far) - math.log(near)
    metres = np.exp(math.log(far) - (grey.astype(np.float64) / 255.0) * span)
    return np.where(grey == 0, 0.0, metres).astype(np.float32)


def encode_depth_cwm(metres: np.ndarray) -> np.ndarray:
    """Metres -> CWM's R plane. Near bright, invalid 0.

    Quantised to uint16 and divided by 257 exactly as ``load_duv_frame`` does for its Qwen plane,
    so the bytes here are the bytes the checkpoint was pretrained on rather than a re-derivation
    that agrees to within a rounding step.
    """
    valid = metres > 1.0e-3
    codes = np.zeros(metres.shape, dtype=np.uint16)
    if bool(np.any(valid)):
        clipped = np.clip(metres[valid].astype(np.float64), CWM_NEAR, CWM_FAR)
        normalized = (math.log(CWM_FAR) - np.log(clipped)) / (math.log(CWM_FAR) - math.log(CWM_NEAR))
        codes[valid] = np.floor(np.clip(normalized, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    return np.floor(codes.astype(np.float64) / 257.0 + 0.5).astype(np.uint8)


def encode_depth_abot(metres: np.ndarray) -> np.ndarray:
    """Metres -> ABot's R plane. Forward log-z, 255 reserved for sky by the caller."""
    span = math.log(ABOT_FAR) - math.log(ABOT_NEAR)
    clipped = np.clip(metres.astype(np.float64), ABOT_NEAR, ABOT_FAR)
    fraction = 1.0 - (math.log(ABOT_FAR) - np.log(clipped)) / span
    return np.rint(fraction * ABOT_MAX).astype(np.uint8)


def compose_frame(metres: np.ndarray, ids: np.ndarray, palette: dict[int, tuple[int, int]], *,
                  convention: str) -> np.ndarray:
    sky = (metres <= 1.0e-3) | (ids == GTA_SKY_ID)
    if convention == "abot":
        red = encode_depth_abot(metres)
        red[sky] = ABOT_SKY
    else:
        red = encode_depth_cwm(metres)
        # CWM stores zero for invalid depth, and the sky never returns a hit. Forced rather than
        # left to the decode so a source that wrote a finite distance for sky cannot leak a
        # foreground code into it.
        red[sky] = 0
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
                                               seg / "proxy" / args.out_name)
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
            palette = build_palette(args.convention, read_class_ids(seg))
            near, far, origin = read_source_depth_range(seg, args.source_near, args.source_far)
            ids0 = semantic_ids(semantic_frames[0])
            metres0 = decode_depth_grey(depth_frames[0], near, far)
            if index == 0 or args.probe_only:
                distinct = len(set(palette.values()))
                red_scale = ("0.3-256 m inverted, near bright, invalid 0"
                             if args.convention == "cwm" else "0.1-8000 m forward, sky 255")
                collapsed = "" if distinct == len(palette) else f"  [{len(palette) - distinct} collapsed]"
                print(f"  convention {args.convention}: R = {red_scale}")
                print(f"  decode range {near}-{far} m, from {origin}")
                print(f"  palette: {len(palette)} classes -> {distinct} distinct (G,B){collapsed}")
                valid = metres0 > 1.0e-3
                red0 = compose_frame(metres0, ids0, palette, convention=args.convention)[..., 0]
                print(f"{seg.name}: {depth_frames[0].shape[1]}x{depth_frames[0].shape[0]} "
                      f"{len(depth_frames)} frames, depth grey "
                      f"min={int(depth_frames[0].min())} max={int(depth_frames[0].max())}, "
                      f"metres p50={float(np.median(metres0[valid])) if np.any(valid) else 0:.2f}, "
                      f"semantic ids {sorted(int(x) for x in np.unique(ids0)[:16])}")
                # A healthy R plane spans most of 0..255. A narrow span means the decode range and
                # the output range disagree, which no amount of training recovers.
                print(f"  R plane: min={int(red0.min())} max={int(red0.max())} "
                      f"p50={int(np.median(red0))} distinct={len(np.unique(red0))}/256")
            if args.probe_only:
                continue
            resize = bool(args.out_width and args.out_height)
            composed = []
            for depth, semantic in zip(depth_frames, semantic_frames, strict=True):
                grey, ids = depth, semantic_ids(semantic)
                if resize:
                    grey = resample_nearest(grey, args.out_width, args.out_height)
                    ids = resample_nearest(ids, args.out_width, args.out_height)
                composed.append(
                    compose_frame(decode_depth_grey(grey, near, far), ids, palette,
                                  convention=args.convention))
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
