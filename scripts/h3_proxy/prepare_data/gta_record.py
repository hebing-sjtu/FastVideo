# SPDX-License-Identifier: Apache-2.0
"""Readers for the ``diffusionshader.dataset/v1`` GTA record layout (``gta_record_*``).

One origin per ``seg_*`` directory::

    seg_0218_drive/
      metadata.json            split, event tag, every node's files and video size
      proxy/color.mp4          native 1280x720 game RGB
      proxy/depth.mp4          h264-logz-gray8; quantization in proxy/track.json
      proxy/semantic.mp4       class id in B, R = G = 0; names in proxy/semantic.json
      proxy/captions/prompt.json
      <render_uid>/output.mp4  MiniMax-H3 restyle, 1344x768
      <render_uid>/captions/prompt.json
      <render_uid>/provenance.json

Shared by ``compose_gta_duv.py``, ``encode_proxy_samples.py`` and
``record_dir_to_omni_manifest.py`` so the three agree on depth, class ids and render geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

# Order of `fastvideo.pipelines.basic.minimax_h3.proxy.PROXY_SEMANTIC_CLASSES`, spelled out so the
# compose script runs without importing fastvideo.
PROXY_CLASSES = (
    "void_unknown",
    "sky",
    "water",
    "terrain",
    "road_paved",
    "vegetation",
    "building_structure",
    "infrastructure",
    "human",
    "animal",
    "vehicle",
    "prop",
)

# Source class name (the part before the first " / ") -> proxy class. The legacy eleven names map
# exactly as compose_gta_duv's former id table did, so older corpora compose to the same bytes.
# gta_record merges sky, buildings, road, ground and vegetation into one "static world" id; it is
# not sky, so it keeps its depth and takes the otherwise unused void_unknown code.
SOURCE_CLASS_TO_PROXY = {
    "static world": "void_unknown",
    "sky": "sky",
    "player": "human",
    "ped": "animal",
    "vehicle": "vehicle",
    "building": "building_structure",
    "road": "road_paved",
    "ground": "infrastructure",
    "vegetation": "vegetation",
    "terrain": "terrain",
    "water": "water",
    "prop": "prop",
}

UNMAPPED = 255


@dataclass(frozen=True, slots=True)
class DepthQuantization:
    """How ``depth.mp4`` grey codes map to metres. Near is bright and grey 0 is invalid."""

    near: float
    far: float
    # 'record': codes 1..255 span [far, near], from track.json's formula.
    # 'legacy': grey / 255 spans [far, near] (DATA_F.md).
    codes: str
    origin: str


def source_class_key(name: str) -> str:
    return name.split("/")[0].strip().lower()


def read_semantic_map(proxy_dir: Path) -> dict[int, str]:
    """Source class id -> name from ``proxy/semantic.json``."""
    path = proxy_dir / "semantic.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"{path} declares no classes")
    return {int(key): str(value.get("name") if isinstance(value, dict) else value) for key, value in classes.items()}


def proxy_class_lut(semantic_map: dict[int, str]) -> np.ndarray:
    """``[256]`` uint8 table from a source class id to a ``PROXY_CLASSES`` index.

    Ids the map does not declare stay :data:`UNMAPPED`, which :func:`map_class_ids` rejects, so a
    stray code cannot quietly borrow a neighbour's class.
    """
    lut = np.full(256, UNMAPPED, dtype=np.uint8)
    unknown = []
    for class_id, name in semantic_map.items():
        target = SOURCE_CLASS_TO_PROXY.get(source_class_key(name))
        if target is None:
            unknown.append(f"{class_id}={name!r}")
            continue
        lut[class_id] = PROXY_CLASSES.index(target)
    if unknown:
        raise ValueError(f"semantic.json classes with no proxy mapping: {', '.join(unknown)}. Add them to "
                         "SOURCE_CLASS_TO_PROXY rather than letting them fall onto another code.")
    return lut


def sky_source_ids(semantic_map: dict[int, str]) -> set[int]:
    return {class_id for class_id, name in semantic_map.items() if SOURCE_CLASS_TO_PROXY.get(source_class_key(name)) == "sky"}


def map_class_ids(ids: np.ndarray, lut: np.ndarray) -> np.ndarray:
    mapped = lut[ids]
    if bool(np.any(mapped == UNMAPPED)):
        stray = sorted(int(value) for value in np.unique(ids[mapped == UNMAPPED])[:8])
        raise ValueError(f"semantic frames carry ids {stray} that semantic.json does not declare")
    return mapped


def read_depth_quantization(proxy_dir: Path, *, near: float = 0.1, far: float = 256.0) -> DepthQuantization:
    """``track.json``'s depth quantization, else the DATA_F.md defaults."""
    path = proxy_dir / "track.json"
    if path.is_file():
        depth = json.loads(path.read_text(encoding="utf-8")).get("depth") or {}
        quantization = depth.get("quantization") or {}
        if "clipNear" in quantization and "clipFar" in quantization:
            if depth.get("encoding") != "h264-logz-gray8" or int(quantization.get("invalidCode", 0)) != 0:
                raise ValueError(f"{path}: unsupported depth encoding {depth.get('encoding')!r} / invalidCode "
                                 f"{quantization.get('invalidCode')!r}")
            return DepthQuantization(float(quantization["clipNear"]), float(quantization["clipFar"]), "record",
                                     path.name)
    return DepthQuantization(near, far, "legacy", "DATA_F default")


def decode_depth_codes(grey: np.ndarray, quantization: DepthQuantization) -> np.ndarray:
    """Grey codes -> metres, 0 where invalid.

    ``record`` inverts ``g = floor(1 + 254 * (1 - log(z/near) / log(far/near)) + 0.5)``, so 255 is
    the near plane and 1 the far plane. ``legacy`` is DATA_F.md's ``g / 255`` ramp.
    """
    span = math.log(quantization.far) - math.log(quantization.near)
    codes = grey.astype(np.float64)
    fraction = (codes - 1.0) / 254.0 if quantization.codes == "record" else codes / 255.0
    metres = np.exp(math.log(quantization.far) - np.clip(fraction, 0.0, 1.0) * span)
    return np.where(grey == 0, 0.0, metres).astype(np.float32)


def render_crop_box(render_size: tuple[int, int], source_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """``(left, top, right, bottom)`` of a render that shows exactly the source's field of view.

    MiniMax-H3 renders a 1280x720 source at 1344x768 by matching width -- the source spans all 1344
    columns and 756 rows -- and invents about six rows above and below. Measured on gta_record_0930
    by edge correlation of each render's first frame against its 1280x720 look image: width-match
    beat a plain stretch on 34 of 40 renders. Cropping those rows puts the render in the source's
    frame, so it lines up with depth and semantic after the same center crop.
    """
    (render_w, render_h), (source_w, source_h) = render_size, source_size
    content_h = round(source_h * render_w / source_w)
    if content_h > render_h:
        content_w = round(source_w * render_h / source_h)
        left = (render_w - content_w) // 2
        return left, 0, left + content_w, render_h
    top = (render_h - content_h) // 2
    return 0, top, render_w, top + content_h


def _decode(path: Path, pixel_format: str, num_frames: int) -> np.ndarray:
    import av

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format=pixel_format))
            if len(frames) >= num_frames:
                break
    if len(frames) < num_frames:
        raise ValueError(f"{path} has {len(frames)} frames; {num_frames} are required")
    return np.stack(frames)


def read_depth_video(path: Path, num_frames: int, quantization: DepthQuantization) -> np.ndarray:
    """``[T, H, W]`` metres from a log-z grey ``depth.mp4``."""
    return decode_depth_codes(_decode(path, "gray", num_frames), quantization)


def read_semantic_video(path: Path, num_frames: int, lut: np.ndarray) -> np.ndarray:
    """``[T, H, W]`` ``PROXY_CLASSES`` indices from a ``semantic.mp4`` whose B channel is the class id."""
    frames = _decode(path, "rgb24", num_frames)
    if int(frames[..., :2].max()) != 0:
        raise ValueError(f"{path}: R/G are not zero, so B is not a bare class id")
    return map_class_ids(frames[..., 2], lut)


def read_proxy_planes_from_videos(depth_path: Path, semantic_path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Metric depth and proxy class indices of one record seg, each ``[T, H, W]``.

    ``semantic.json`` and ``track.json`` are read from the videos' directory.
    """
    lut = proxy_class_lut(read_semantic_map(semantic_path.parent))
    depth = read_depth_video(depth_path, num_frames, read_depth_quantization(depth_path.parent))
    semantic = read_semantic_video(semantic_path, num_frames, lut)
    if depth.shape != semantic.shape:
        raise ValueError(f"depth {depth.shape} and semantic {semantic.shape} differ for {depth_path.parent}")
    return depth, semantic
