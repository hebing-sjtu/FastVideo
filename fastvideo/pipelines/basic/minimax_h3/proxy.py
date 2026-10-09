# SPDX-License-Identifier: Apache-2.0
"""Proxy-render conditioning for MiniMax-H3.

A "proxy" is a cheap render of the scene we want a photoreal video of: untextured geometry from a
game engine, a blockout, a low-poly stand-in. It carries layout, occlusion and motion but none of
the appearance. This module packs one into the three channels of an ordinary RGB video so that H3's
Ref2VA path can carry it as a video reference, with no change to the transformer.

The packing is *DUV* — one depth channel and two semantic channels:

* **D**: metric depth, log-normalized between :data:`PROXY_DEPTH_NEAR_METRES` and
  :data:`PROXY_DEPTH_FAR_METRES` and inverted so near is bright. Log spacing rather than linear
  because depth error that matters is relative: 10 cm at arm's length is a different surface, 10 cm
  at 200 m is the same one. Invalid depth stores zero, which reads as "infinitely far" — the safe
  direction, since a spurious far surface is ignored while a spurious near one occludes.
* **U**, **V**: the semantic class id, split across two channels by a 4x3 lookup rather than written
  as a single ramp. A ramp would put ``vegetation`` numerically adjacent to ``road`` and let the VAE
  blur one into the other; two channels at well-separated levels give every class a corner of the
  UV square that survives lossy encoding.

The result is a ``[0, 1]`` float image that goes through H3's video VAE on exactly the path a real
RGB reference takes. Callers that already have an RGB proxy render and no depth can skip all of this
and pass the render straight through — the reference slot does not care how the three channels were
produced, only that the same convention is used at cache time and at inference.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# Depth is clipped to this range before log-normalization. The near plane also sets the resolution
# floor: below it every surface collapses to full brightness.
PROXY_DEPTH_NEAR_METRES = 0.3
PROXY_DEPTH_FAR_METRES = 256.0
# Depth at or below this reads as "no surface here" rather than "a surface 0 mm away". Rasterizers
# write zero into pixels the depth pass never touched, and those must not become the nearest thing
# in the frame.
PROXY_DEPTH_VALID_EPSILON_METRES = 1.0e-3

# 12 classes on a 4x3 grid of the UV square. The levels are interior — neither 0 nor 255 — so that
# VAE ringing at a class boundary cannot push a code outside the range its neighbours occupy.
PROXY_SEMANTIC_U = (32, 96, 160, 224)
PROXY_SEMANTIC_V = (43, 128, 213)
PROXY_SEMANTIC_CLASSES = (
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
PROXY_SEMANTIC_NUM_CLASSES = len(PROXY_SEMANTIC_CLASSES)


def encode_depth(metric_depth: np.ndarray) -> np.ndarray:
    """Log-normalize metric depth into the inverted ``[0, 1]`` channel the proxy pack uses.

    Args:
        metric_depth: Float array of depth in metres. Non-finite or negative values are rejected;
            values at or below :data:`PROXY_DEPTH_VALID_EPSILON_METRES` are treated as invalid.

    Returns:
        Float32 array in ``[0, 1]``, same shape, with 1 at the near plane and 0 at the far plane and
        at every invalid pixel.
    """
    metric = np.asarray(metric_depth, dtype=np.float64)
    if not bool(np.all(np.isfinite(metric))) or bool(np.any(metric < 0.0)):
        raise ValueError("Proxy depth must be finite and non-negative metres.")

    normalized = np.zeros(metric.shape, dtype=np.float32)
    valid = metric > PROXY_DEPTH_VALID_EPSILON_METRES
    if bool(np.any(valid)):
        clipped = np.clip(metric[valid], PROXY_DEPTH_NEAR_METRES, PROXY_DEPTH_FAR_METRES)
        span = math.log(PROXY_DEPTH_FAR_METRES) - math.log(PROXY_DEPTH_NEAR_METRES)
        normalized[valid] = np.clip((math.log(PROXY_DEPTH_FAR_METRES) - np.log(clipped)) / span, 0.0, 1.0)
    return normalized


def encode_semantic_ids(semantic_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split integer class ids into the two ``[0, 1]`` UV channels.

    Args:
        semantic_ids: Integer array with values in ``[0, PROXY_SEMANTIC_NUM_CLASSES)``.

    Returns:
        ``(u, v)`` float32 arrays in ``[0, 1]``, same shape as the input.
    """
    ids = np.asarray(semantic_ids)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= PROXY_SEMANTIC_NUM_CLASSES):
        raise ValueError(f"Proxy semantic ids must lie in [0, {PROXY_SEMANTIC_NUM_CLASSES}), got "
                         f"[{int(ids.min())}, {int(ids.max())}].")
    ids = ids.astype(np.int64)
    grid_width = len(PROXY_SEMANTIC_U)
    u = np.asarray(PROXY_SEMANTIC_U, dtype=np.float32)[ids % grid_width] / np.float32(255.0)
    v = np.asarray(PROXY_SEMANTIC_V, dtype=np.float32)[ids // grid_width] / np.float32(255.0)
    return u, v


def pack_duv_frame(metric_depth: np.ndarray, semantic_ids: np.ndarray) -> np.ndarray:
    """Pack one depth map and one semantic-id map into a ``[H, W, 3]`` float32 image in ``[0, 1]``."""
    if metric_depth.shape != semantic_ids.shape:
        raise ValueError(f"Proxy depth {metric_depth.shape} and semantic ids {semantic_ids.shape} must have the "
                         "same shape; they are two views of one render.")
    depth = encode_depth(metric_depth)
    u, v = encode_semantic_ids(semantic_ids)
    return np.stack((depth, u, v), axis=-1)


# The separate semantic reference has all three channels to itself, so the 12 classes go on a
# 3x2x2 lattice of the RGB cube rather than DUV's 4x3 UV square: the nearest two classes are 85
# levels apart instead of 64, with every level still interior for the same ringing reason.
PROXY_SEMANTIC_R = (43, 128, 213)
PROXY_SEMANTIC_G = (64, 192)
PROXY_SEMANTIC_B = (64, 192)

# What each separate proxy reference is called in manifests, cache metadata and configs.
PROXY_REFERENCE_KINDS = ("duv", "depth", "semantic")
PROXY_VARIANT_KINDS = (*PROXY_REFERENCE_KINDS, "style")


def semantic_palette() -> np.ndarray:
    """``[PROXY_SEMANTIC_NUM_CLASSES, 3]`` uint8 colours of the separate semantic reference."""
    ids = np.arange(PROXY_SEMANTIC_NUM_CLASSES)
    return np.stack(
        (
            np.asarray(PROXY_SEMANTIC_R)[ids % 3],
            np.asarray(PROXY_SEMANTIC_G)[(ids // 3) % 2],
            np.asarray(PROXY_SEMANTIC_B)[ids // 6],
        ),
        axis=-1,
    ).astype(np.uint8)


def depth_frame_pixels(metric_depth: np.ndarray) -> np.ndarray:
    """One depth map as a grey ``[H, W, 3]`` float32 image: DUV's D channel in all three."""
    depth = encode_depth(metric_depth)
    return np.repeat(depth[..., None], 3, axis=-1)


def semantic_frame_pixels(semantic_ids: np.ndarray) -> np.ndarray:
    """One class-id map as a flat-colour ``[H, W, 3]`` float32 image in ``[0, 1]``."""
    ids = np.asarray(semantic_ids)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= PROXY_SEMANTIC_NUM_CLASSES):
        raise ValueError(f"Proxy semantic ids must lie in [0, {PROXY_SEMANTIC_NUM_CLASSES}), got "
                         f"[{int(ids.min())}, {int(ids.max())}].")
    return semantic_palette()[ids.astype(np.int64)].astype(np.float32) / np.float32(255.0)


def proxy_reference_clip(kind: str, depth_frames: np.ndarray, semantic_frames: np.ndarray) -> torch.Tensor:
    """Build one proxy reference's ``[1, 3, T, H, W]`` VAE input from depth and class ids.

    ``duv`` is the packed single reference; ``depth`` and ``semantic`` are the two halves of it as
    separate references, each with the full three channels.
    """
    if kind == "duv":
        return pack_duv_clip(depth_frames, semantic_frames)
    if depth_frames.shape != semantic_frames.shape or depth_frames.ndim != 3:
        raise ValueError("A proxy clip needs matching [frames, height, width] depth and semantic arrays, got "
                         f"{depth_frames.shape} and {semantic_frames.shape}.")
    if kind == "depth":
        frames = np.stack([depth_frame_pixels(depth) for depth in depth_frames])
    elif kind == "semantic":
        frames = np.stack([semantic_frame_pixels(ids) for ids in semantic_frames])
    else:
        raise ValueError(f"Unknown proxy reference {kind!r}; expected one of {list(PROXY_REFERENCE_KINDS)}.")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(torch.float32)
    return tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def center_crop_box(source: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    """Top-left ``(top, left)`` of a ``target`` ``(H, W)`` window centred in ``source`` ``(H, W)``.

    Cropping, never scaling: depth and class ids cannot be interpolated, so every reference of a
    sample is cut from the same box and stays pixel-aligned with the target.
    """
    (source_h, source_w), (target_h, target_w) = source, target
    if target_h > source_h or target_w > source_w:
        raise ValueError(f"Cannot center-crop {target_w}x{target_h} out of a {source_w}x{source_h} frame.")
    return (source_h - target_h) // 2, (source_w - target_w) // 2


def read_raw_depth(path: str | Path, *, height: int, width: int) -> np.ndarray:
    """Read a headerless C-order little-endian float32 depth map of known shape."""
    payload = Path(path).read_bytes()
    expected = height * width * 4
    if len(payload) != expected:
        raise ValueError(f"Raw depth {path!s} must be exactly {expected} bytes for {height}x{width}, "
                         f"got {len(payload)}.")
    return np.frombuffer(payload, dtype="<f4").reshape(height, width)


def read_semantic_png(path: str | Path, *, height: int, width: int) -> np.ndarray:
    """Read an 8-bit grayscale semantic-id PNG of known shape."""
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (width, height):
            raise ValueError(f"Semantic ids {path!s} must be an 8-bit grayscale {width}x{height} PNG, got "
                             f"mode {image.mode} at {image.size[0]}x{image.size[1]}.")
        return np.asarray(image, dtype=np.uint8).copy()


def pack_duv_clip(
    depth_frames: np.ndarray,
    semantic_frames: np.ndarray,
) -> torch.Tensor:
    """Pack a clip of depth and semantic-id frames into the VAE's ``[1, 3, T, H, W]`` input layout.

    Returns a float32 tensor in ``[0, 1]``, which is the range
    :meth:`~fastvideo.models.vaes.minimax_h3_video.AutoencoderKLMiniMaxH3.normalize_pixels` expects —
    the same range an RGB reference reaches after dividing by 255.
    """
    if depth_frames.ndim != 3 or semantic_frames.ndim != 3:
        raise ValueError("A proxy clip needs [frames, height, width] depth and semantic arrays, got "
                         f"{depth_frames.shape} and {semantic_frames.shape}.")
    packed = np.stack(
        [pack_duv_frame(depth, semantic) for depth, semantic in zip(depth_frames, semantic_frames, strict=True)])
    tensor = torch.from_numpy(np.ascontiguousarray(packed)).to(torch.float32)
    return tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def rgb_clip_to_pixels(frames: np.ndarray) -> torch.Tensor:
    """Convert a uint8 ``[T, H, W, 3]`` proxy render into the same ``[1, 3, T, H, W]`` VAE input."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"A proxy RGB clip needs [frames, height, width, 3], got {frames.shape}.")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(torch.float32).div_(255.0)
    return tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


__all__ = [
    "PROXY_DEPTH_FAR_METRES",
    "PROXY_DEPTH_NEAR_METRES",
    "PROXY_DEPTH_VALID_EPSILON_METRES",
    "PROXY_REFERENCE_KINDS",
    "PROXY_VARIANT_KINDS",
    "PROXY_SEMANTIC_B",
    "PROXY_SEMANTIC_CLASSES",
    "PROXY_SEMANTIC_G",
    "PROXY_SEMANTIC_NUM_CLASSES",
    "PROXY_SEMANTIC_R",
    "PROXY_SEMANTIC_U",
    "PROXY_SEMANTIC_V",
    "center_crop_box",
    "depth_frame_pixels",
    "encode_depth",
    "encode_semantic_ids",
    "pack_duv_clip",
    "pack_duv_frame",
    "proxy_reference_clip",
    "read_raw_depth",
    "read_semantic_png",
    "rgb_clip_to_pixels",
    "semantic_frame_pixels",
    "semantic_palette",
]
