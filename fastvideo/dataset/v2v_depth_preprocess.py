# SPDX-License-Identifier: Apache-2.0
"""Pixel preprocessing shared by video-to-video + depth caching and validation.

The cached ``.pt`` latents that training consumes and the latents that
validation encodes on the fly must come out of an identical pixel pipeline;
otherwise the control channel a checkpoint learned to trust is not the control
channel it is evaluated against. Both paths therefore go through this module
rather than keeping their own crop/resize/depth-encoding code.

Two entry points per modality: a ``load_*`` variant that reads from disk and a
``frames_to_*`` variant for callers that already hold decoded frames (the
validation dataset decodes videos before a stage ever sees them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "crop_box",
    "frames_to_depth_clip",
    "frames_to_rgb_clip",
    "load_depth_clip",
    "load_rgb_clip",
    "read_frames",
    "validate_crop",
]


def validate_crop(
    crop_top: float,
    crop_bottom: float,
    crop_left: float,
    crop_right: float,
) -> None:
    """Reject crop fractions that cannot describe a non-empty frame."""
    for name, value in (
        ("crop_top", crop_top),
        ("crop_bottom", crop_bottom),
        ("crop_left", crop_left),
        ("crop_right", crop_right),
    ):
        if not 0.0 <= float(value) < 0.5:
            raise ValueError(f"{name} must be in [0, 0.5), got {value!r}")
    if crop_top + crop_bottom >= 1.0 or crop_left + crop_right >= 1.0:
        raise ValueError("crop fractions leave an empty frame")


def crop_box(
    width: int,
    height: int,
    *,
    crop_top: float,
    crop_bottom: float,
    crop_left: float,
    crop_right: float,
) -> tuple[int, int, int, int]:
    """Return a PIL-style ``(left, top, right, bottom)`` pixel box."""
    left = int(round(crop_left * width))
    top = int(round(crop_top * height))
    right = width - int(round(crop_right * width))
    bottom = height - int(round(crop_bottom * height))
    if right - left < 2 or bottom - top < 2:
        raise ValueError(f"crop emptied the frame: box=({left}, {top}, {right}, {bottom}) from {width}x{height}")
    return left, top, right, bottom


def read_frames(path: str | Path, num_frames: int) -> list[Any]:
    """Read a clip as a list of PIL images, from a video file or a frame directory."""
    from PIL import Image

    path = Path(path)
    if path.is_dir():
        frame_paths = sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".exr", ".tiff"}
        )
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {path}")
        frames = [Image.open(p) for p in frame_paths]
    else:
        from fastvideo.models.vision_utils import load_video

        frames = load_video(str(path))
    if len(frames) < num_frames:
        raise ValueError(f"{path} has {len(frames)} frames, need {num_frames}")
    return frames[:num_frames]


def frames_to_rgb_clip(
    frames: list[Any],
    *,
    num_frames: int,
    height: int,
    width: int,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
) -> torch.Tensor:
    """Convert decoded frames to an RGB clip ``[1, 3, T, H, W]`` in [-1, 1].

    The outer-edge crop runs before the resize to ``(width, height)`` so HUD
    bands are removed at the source aspect ratio rather than after squashing.
    """
    from PIL import Image

    validate_crop(crop_top, crop_bottom, crop_left, crop_right)
    if len(frames) < num_frames:
        raise ValueError(f"got {len(frames)} frames, need {num_frames}")

    tensors = []
    box = None
    for frame in frames[:num_frames]:
        frame = frame.convert("RGB")
        if box is None:
            box = crop_box(
                frame.size[0],
                frame.size[1],
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                crop_left=crop_left,
                crop_right=crop_right,
            )
        frame = frame.crop(box).resize((width, height), Image.BICUBIC)
        array = np.asarray(frame, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    clip = torch.stack(tensors, dim=1).unsqueeze(0)
    return clip * 2.0 - 1.0


def frames_to_depth_clip(
    frames: list[Any],
    *,
    num_frames: int,
    height: int,
    width: int,
    near: float,
    far: float,
    encoding: str = "disparity",
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
) -> torch.Tensor:
    """Convert decoded depth frames to a clip ``[1, 3, T, H, W]`` in [-1, 1].

    Accepts 16-bit PNG/TIFF frames carrying millimetres directly and 8-bit
    videos already carrying a normalized map. The distinction matters: a 16-bit
    source is remapped through the fixed near/far pair, while an 8-bit source is
    assumed to have been normalized by the renderer and is only rescaled.

    Depth is stored as an encoded *video* because the Wan VAE is the only
    encoder available and it expects 3-channel [-1, 1] input. Fixing the range
    (rather than per-clip min/max) is what keeps the control signal comparable
    across clips and across chunks of one streaming rollout.
    """
    from PIL import Image

    validate_crop(crop_top, crop_bottom, crop_left, crop_right)
    if encoding not in {"disparity", "linear"}:
        raise ValueError(f"encoding must be 'disparity' or 'linear', got {encoding!r}")
    if not 0.0 < near < far:
        raise ValueError(f"need 0 < near < far, got near={near!r} far={far!r}")
    if len(frames) < num_frames:
        raise ValueError(f"got {len(frames)} depth frames, need {num_frames}")

    tensors = []
    box = None
    for frame in frames[:num_frames]:
        array = np.asarray(frame)
        if array.ndim == 3:
            array = array[..., 0]
        if box is None:
            left, top, right, bottom = crop_box(
                array.shape[1],
                array.shape[0],
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                crop_left=crop_left,
                crop_right=crop_right,
            )
            box = (top, bottom, left, right)
        top, bottom, left, right = box
        array = array[top:bottom, left:right].astype(np.float32)
        if array.max() > 1.5:
            # Integer source: >255 means a 16-bit metric map, otherwise 8-bit
            # normalized. Both end up in [0, 1] with 1 == nearest.
            if array.max() > 255.0:
                metres = np.clip(array / 1000.0, near, far)
                if encoding == "disparity":
                    normalized = (1.0 / metres - 1.0 / far) / (1.0 / near - 1.0 / far)
                else:
                    normalized = 1.0 - (metres - near) / (far - near)
            else:
                normalized = array / 255.0
        else:
            normalized = array
        normalized = np.clip(normalized, 0.0, 1.0)
        image = Image.fromarray((normalized * 255.0).astype(np.uint8)).resize((width, height), Image.BILINEAR)
        single = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        tensors.append(single.unsqueeze(0).repeat(3, 1, 1))
    clip = torch.stack(tensors, dim=1).unsqueeze(0)
    return clip * 2.0 - 1.0


def load_rgb_clip(
    path: str | Path,
    *,
    num_frames: int,
    height: int,
    width: int,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
) -> torch.Tensor:
    """Read an RGB clip from disk as ``[1, 3, T, H, W]`` in [-1, 1]."""
    return frames_to_rgb_clip(
        read_frames(path, num_frames),
        num_frames=num_frames,
        height=height,
        width=width,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
    )


def load_depth_clip(
    path: str | Path,
    *,
    num_frames: int,
    height: int,
    width: int,
    near: float,
    far: float,
    encoding: str = "disparity",
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
) -> torch.Tensor:
    """Read a depth clip from disk as ``[1, 3, T, H, W]`` in [-1, 1]."""
    return frames_to_depth_clip(
        read_frames(path, num_frames),
        num_frames=num_frames,
        height=height,
        width=width,
        near=near,
        far=far,
        encoding=encoding,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
    )
