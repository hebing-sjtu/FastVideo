# SPDX-License-Identifier: Apache-2.0
"""Compose several clips into one side-by-side comparison video.

A conditional generator is judged by whether its output follows its condition, which a prediction
shown on its own cannot answer: the viewer has to hold the proxy and the target in mind and scrub
three players in sync. Concatenating the frames horizontally makes the comparison a property of a
single artifact, which is also the only form a tracker can keep aligned across steps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

# Dark enough to read as a gap against game footage, which is rarely this flat.
SEPARATOR_COLOR = (24, 24, 24)
LABEL_BACKGROUND = (16, 16, 16)
LABEL_FOREGROUND = (240, 240, 240)


def _load_label_font(pixel_height: int):
    """Best available font at roughly ``pixel_height``, or None if text cannot be drawn.

    Pillow's bundled bitmap font is ~11px tall, unreadable over a 768px panel, and only newer
    versions accept a size for it. Labels are a convenience, so every step here is allowed to fail:
    the caller falls back to a bare composite and the panel order still reaches the viewer through
    the artifact caption.
    """
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, pixel_height)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=pixel_height)
    except TypeError:
        # Pillow < 10.1: load_default takes no size.
        try:
            return ImageFont.load_default()
        except OSError:
            return None
    except OSError:
        return None


def _label_band(labels: Sequence[str], widths: Sequence[int], separator_px: int, band_height: int) -> np.ndarray | None:
    """Render a static caption strip whose text sits centred over each panel."""
    from PIL import ImageDraw

    total_width = sum(widths) + separator_px * (len(widths) - 1)
    band = Image.new("RGB", (total_width, band_height), LABEL_BACKGROUND)
    font = _load_label_font(max(10, int(band_height * 0.62)))
    if font is None:
        return None
    draw = ImageDraw.Draw(band)
    offset = 0
    for label, width in zip(labels, widths, strict=True):
        try:
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_width, text_height = right - left, bottom - top
        except AttributeError:
            # Pillow < 8 has no textbbox; centring degrades to left-aligned.
            text_width, text_height = 0, 0
        draw.text(
            (offset + max(0, (width - text_width) // 2), max(0, (band_height - text_height) // 2)),
            label,
            fill=LABEL_FOREGROUND,
            font=font,
        )
        offset += width + separator_px
    return np.asarray(band, dtype=np.uint8)


def resize_frames(frames: Sequence[np.ndarray], height: int) -> list[np.ndarray]:
    """Scale every frame to ``height``, preserving aspect ratio."""
    resized: list[np.ndarray] = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"A panel frame must be HxWx3 RGB, got {tuple(frame.shape)}.")
        if frame.shape[0] == height:
            resized.append(np.ascontiguousarray(frame))
            continue
        width = max(1, round(frame.shape[1] * height / frame.shape[0]))
        resized.append(
            np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8))
    return resized


def compose_comparison_video(
    panels: Sequence[Sequence[np.ndarray]],
    *,
    labels: Sequence[str] | None = None,
    separator_px: int = 4,
    height: int | None = None,
) -> list[np.ndarray]:
    """Lay out clips left to right as one uint8 RGB clip.

    ``height`` sets the canvas and should be the prediction's, which is the resolution under
    judgement. It is worth passing explicitly rather than relying on the default of the first
    panel's: a natural reading order puts the proxy first, and a proxy is deliberately encoded at a
    quarter of the target's edge length, so that default would downscale the very thing being
    evaluated.

    The result is truncated to the shortest panel rather than padded. A held last frame would show
    the proxy diverging from the prediction at the tail, which is an artifact of the mismatch and
    not something the model did.
    """
    if not panels:
        raise ValueError("compose_comparison_video needs at least one panel.")
    for index, panel in enumerate(panels):
        if len(panel) == 0:
            raise ValueError(f"Panel {index} has no frames.")
    if labels is not None and len(labels) != len(panels):
        raise ValueError(f"Got {len(labels)} labels for {len(panels)} panels.")

    height = int(height) if height else panels[0][0].shape[0]
    if height <= 0:
        raise ValueError(f"Panel height must be positive, got {height}.")
    num_frames = min(len(panel) for panel in panels)
    scaled = [resize_frames(panel[:num_frames], height) for panel in panels]
    widths = [panel[0].shape[1] for panel in scaled]

    band = _label_band(labels, widths, separator_px, max(18, height // 22)) if labels else None
    gap = (np.full((height, separator_px, 3), SEPARATOR_COLOR, dtype=np.uint8) if separator_px > 0 else None)

    composed: list[np.ndarray] = []
    for frame_index in range(num_frames):
        pieces: list[np.ndarray] = []
        for panel_index, panel in enumerate(scaled):
            if panel_index and gap is not None:
                pieces.append(gap)
            pieces.append(panel[frame_index])
        frame = np.concatenate(pieces, axis=1)
        if band is not None:
            frame = np.concatenate([band, frame], axis=0)
        composed.append(np.ascontiguousarray(_pad_to_even(frame)))
    return composed


def _pad_to_even(frame: np.ndarray) -> np.ndarray:
    """Grow odd dimensions by a pixel so H.264 4:2:0 can encode the result.

    Chroma is subsampled by two in each axis, so libx264 rejects an odd width or height outright.
    Panel widths and a label band are all derived from arbitrary source sizes, and the caller that
    encodes this treats a failed write as a skipped artifact -- an odd dimension would drop the
    comparison video with no visible cause.
    """
    height, width = frame.shape[:2]
    pad_h, pad_w = height % 2, width % 2
    if not (pad_h or pad_w):
        return frame
    padded = np.full((height + pad_h, width + pad_w, 3), SEPARATOR_COLOR, dtype=np.uint8)
    padded[:height, :width] = frame
    return padded


__all__ = ["compose_comparison_video", "resize_frames"]
