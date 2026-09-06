# SPDX-License-Identifier: Apache-2.0
"""CPU tests for side-by-side validation panel composition."""

import numpy as np
import pytest

from fastvideo.train.utils.video_panels import compose_comparison_video, resize_frames


def _clip(frame_count: int, height: int, width: int, value: int = 0) -> list[np.ndarray]:
    """Create a uniformly filled RGB clip."""
    return [np.full((height, width, 3), fill_value=value, dtype=np.uint8) for _ in range(frame_count)]


def test_panels_are_concatenated_left_to_right_with_separators() -> None:
    """Verify the composed width covers every panel plus the gaps between them."""
    composed = compose_comparison_video(
        [_clip(5, 64, 96), _clip(5, 64, 96), _clip(5, 64, 96)],
        separator_px=4,
    )

    assert len(composed) == 5
    assert composed[0].shape == (64, 96 * 3 + 4 * 2, 3)


def test_canvas_height_argument_overrides_the_first_panel() -> None:
    """A proxy listed first must not shrink the prediction it is compared against."""
    proxy = _clip(3, 192, 336)
    prediction = _clip(3, 768, 1344)

    composed = compose_comparison_video([proxy, prediction], separator_px=0, height=768)

    # Both panels are scaled to 768 tall, so the proxy widens from 336 to 1344.
    assert composed[0].shape == (768, 1344 * 2, 3)


def test_shortest_panel_bounds_the_result() -> None:
    """Truncation avoids showing held frames as though the model produced them."""
    composed = compose_comparison_video([_clip(9, 32, 32), _clip(4, 32, 32), _clip(7, 32, 32)])

    assert len(composed) == 4


@pytest.mark.parametrize(
    ("height", "width"),
    [(101, 101), (100, 101), (101, 100)],
)
def test_odd_dimensions_are_padded_for_chroma_subsampling(height: int, width: int) -> None:
    """H.264 4:2:0 rejects odd geometry, and the caller treats a failed write as no artifact."""
    composed = compose_comparison_video([_clip(2, height, width)], separator_px=0, height=height)

    assert composed[0].shape[0] % 2 == 0
    assert composed[0].shape[1] % 2 == 0


def test_labels_add_a_band_above_the_panels() -> None:
    """The band grows the frame vertically without disturbing the panel width."""
    unlabelled = compose_comparison_video([_clip(2, 220, 320)] * 3, separator_px=4)
    labelled = compose_comparison_video(
        [_clip(2, 220, 320)] * 3,
        labels=["proxy", "prediction", "target"],
        separator_px=4,
    )

    assert labelled[0].shape[1] == unlabelled[0].shape[1]
    assert labelled[0].shape[0] > unlabelled[0].shape[0]


def test_composed_frames_are_contiguous_uint8() -> None:
    """The MP4 encoder consumes these frames directly as rgb24."""
    composed = compose_comparison_video([_clip(3, 32, 48)] * 2, labels=["a", "b"])

    for frame in composed:
        assert frame.dtype == np.uint8
        assert frame.flags["C_CONTIGUOUS"]


def test_resize_frames_preserves_aspect_ratio() -> None:
    """Panels keep their geometry so a proxy is not stretched against its target."""
    resized = resize_frames(_clip(2, 50, 200), height=100)

    assert all(frame.shape == (100, 400, 3) for frame in resized)


@pytest.mark.parametrize(
    ("panels", "labels", "message"),
    [
        ([], None, "at least one panel"),
        ([[]], None, "no frames"),
        ([_clip(2, 8, 8)], ["a", "b"], "labels for"),
    ],
)
def test_invalid_input_is_rejected(panels: list, labels: list | None, message: str) -> None:
    """Verify the caller learns why a panel could not be built."""
    with pytest.raises(ValueError, match=message):
        compose_comparison_video(panels, labels=labels)


def test_non_rgb_frames_are_rejected() -> None:
    """A grayscale or channels-first panel would silently misrender."""
    with pytest.raises(ValueError, match="HxWx3 RGB"):
        compose_comparison_video([[np.zeros((4, 4), dtype=np.uint8)]])
