# SPDX-License-Identifier: Apache-2.0
"""Locking target latent frame 0 to the appearance anchor: row grouping and canvas."""

from PIL import Image
import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_TEXT_TAG,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    target_rows_per_latent_frame,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference, MiniMaxH3Reference
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import (
    MiniMaxH3InputPreparationStage, )

PATCH_SIZE = (1, 2, 2)


def _layout(num_latent_frames: int = 4, latent_height: int = 4, latent_width: int = 6):
    """A Ref2VA layout with one image reference ahead of the target."""
    references = [
        MiniMaxH3PreparedReference(media_type="image", num_latent_frames=1, latent_height=2, latent_width=2),
        MiniMaxH3PreparedReference(media_type="video", num_latent_frames=2, latent_height=2, latent_width=4),
    ]
    return build_ref2va_packed_sequence(
        torch.full((7, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
        references,
        num_latent_frames,
        latent_height,
        latent_width,
        3,
        PATCH_SIZE,
    )


def test_target_rows_per_latent_frame_counts_the_patched_grid():
    layout = _layout(latent_height=4, latent_width=6)
    assert target_rows_per_latent_frame(layout, PATCH_SIZE) == (4 // 2) * (6 // 2)


def test_target_rows_per_latent_frame_rejects_an_untileable_grid():
    layout = _layout(latent_height=4, latent_width=6)
    with pytest.raises(ValueError, match="not divisible by the spatial patch"):
        target_rows_per_latent_frame(layout, (1, 4, 4))


def test_locked_rows_join_the_condition_prefix_without_a_new_timestep():
    layout = _layout()
    rows_per_frame = target_rows_per_latent_frame(layout, PATCH_SIZE)
    plain = build_row_timesteps(layout, 0.25, 0.5, 0.999, 0.75)
    locked = build_row_timesteps(layout, 0.25, 0.5, 0.999, 0.75, rows_per_frame)

    # Same distinct amounts either way: a locked frame reuses the reference prefix's own.
    assert torch.equal(plain[0], locked[0])

    condition = layout.video_indices[:layout.num_condition_video_rows]
    fixed = layout.video_indices[layout.num_condition_video_rows:layout.num_condition_video_rows + rows_per_frame]
    denoised = layout.video_indices[layout.num_condition_video_rows + rows_per_frame:]
    amounts = locked[0][locked[1]]
    # The locked rows carry the reference prefix's own amount, which is what keeps them out of the
    # denoised set without introducing a group of their own. Compared against the amount rather than
    # against a slice of the prefix, which need not be as long as one target frame.
    assert fixed.numel() == rows_per_frame
    assert torch.all(amounts[condition] == 0.999)
    assert torch.all(amounts[fixed] == 0.999)
    assert torch.all(amounts[denoised] == 0.25)
    # Exactly one latent frame moved out of the denoised set.
    assert denoised.numel() == plain[1][layout.video_indices].numel() - layout.num_condition_video_rows - rows_per_frame


def test_build_row_timesteps_rejects_locking_more_than_the_target():
    layout = _layout()
    num_target_rows = layout.video_indices.numel() - layout.num_condition_video_rows
    with pytest.raises(ValueError, match="Cannot fix"):
        build_row_timesteps(layout, 0.25, 0.5, 0.999, 0.75, num_target_rows + 1)
    with pytest.raises(ValueError, match="must be non-negative"):
        build_row_timesteps(layout, 0.25, 0.5, 0.999, 0.75, -1)


def test_fixed_first_frame_lands_on_the_target_canvas_not_the_anchors():
    # An anchor whose own canvas differs from the target's on both axes, so reusing it would be
    # visible as a row-count mismatch rather than a silent resample.
    anchor = Image.new("RGB", (640, 480), color=(9, 9, 9))
    fixed = MiniMaxH3InputPreparationStage._fixed_first_frame(
        [MiniMaxH3Reference(source=anchor, media_type="image", short_edge=2048)],
        768,
        1344,
    )
    assert fixed.size == (1344, 768)


def test_fixed_first_frame_requires_an_image_reference():
    with pytest.raises(ValueError, match="no image"):
        MiniMaxH3InputPreparationStage._fixed_first_frame(
            [MiniMaxH3Reference(source="proxy.mp4", media_type="video")],
            768,
            1344,
        )
