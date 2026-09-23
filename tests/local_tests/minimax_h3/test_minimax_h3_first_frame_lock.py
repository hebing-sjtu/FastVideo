# SPDX-License-Identifier: Apache-2.0
"""Holding leading target latent frames as a given: row grouping, canvas, and regime contract."""

import numpy as np
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
    MINIMAX_H3_GIVEN_FRAMES_KEY,
    MiniMaxH3InputPreparationStage,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.train.callbacks.minimax_h3_proxy_validation import _ordered_visual_references

PATCH_SIZE = (1, 2, 2)


def test_validation_can_reverse_reference_order_without_changing_the_references():
    picture, video = object(), object()

    assert _ordered_visual_references(picture, video, "picture_video") == [picture, video]
    assert _ordered_visual_references(picture, video, "video_picture") == [video, picture]
    with pytest.raises(ValueError, match="reference_order"):
        _ordered_visual_references(picture, video, "unknown")


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


def test_given_rows_scale_with_the_prefix_length():
    """CWM's wn regime holds 10 latent frames, so the fixed row count has to be a multiple."""
    layout = _layout(num_latent_frames=4)
    rows_per_frame = target_rows_per_latent_frame(layout, PATCH_SIZE)
    given = 3
    timesteps = build_row_timesteps(layout, 0.25, 0.5, 0.999, 0.75, given * rows_per_frame)
    amounts = timesteps[0][timesteps[1]]

    start = layout.num_condition_video_rows
    fixed = layout.video_indices[start:start + given * rows_per_frame]
    denoised = layout.video_indices[start + given * rows_per_frame:]
    assert torch.all(amounts[fixed] == 0.999)
    assert torch.all(amounts[denoised] == 0.25)
    # One latent frame is left to predict out of four, and it is the last one.
    assert denoised.numel() == rows_per_frame


def test_given_frames_land_on_the_target_canvas():
    frames = np.zeros((6, 480, 640, 3), dtype=np.uint8)
    batch = ForwardBatch(data_type="video", prompt="x")
    batch.extra[MINIMAX_H3_GIVEN_FRAMES_KEY] = frames
    resized = MiniMaxH3InputPreparationStage._given_frames(batch, 768, 1344, 4)
    # Trimmed to the request and stretched onto the target canvas, as the anchor's path is.
    assert resized.shape == (4, 768, 1344, 3)


def test_given_frames_rejects_a_still_and_a_short_clip():
    batch = ForwardBatch(data_type="video", prompt="x")
    batch.extra[MINIMAX_H3_GIVEN_FRAMES_KEY] = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError, match="decoded RGB frames"):
        MiniMaxH3InputPreparationStage._given_frames(batch, 768, 1344, 4)

    batch.extra[MINIMAX_H3_GIVEN_FRAMES_KEY] = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="holds 3 frames"):
        MiniMaxH3InputPreparationStage._given_frames(batch, 768, 1344, 4)


def test_lock_first_frame_is_rejected_rather_than_read_as_one():
    """A boolean cannot express wn's 10 given frames, so silently reading it as 1 is a trap."""
    from fastvideo.train.callbacks.minimax_h3_proxy_validation import MiniMaxH3ProxyValidationCallback

    with pytest.raises(ValueError, match="num_given_latent_frames"):
        MiniMaxH3ProxyValidationCallback(lock_first_frame=True)


def test_a_multi_frame_prefix_requires_the_wn_prompt():
    from fastvideo.train.callbacks.minimax_h3_proxy_validation import MiniMaxH3ProxyValidationCallback

    with pytest.raises(ValueError, match="very beginning of the take"):
        MiniMaxH3ProxyValidationCallback(num_given_latent_frames=10, cwm_system_prompt="w0")


def _resolve(config, cwm_system):
    """``MiniMaxH3ProxyModel._resolve_given_latent_frames`` without building a 14-shard model.

    It reads two attributes and a dataloader batch, so a stand-in carries everything it needs.
    """
    from types import SimpleNamespace

    from fastvideo.train.models.minimax_h3.minimax_h3_proxy import (
        MiniMaxH3ProxyModel,
        _parse_given_latent_frames,
    )

    by_role, scalar = _parse_given_latent_frames(config)
    model = SimpleNamespace(_given_by_role=by_role, _num_given_latent_frames=scalar)
    info = {} if cwm_system is None else {"cwm_system": cwm_system}
    return MiniMaxH3ProxyModel._resolve_given_latent_frames(model, {"info_list": [info]})


@pytest.mark.parametrize(("cwm_system", "expected"), [("w0", 1), ("wn", 10)])
def test_a_mapping_takes_the_count_from_the_prompt_the_sample_was_encoded_with(cwm_system, expected):
    """One adapter serves both CWM windows, so one corpus has to be able to hold both contracts."""
    assert _resolve({"w0": 1, "wn": 10}, cwm_system) == expected


def test_a_mapping_refuses_a_sample_that_records_no_regime():
    # Silently picking either count would train half the corpus against the wrong promise.
    with pytest.raises(ValueError, match="records no `cwm_system`"):
        _resolve({"w0": 1, "wn": 10}, None)
    with pytest.raises(ValueError, match="records no `cwm_system`"):
        _resolve({"w0": 1, "wn": 10}, "none")


def test_a_mapping_refuses_a_regime_it_does_not_cover():
    with pytest.raises(ValueError, match="does not map"):
        _resolve({"wn": 10}, "w0")


def test_a_mapping_holds_each_regime_to_its_own_prompt():
    from fastvideo.train.models.minimax_h3.minimax_h3_proxy import _parse_given_latent_frames

    with pytest.raises(ValueError, match="its count is 1"):
        _parse_given_latent_frames({"w0": 10, "wn": 10})
    with pytest.raises(ValueError, match="ALREADY GIVEN"):
        _parse_given_latent_frames({"w0": 1, "wn": 1})
    with pytest.raises(ValueError, match="not roles a cache can record"):
        _parse_given_latent_frames({"w0": 1, "w1": 10})


def test_a_scalar_still_checks_itself_against_the_cache():
    assert _resolve(10, "wn") == 10
    assert _resolve(1, "w0") == 1
    # An unlabelled cache has no contract to disagree with, so the config stands.
    assert _resolve(4, None) == 4
    with pytest.raises(ValueError, match="ALREADY GIVEN"):
        _resolve(1, "wn")


@pytest.mark.parametrize("given", [0, 1])
def test_the_wn_prompt_requires_a_multi_frame_prefix(given):
    """The direction a step-0 baseline falls into: a wn cache sampled by a w0 config.

    Nothing downstream notices. The prompt promises thirty-four given frames, the rows supply one,
    and the panel reads as a bad checkpoint rather than as a misconfigured run.
    """
    from fastvideo.train.callbacks.minimax_h3_proxy_validation import MiniMaxH3ProxyValidationCallback

    with pytest.raises(ValueError, match="ALREADY GIVEN"):
        MiniMaxH3ProxyValidationCallback(num_given_latent_frames=given, cwm_system_prompt="wn")
