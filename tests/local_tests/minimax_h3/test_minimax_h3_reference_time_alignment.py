# SPDX-License-Identifier: Apache-2.0
"""What `time_aligned` moves, and what it must leave alone.

H3 packs references ahead of the target and clocks them sequentially, so the proxy sits a whole
clip earlier in rotary time than the frames it is meant to steer. These tests pin the alternative:
the proxy's frame k at the target's frame k, with everything else -- including the target's own
coordinates -- byte-identical to the default layout.
"""

from __future__ import annotations

import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_TEXT_TAG,
    build_ref2va_packed_sequence,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference

PATCH = (1, 2, 2)
# The released geometry: 768x1344 target over a 192x336 proxy, 124 frames -> 37 latent.
TARGET = {"num_latent_frames": 37, "latent_height": 48, "latent_width": 84}
PROXY = {"num_latent_frames": 37, "latent_height": 12, "latent_width": 22}
# Anchor ahead of the proxy: appearance dictionary, one latent frame on a 2048-short-edge canvas.
ANCHOR_ROWS = (128 // 2) * (224 // 2)
PROXY_ROWS = (PROXY["latent_height"] // 2) * (PROXY["latent_width"] // 2) * PROXY["num_latent_frames"]
TARGET_ROWS = (TARGET["latent_height"] // 2) * (TARGET["latent_width"] // 2) * TARGET["num_latent_frames"]
PROXY_ROWS_PER_FRAME = (PROXY["latent_height"] // 2) * (PROXY["latent_width"] // 2)
TARGET_ROWS_PER_FRAME = (TARGET["latent_height"] // 2) * (TARGET["latent_width"] // 2)


def _layout(*, aligned: bool, proxy: dict | None = None):
    proxy_geometry = dict(PROXY if proxy is None else proxy)
    return build_ref2va_packed_sequence(
        torch.full((16, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
        [
            MiniMaxH3PreparedReference(media_type="image", num_latent_frames=1, latent_height=128, latent_width=224),
            MiniMaxH3PreparedReference(media_type="video", time_aligned=aligned, **proxy_geometry),
        ],
        num_audio_latents=32,
        patch_size=PATCH,
        **TARGET,
    )


def _proxy_and_target_indices(layout):
    """Proxy and target video row indices, in pack order: anchor, proxy, target."""
    video = layout.video_indices
    assert video.shape[0] == ANCHOR_ROWS + PROXY_ROWS + TARGET_ROWS
    return video[ANCHOR_ROWS:ANCHOR_ROWS + PROXY_ROWS], video[ANCHOR_ROWS + PROXY_ROWS:]


def test_the_default_puts_the_proxy_a_whole_clip_before_the_target():
    layout = _layout(aligned=False)
    proxy_rows, target_rows = _proxy_and_target_indices(layout)
    proxy_time = layout.position_ids[proxy_rows, 0]
    target_time = layout.position_ids[target_rows, 0]
    # No proxy row shares a rotary instant with any target row, and the gap exceeds the target's own
    # span -- which is the mechanism the aligned variant exists to remove.
    assert float(proxy_time.max()) < float(target_time.min())
    gap = float(target_time.min()) - float(proxy_time.min())
    span = float(target_time.max()) - float(target_time.min())
    assert gap > span


def test_aligning_puts_proxy_frame_k_on_target_frame_k():
    layout = _layout(aligned=True)
    proxy_rows, target_rows = _proxy_and_target_indices(layout)
    proxy_frame_time = layout.position_ids[proxy_rows, 0].view(PROXY["num_latent_frames"], PROXY_ROWS_PER_FRAME)[:, 0]
    target_frame_time = layout.position_ids[target_rows, 0].view(TARGET["num_latent_frames"],
                                                                 TARGET_ROWS_PER_FRAME)[:, 0]
    # Exactly, not to within a tolerance: both go through the same sequential clock accumulation.
    assert torch.equal(proxy_frame_time, target_frame_time)


def test_aligning_leaves_the_target_and_the_anchor_untouched():
    """The whole point of advancing the clock anyway: only the proxy moves."""
    default, aligned = _layout(aligned=False), _layout(aligned=True)
    assert default.position_ids.shape == aligned.position_ids.shape
    proxy_rows, _ = _proxy_and_target_indices(default)
    before = torch.ones(default.position_ids.shape[0], dtype=torch.bool)
    before[proxy_rows] = False
    assert torch.equal(default.position_ids[before], aligned.position_ids[before])
    # And the proxy really did move, so the test above is not passing on an unchanged tensor.
    assert not torch.equal(default.position_ids[proxy_rows], aligned.position_ids[proxy_rows])


def test_alignment_does_not_touch_the_spatial_grid():
    """Spatially the two already agreed; this flag is about time only."""
    default, aligned = _layout(aligned=False), _layout(aligned=True)
    proxy_rows, _ = _proxy_and_target_indices(default)
    assert torch.equal(default.position_ids[proxy_rows, 1:], aligned.position_ids[proxy_rows, 1:])


def test_aligning_an_image_reference_is_refused():
    with pytest.raises(ValueError, match="only meaningful for a video reference"):
        build_ref2va_packed_sequence(
            torch.full((16, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            [
                MiniMaxH3PreparedReference(media_type="image",
                                           time_aligned=True,
                                           num_latent_frames=1,
                                           latent_height=128,
                                           latent_width=224),
                MiniMaxH3PreparedReference(media_type="video", **PROXY),
            ],
            num_audio_latents=32,
            patch_size=PATCH,
            **TARGET,
        )


def test_aligning_a_proxy_of_the_wrong_length_is_refused():
    """A shorter proxy would cover a prefix of the target's timeline and look aligned."""
    with pytest.raises(ValueError, match="target's latent frame count"):
        _layout(aligned=True, proxy={**PROXY, "num_latent_frames": 19})
