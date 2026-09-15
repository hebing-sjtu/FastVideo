# SPDX-License-Identifier: Apache-2.0
"""Replicating the proxy onto the target latent grid for the control trunk.

The trunk is the one route by which a proxy can act per-token, and everything on the way in fails
quietly. A grid that does not divide would spread one proxy cell over a fractional number of target
cells, so the registration would vary across the frame -- while still producing a control signal
that trains. Replicating the *padded* reference latent would put the pad column inside the frame.
A trunk pointed at the anchor would train on a still. A sampler that skips the replication renders
from a trunk that saw nothing and reports a live checkpoint as dead.

So these cover the guards and the registration, not the arithmetic of the residual.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from fastvideo.models.dits.minimax_h3_camera_controlnet import (
    CAMERA_CONTROL_LATENT_KWARGS,
    CAMERA_CONTROL_MODALITIES,
)
from fastvideo.pipelines.basic.minimax_h3.packing import patchify_video_latents, replicate_latents_to_grid
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

PATCH_SIZE = (1, 2, 2)
# The released geometries: a 336x192 proxy is 12 x 21 latents, a 768x1344 target is 48 x 84.
PROXY_GRID = (12, 21)
TARGET_GRID = (48, 84)


def _latent(height: int, width: int, num_frames: int = 3) -> torch.Tensor:
    """A latent whose every cell is distinguishable, so replication is checkable cell by cell."""
    cells = torch.arange(float(num_frames * height * width)).reshape(1, 1, num_frames, height, width)
    return cells.expand(1, 24, num_frames, height, width).contiguous()


# --- replication, which is the whole mechanism ----------------------------------------------------


def test_each_proxy_cell_lands_on_the_block_of_target_cells_it_describes():
    """The VAE's stride of 16 makes proxy cell (i, j) cover target cells (4i..4i+3, 4j..4j+3)."""
    proxy = _latent(*PROXY_GRID)

    replicated = replicate_latents_to_grid(proxy, *TARGET_GRID)

    assert replicated.shape == (1, 24, 3, *TARGET_GRID)
    # Registration is exact, not approximate: every target cell holds the proxy cell over it.
    scale_h, scale_w = TARGET_GRID[0] // PROXY_GRID[0], TARGET_GRID[1] // PROXY_GRID[1]
    for i, j in ((0, 0), (1, 2), (11, 20)):
        block = replicated[0, 0, 0, i * scale_h:(i + 1) * scale_h, j * scale_w:(j + 1) * scale_w]
        assert torch.equal(block, proxy[0, 0, 0, i, j].expand(scale_h, scale_w))
    # No value is invented, which is what separates this from interpolating.
    assert torch.equal(torch.unique(replicated), torch.unique(proxy))


def test_the_replicated_grid_fills_the_target_rows_exactly():
    """The residual is an elementwise add onto target rows, so the counts have to match."""
    proxy_rows = patchify_video_latents(replicate_latents_to_grid(_latent(*PROXY_GRID), *TARGET_GRID), PATCH_SIZE)
    target_rows = patchify_video_latents(_latent(*TARGET_GRID), PATCH_SIZE)

    assert proxy_rows.shape == target_rows.shape


def test_a_grid_that_does_not_divide_is_refused():
    """704x1280 is 44 x 80 latents, which is 3.67x by 3.81x of the proxy's -- the cache gta_v2_cwm used."""
    with pytest.raises(ValueError, match="integer multiple"):
        replicate_latents_to_grid(_latent(*PROXY_GRID), 44, 80)


def test_the_padded_reference_latent_would_overshoot():
    """Why the unpadded latent is the one staged: 21 columns pad to 22, and 22 does not divide 84."""
    with pytest.raises(ValueError, match="integer multiple"):
        replicate_latents_to_grid(_latent(12, 22), *TARGET_GRID)


# --- the sampler's gating, which decides whether any of the above runs ----------------------------


def _stage(modalities: tuple[str, ...] | None) -> MiniMaxH3LatentPreparationStage:
    """A latent-preparation stage holding only what the replication step reads.

    Constructed without ``__init__`` on purpose: the real one builds a VAE and a text encoder, and
    this step needs neither -- which is the point of replicating latents instead of re-encoding.
    """
    stage = object.__new__(MiniMaxH3LatentPreparationStage)
    controlnet = None if modalities is None else types.SimpleNamespace(enabled_modalities=modalities)
    stage.transformer = types.SimpleNamespace(camera_controlnet=controlnet, patch_size=PATCH_SIZE)
    return stage


def _batch(latent_height: int = TARGET_GRID[0], latent_width: int = TARGET_GRID[1]) -> ForwardBatch:
    batch = ForwardBatch(data_type="video", num_frames=124)
    batch.raw_latent_shape = (1, 24, 3, latent_height, latent_width)
    return batch


def _proxy_reference(height: int = PROXY_GRID[0], width: int = PROXY_GRID[1]) -> MiniMaxH3PreparedReference:
    reference = MiniMaxH3PreparedReference(media_type="video", num_latent_frames=3, latent_height=height,
                                           latent_width=width)
    reference.latents = _latent(height, width)
    return reference


@pytest.mark.parametrize("modalities", [None, ("camera", )])
def test_replication_is_skipped_when_no_trunk_reads_the_proxy(modalities):
    """A plain backbone, or a trunk on the Plücker field alone, must get no control rows."""
    assert _stage(modalities)._control_proxy_rows([_proxy_reference()], _batch()) is None


def test_the_sampler_produces_one_control_row_per_target_row():
    rows = _stage(("proxy", ))._control_proxy_rows([_proxy_reference()], _batch())

    assert rows is not None
    assert rows.shape == (1, patchify_video_latents(_latent(*TARGET_GRID), PATCH_SIZE).shape[0], 24 * 2 * 2)


def test_a_target_grid_the_proxy_does_not_divide_is_refused_at_sampling_too():
    with pytest.raises(ValueError, match="integer multiple"):
        _stage(("proxy", ))._control_proxy_rows([_proxy_reference()], _batch(latent_height=44, latent_width=80))


def test_the_trunk_reads_the_proxy_and_not_the_anchor():
    """The anchor is an image reference, so it is not a candidate; a second video makes it ambiguous."""
    stage = _stage(("proxy", ))
    anchor = MiniMaxH3PreparedReference(media_type="image", num_latent_frames=1, latent_height=128, latent_width=224)
    anchor.latents = _latent(128, 224, num_frames=1)

    assert stage._control_proxy_rows([anchor, _proxy_reference()], _batch()) is not None
    for references in ([anchor], [anchor, _proxy_reference(), _proxy_reference()]):
        with pytest.raises(ValueError, match="exactly one video reference"):
            stage._control_proxy_rows(references, _batch())


def test_an_unencoded_reference_is_reported_rather_than_skipped():
    """`latents` is staged by condition encoding; None means the stages ran out of order."""
    reference = _proxy_reference()
    reference.latents = None

    with pytest.raises(ValueError, match="latents are missing"):
        _stage(("proxy", ))._control_proxy_rows([reference], _batch())


# --- which kwargs reach the trunk ----------------------------------------------------------------


def test_every_modality_has_exactly_one_forward_kwarg():
    """A modality added to the registry without a kwarg would build embeddings nothing ever feeds."""
    assert len(CAMERA_CONTROL_LATENT_KWARGS) == len(CAMERA_CONTROL_MODALITIES)
    for modality, kwarg in zip(CAMERA_CONTROL_MODALITIES, CAMERA_CONTROL_LATENT_KWARGS, strict=True):
        assert modality in kwarg


def test_the_reference_prefix_is_not_a_trunk_kwarg():
    """`prepare_batch` keeps the prefix rows in the same dict as the control rows.

    So the dict cannot be forwarded wholesale, and the backbone rejects an unknown kwarg rather than
    ignoring it -- which turns a wholesale forward into an immediate TypeError rather than a branch
    that silently does nothing.
    """
    assert "condition_video_rows" not in CAMERA_CONTROL_LATENT_KWARGS


def test_the_staged_latent_is_unpadded_so_it_can_be_replicated():
    """The contract between the two halves: what condition encoding stages, replication can consume."""
    reference = _proxy_reference()

    assert reference.latents is not None
    assert reference.latents.shape[-2:] == PROXY_GRID
    assert np.prod(TARGET_GRID) % np.prod(PROXY_GRID) == 0
