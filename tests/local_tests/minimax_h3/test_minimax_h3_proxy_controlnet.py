# SPDX-License-Identifier: Apache-2.0
"""Lifting the proxy onto the target latent grid for the control trunk.

The trunk is the one route by which a proxy can act per-token, and everything that can go wrong on
the way in goes wrong quietly. An interpolated DUV frame looks like a plausible image while carrying
depths that were never rendered and class colours no segmenter predicts. A cache encoded at a canvas
the proxy grid does not divide would train on that. A trunk pointed at the wrong reference would
train on the anchor. A sampler that skips the lift would render from a trunk that saw nothing, and
report a checkpoint as dead.

So these cover the guards rather than the arithmetic of the residual: exact block replication, the
canvas contract, which reference is read, and when the lift is skipped entirely.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "h3_proxy" / "prepare_data"


def _encoder_module() -> types.ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("encode_proxy_samples", SCRIPTS / "encode_proxy_samples.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the encoder's lift, which writes `depth_latent` into the cache ------------------------------


def test_replication_onto_the_canvas_repeats_codes_and_invents_none():
    """The whole point of replicating rather than interpolating: the code set is unchanged."""
    encoder = _encoder_module()
    # Three distinct codes with nothing between them, so any averaging shows up as a new value.
    pixels = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 1, 3).expand(1, 3, 2, 1, 3).contiguous()

    lifted = encoder.replicate_onto_canvas(pixels, height=4, width=12)

    assert lifted.shape == (1, 3, 2, 4, 12)
    assert torch.equal(torch.unique(lifted), torch.unique(pixels))
    # Each source pixel occupies a 4x4 block, and the blocks tile the canvas in source order.
    assert torch.equal(lifted[0, :, 0, 0, :4], pixels[0, :, 0, 0, 0].unsqueeze(-1).expand(3, 4))
    assert torch.equal(lifted[0, :, 0, 3, 8:], pixels[0, :, 0, 0, 2].unsqueeze(-1).expand(3, 4))


def test_the_released_geometries_divide_and_a_stale_one_does_not():
    """704x1280 is the canvas the earlier GTA caches used; it cannot carry a replicated proxy."""
    encoder = _encoder_module()
    pixels = torch.zeros(1, 3, 2, 192, 336)

    assert encoder.replicate_onto_canvas(pixels, height=768, width=1344).shape[-2:] == (768, 1344)
    with pytest.raises(ValueError, match="integer multiple"):
        encoder.replicate_onto_canvas(pixels, height=704, width=1280)


# --- the sampler's lift, which has to agree with the cache's --------------------------------------


def _stage(modalities: tuple[str, ...] | None) -> MiniMaxH3LatentPreparationStage:
    """A latent-preparation stage with only what the lift reads.

    Constructed without ``__init__`` on purpose: the real one builds a VAE and a text encoder, and
    every path under test refuses before the VAE would be touched.
    """
    stage = object.__new__(MiniMaxH3LatentPreparationStage)
    controlnet = None if modalities is None else types.SimpleNamespace(enabled_modalities=modalities)
    stage.transformer = types.SimpleNamespace(camera_controlnet=controlnet, patch_size=(1, 2, 2))
    return stage


def _batch(height: int = 768, width: int = 1344) -> ForwardBatch:
    return ForwardBatch(data_type="video", num_frames=124, height=height, width=width)


def _proxy(height: int = 192, width: int = 336) -> MiniMaxH3PreparedReference:
    reference = MiniMaxH3PreparedReference(media_type="video", num_latent_frames=2, latent_height=12, latent_width=21)
    reference.frames = np.zeros((124, height, width, 3), dtype=np.uint8)
    return reference


@pytest.mark.parametrize("modalities", [None, ("camera", )])
def test_the_lift_is_skipped_when_no_trunk_reads_the_proxy(modalities):
    """A plain backbone, or a trunk on the Plücker field alone, must not pay for a second encode."""
    stage = _stage(modalities)

    assert stage._encode_control_depth_rows([_proxy()], _batch(), torch.device("cpu")) is None


def test_a_canvas_the_proxy_grid_does_not_divide_is_refused():
    stage = _stage(("depth", ))

    with pytest.raises(ValueError, match="integer multiple"):
        stage._encode_control_depth_rows([_proxy()], _batch(height=704, width=1280), torch.device("cpu"))


def test_the_trunk_reads_the_proxy_and_not_the_anchor():
    """The anchor is an image reference, so it is not a candidate; a second video makes it ambiguous."""
    stage = _stage(("depth", ))
    anchor = MiniMaxH3PreparedReference(media_type="image", num_latent_frames=1, latent_height=128, latent_width=224)
    device = torch.device("cpu")

    # One video alongside the anchor is unambiguous, and gets as far as the canvas check.
    with pytest.raises(ValueError, match="integer multiple"):
        stage._encode_control_depth_rows([anchor, _proxy()], _batch(height=704, width=1280), device)

    for references in ([anchor], [anchor, _proxy(), _proxy()]):
        with pytest.raises(ValueError, match="exactly one video reference"):
            stage._encode_control_depth_rows(references, _batch(), device)
