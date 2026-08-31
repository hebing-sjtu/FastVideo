# SPDX-License-Identifier: Apache-2.0
"""Turn a requested camera trajectory into ControlNet rows for MiniMax-H3.

Runs after latent preparation and before denoising, because it needs two things the layout owns:
the target latent grid the ray field is sampled on, and the packed row index of every target video
token. It writes both the field and those indices into ``batch.extra``; the denoising stage forwards
them to the transformer, which is where the ControlNet picks them up.

Producing nothing is a supported outcome. A request without a trajectory leaves ``batch.extra``
untouched, the denoising stage passes no control kwargs, and the model runs as the plain Ref2VA
checkpoint does — which is also the branch that classifier-free guidance on the camera would use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.minimax_h3.camera import build_camera_latent
from fastvideo.pipelines.basic.minimax_h3.packing import MiniMaxH3PackedLayout, patchify_video_latents
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import get_compute_dtype

logger = init_logger(__name__)

# Request key: an `.npz` path, or a mapping with `extrinsics`, `intrinsics` and optional
# `pixel_size`. Both forms are accepted because a trajectory is usually a file on disk during
# evaluation and an in-memory array when a caller drives the pipeline from Python.
MINIMAX_H3_CAMERA_TRAJECTORY_KEY = "minimax_h3_camera"
# Output keys, named for the transformer kwargs they become.
MINIMAX_H3_CAMERA_LATENT_KEY = "camera_latent"
MINIMAX_H3_CAMERA_ROWS_KEY = "camera_row_indices"


def _first(value: list[int] | int | None) -> int | None:
    if isinstance(value, list):
        return int(value[0]) if value else None
    return None if value is None else int(value)


def load_camera_trajectory(source: Any) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int] | None]:
    """Read extrinsics, intrinsics and the resolution the intrinsics were measured at.

    Kept in step with ``scripts/h3_proxy/prepare_data/encode_proxy_samples.py`` by hand rather than
    shared: training reads the trajectory once into a cache and inference reads it per request, and
    a shared reader would have to serve both without either owning it.
    """
    if isinstance(source, str | Path):
        with np.load(str(source)) as payload:
            extrinsics = torch.from_numpy(np.asarray(payload["extrinsics"], dtype=np.float32))
            intrinsics = torch.from_numpy(np.asarray(payload["intrinsics"], dtype=np.float32))
            # `.files` rather than `in payload`: NpzFile only became a Mapping in recent NumPy.
            raw_size = payload["pixel_size"] if "pixel_size" in payload.files else None
    elif isinstance(source, dict):
        extrinsics = torch.as_tensor(np.asarray(source["extrinsics"], dtype=np.float32))
        intrinsics = torch.as_tensor(np.asarray(source["intrinsics"], dtype=np.float32))
        raw_size = source.get("pixel_size")
    else:
        raise TypeError(f"{MINIMAX_H3_CAMERA_TRAJECTORY_KEY} must be an .npz path or a mapping with 'extrinsics' "
                        f"and 'intrinsics', got {type(source).__name__}.")

    if extrinsics.shape[0] != intrinsics.shape[0]:
        raise ValueError(f"The trajectory has {extrinsics.shape[0]} extrinsics and {intrinsics.shape[0]} "
                         "intrinsics; they must cover the same frames.")
    pixel_size = None if raw_size is None else (int(np.asarray(raw_size).reshape(-1)[0]),
                                                int(np.asarray(raw_size).reshape(-1)[1]))
    return extrinsics, intrinsics, pixel_size


class MiniMaxH3CameraConditioningStage(PipelineStage):
    """Build the camera ControlNet's patchified rows for one request."""

    def __init__(self, transformer: Any) -> None:
        super().__init__()
        self.transformer = transformer

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        # Nothing is required: a request may legitimately carry no trajectory.
        return VerificationResult()

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult()

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        source = batch.extra.get(MINIMAX_H3_CAMERA_TRAJECTORY_KEY)
        if source is None:
            return batch

        controlnet = getattr(self.transformer, "camera_controlnet", None)
        if controlnet is None:
            raise ValueError(f"A camera trajectory was requested but {type(self.transformer).__name__} has no camera "
                             "ControlNet. Point the run at a checkpoint fine-tuned with "
                             "MiniMaxH3ProxyModel, or drop the trajectory from the request.")

        layout = batch.extra.get(MINIMAX_H3_LAYOUT_KEY)
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise ValueError("Camera conditioning must run after MiniMax-H3 latent preparation.")

        device = get_local_torch_device()
        extrinsics, intrinsics, pixel_size = load_camera_trajectory(source)
        if pixel_size is None:
            # `height`/`width` are per-request scalars here but typed as optional lists on the batch.
            height, width = _first(batch.height), _first(batch.width)
            if height is None or width is None:
                raise ValueError("The trajectory carries no `pixel_size` and the request has no height/width to "
                                 "fall back on; intrinsics cannot be rescaled to the latent grid.")
            pixel_size = (height, width)

        camera_latent = build_camera_latent(
            extrinsics.to(device=device, dtype=torch.float32),
            intrinsics.to(device=device, dtype=torch.float32),
            latent_height=int(layout.latent_height),
            latent_width=int(layout.latent_width),
            pixel_size=pixel_size,
            num_latent_frames=int(layout.num_video_latent_frames),
        )
        rows = patchify_video_latents(camera_latent.to(get_compute_dtype()), self.transformer.patch_size)

        # The condition prefix belongs to the references; the ControlNet only drives what is being
        # generated, so the trailing target rows are the ones it is given.
        target_rows = layout.video_indices[layout.num_condition_video_rows:].to(device)
        if rows.shape[0] != target_rows.numel():
            raise ValueError(f"The trajectory produced {rows.shape[0]} control rows for {target_rows.numel()} target "
                             "video rows; the ray field and the target latent grid disagree.")

        batch.extra[MINIMAX_H3_CAMERA_LATENT_KEY] = rows[None]
        batch.extra[MINIMAX_H3_CAMERA_ROWS_KEY] = target_rows
        logger.info("MiniMax-H3 camera control: %d rows over a %dx%dx%d latent grid", rows.shape[0],
                    layout.num_video_latent_frames, layout.latent_height, layout.latent_width)
        return batch


__all__ = [
    "MINIMAX_H3_CAMERA_LATENT_KEY",
    "MINIMAX_H3_CAMERA_ROWS_KEY",
    "MINIMAX_H3_CAMERA_TRAJECTORY_KEY",
    "MiniMaxH3CameraConditioningStage",
    "load_camera_trajectory",
]
