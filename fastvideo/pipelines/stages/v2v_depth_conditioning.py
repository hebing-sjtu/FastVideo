# SPDX-License-Identifier: Apache-2.0
"""Conditioning stage for the Wan video-to-video + depth ControlNet pipeline.

This stage exists to make evaluation reproduce training exactly. The training
plugin fills Wan I2V's extra input channels with ``[noise | mask | source]``,
where the mask is all ones because in video-to-video every frame is
conditioned. Reusing the generic ``VideoVAEEncodingStage`` would instead build
the Wan-Fun-Control layout ``[noise | source | zeros]``, so a checkpoint would
be evaluated against a channel arrangement it never saw.

The trick that keeps this short: publish the conditioning as
``batch.image_latent`` holding ``[mask | source]``. ``DenoisingStage`` already
concatenates that onto the noise for image-to-video, which yields the exact
tensor the training plugin assembles. Depth rides alongside as an explicit
transformer kwarg instead of a channel.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Defaults match ``encode_v2v_depth_samples.py`` so a validation record that
# omits them lands on the same depth range the cache was built with.
DEFAULT_DEPTH_NEAR = 0.1
DEFAULT_DEPTH_FAR = 500.0
DEFAULT_DEPTH_ENCODING = "disparity"


def _preprocess():
    """Import the shared clip preprocessing lazily.

    ``fastvideo.dataset`` pulls in torchvision, pyarrow and transformers at
    package import. This stage is reachable from ``stages/__init__``, which
    every inference path loads, so the cost is deferred to the one call that
    actually needs it.
    """
    from fastvideo.dataset import v2v_depth_preprocess

    return v2v_depth_preprocess


class WanV2VDepthConditioningStage(PipelineStage):
    """Encode the source clip (and optional depth) into Wan's conditioning slots."""

    def __init__(self, vae, transformer) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("image_latent", batch.image_latent, V.is_tensor)
        return result

    # ------------------------------------------------------------------
    # Pixel sourcing
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_kwargs(batch: ForwardBatch) -> dict[str, float]:
        """Read the crop the cache was built with off the validation record.

        The crop belongs to the dataset encoding, not to the model, so it
        travels with the data rather than with the pipeline config.
        """
        return {
            name: float(batch.extra.get(name, 0.0))
            for name in ("crop_top", "crop_bottom", "crop_left", "crop_right")
        }

    def _source_clip(
        self,
        batch: ForwardBatch,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build the source clip tensor from decoded frames or a path."""
        preprocess = _preprocess()
        crop = self._crop_kwargs(batch)
        frames = batch.extra.get("control_video")
        if frames:
            return preprocess.frames_to_rgb_clip(
                frames,
                num_frames=num_frames,
                height=height,
                width=width,
                **crop,
            )

        path = batch.extra.get("control_video_path") or batch.video_path
        if not path:
            raise ValueError("video-to-video validation needs a source clip: set 'control_video_path' (or "
                             "'video_path') on the validation record.")
        return preprocess.load_rgb_clip(
            path,
            num_frames=num_frames,
            height=height,
            width=width,
            **crop,
        )

    def _depth_clip(
        self,
        batch: ForwardBatch,
        key: str,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        """Build a depth clip for ``key`` (``depth`` or ``depth_wide``), if present."""
        preprocess = _preprocess()
        crop = self._crop_kwargs(batch)
        depth_kwargs: dict[str, Any] = {
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "near": float(batch.extra.get("depth_near", DEFAULT_DEPTH_NEAR)),
            "far": float(batch.extra.get("depth_far", DEFAULT_DEPTH_FAR)),
            "encoding": str(batch.extra.get("depth_encoding", DEFAULT_DEPTH_ENCODING)),
            **crop,
        }

        frames = batch.extra.get(f"{key}_video")
        if frames:
            return preprocess.frames_to_depth_clip(frames, **depth_kwargs)
        path = batch.extra.get(f"{key}_video_path")
        if path:
            return preprocess.load_depth_clip(path, **depth_kwargs)
        return None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode(self, clip: torch.Tensor, fastvideo_args: FastVideoArgs) -> torch.Tensor:
        """VAE-encode ``[1, 3, T, H, W]`` pixels into normalized latents.

        The distribution mode is used rather than a sample because the cached
        training latents were built the same way; sampling here would add noise
        to the control signal that training never saw.
        """
        device = get_local_torch_device()
        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        autocast_enabled = (vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        clip = clip.to(device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=vae_dtype, enabled=autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not autocast_enabled:
                clip = clip.to(vae_dtype)
            encoder_output = self.vae.encode(clip)

        latent = encoder_output.mode() if hasattr(encoder_output, "mode") else encoder_output
        latent = latent.to(torch.float32)

        # Same normalization the training loader applies via
        # ``normalize_dit_input``: (x - latents_mean) / latents_std.
        shift = getattr(self.vae, "shift_factor", None)
        if shift is not None:
            shift = shift.to(latent.device, latent.dtype) if isinstance(shift, torch.Tensor) else shift
            latent = latent - shift
        scale = getattr(self.vae, "scaling_factor", None)
        if scale is not None:
            scale = scale.to(latent.device, latent.dtype) if isinstance(scale, torch.Tensor) else scale
            latent = latent * scale
        return latent

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert isinstance(batch.height, int) and isinstance(batch.width, int)
        assert isinstance(batch.num_frames, int)
        height, width, num_frames = (batch.height, batch.width, batch.num_frames)

        self.vae = self.vae.to(get_local_torch_device())
        try:
            source_latent = self._encode(
                self._source_clip(batch, num_frames=num_frames, height=height, width=width),
                fastvideo_args,
            )

            depth_latents: dict[str, torch.Tensor] = {}
            for key, kwarg in (("depth", "depth_latent"), ("depth_wide", "depth_wide_latent")):
                clip = self._depth_clip(batch, key, num_frames=num_frames, height=height, width=width)
                if clip is not None:
                    depth_latents[kwarg] = self._encode(clip, fastvideo_args)
        finally:
            self.vae.to("cpu")

        target_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        source_latent = source_latent.to(target_dtype)

        # The transformer's in_channels decides how wide the mask band is; Wan
        # I2V ships 36 = 16 noise + 4 mask + 16 condition. This reads the same
        # attribute the training plugin does, so the two cannot disagree.
        in_channels = int(self.transformer.in_channels)
        latent_channels = source_latent.shape[1]
        mask_channels = in_channels - 2 * latent_channels
        if mask_channels < 0:
            raise ValueError(f"transformer in_channels={in_channels} is too small for {latent_channels} noise + "
                             f"{latent_channels} condition channels. Video-to-video needs an I2V checkpoint, "
                             "not T2V.")

        parts = [source_latent]
        if mask_channels:
            mask = torch.ones(
                source_latent.shape[0],
                mask_channels,
                *source_latent.shape[2:],
                device=source_latent.device,
                dtype=source_latent.dtype,
            )
            parts.insert(0, mask)
        batch.image_latent = torch.cat(parts, dim=1)

        for kwarg, latent in depth_latents.items():
            batch.extra[kwarg] = latent.to(target_dtype)

        logger.info(
            "V2V conditioning: image_latent=%s depth=%s",
            tuple(batch.image_latent.shape),
            {
                k: tuple(v.shape)
                for k, v in depth_latents.items()
            } or "none",
        )
        return batch
