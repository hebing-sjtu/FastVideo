# SPDX-License-Identifier: Apache-2.0
"""Conditioning stage for the Wan video-to-video + depth ControlNet pipeline.

This stage exists to make evaluation reproduce training exactly. The training
plugin fills Wan I2V's extra input channels with ``[noise | mask | source]``,
where the mask is all ones because in video-to-video every frame is
conditioned. The generic ``VideoVAEEncodingStage`` cannot stand in: it requires
a pre-populated ``batch.video_latent`` that no validation record can supply, and
``DenoisingStage`` would then build the Wan-Fun-Control layout
``[noise | source | zeros]``, which is 48 channels against a 36-channel Fun-InP
input convolution.

The trick that keeps this short: publish the conditioning as
``batch.image_latent`` holding ``[mask | source]``. ``DenoisingStage`` already
concatenates that onto the noise for image-to-video, which yields the exact
tensor the training plugin assembles. Depth rides alongside as an explicit
transformer kwarg instead of a channel.

The crop/resize/depth-encoding below intentionally mirrors
``scripts/v2v_depth/prepare_data/encode_v2v_depth_samples.py``. Keep the two in
sync: a checkpoint trained on a cache built with one crop, then evaluated under
another, sees a control signal shifted against the frame. Nothing raises -- the
metrics just quietly get worse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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

# Defaults match the encode script so a validation record that omits them lands
# on the same depth range the cache was built with.
DEFAULT_DEPTH_NEAR = 0.1
DEFAULT_DEPTH_FAR = 500.0
DEFAULT_DEPTH_ENCODING = "disparity"


def _crop_box(width: int, height: int, crop: dict[str, float]) -> tuple[int, int, int, int]:
    """Return a PIL-style ``(left, top, right, bottom)`` pixel box."""
    left = int(round(crop["crop_left"] * width))
    top = int(round(crop["crop_top"] * height))
    right = width - int(round(crop["crop_right"] * width))
    bottom = height - int(round(crop["crop_bottom"] * height))
    if right - left < 2 or bottom - top < 2:
        raise ValueError(f"crop emptied the frame: box=({left}, {top}, {right}, {bottom}) from {width}x{height}")
    return left, top, right, bottom


def _read_frames(path: str, num_frames: int) -> list[Any]:
    """Read a clip as PIL images, from a video file or a frame directory."""
    from PIL import Image

    directory = Path(path)
    if directory.is_dir():
        frame_paths = sorted(p for p in directory.iterdir()
                             if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".exr", ".tiff"})
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {path}")
        frames: list[Any] = [Image.open(p) for p in frame_paths]
    else:
        from fastvideo.models.vision_utils import load_video

        frames = load_video(str(path))
    if len(frames) < num_frames:
        raise ValueError(f"{path} has {len(frames)} frames, need {num_frames}")
    return frames[:num_frames]


def _rgb_clip(frames: list[Any], num_frames: int, height: int, width: int, crop: dict[str, float]) -> torch.Tensor:
    """Build an RGB clip ``[1, 3, T, H, W]`` in [-1, 1].

    The crop runs before the resize so HUD bands are removed at the source
    aspect ratio rather than after squashing.
    """
    from PIL import Image

    if len(frames) < num_frames:
        raise ValueError(f"got {len(frames)} frames, need {num_frames}")
    tensors = []
    box = None
    for frame in frames[:num_frames]:
        frame = frame.convert("RGB")
        if box is None:
            box = _crop_box(frame.size[0], frame.size[1], crop)
        frame = frame.crop(box).resize((width, height), Image.BICUBIC)
        array = np.asarray(frame, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors, dim=1).unsqueeze(0) * 2.0 - 1.0


def _depth_clip(
    frames: list[Any],
    num_frames: int,
    height: int,
    width: int,
    crop: dict[str, float],
    *,
    near: float,
    far: float,
    encoding: str,
) -> torch.Tensor:
    """Build a depth clip ``[1, 3, T, H, W]`` in [-1, 1].

    A 16-bit source carries millimetres and is remapped through the fixed
    near/far pair; an 8-bit source is assumed already normalized by the renderer
    and is only rescaled. Fixing the range rather than using per-clip min/max is
    what keeps the control signal comparable across clips.
    """
    from PIL import Image

    if len(frames) < num_frames:
        raise ValueError(f"got {len(frames)} depth frames, need {num_frames}")
    tensors = []
    box = None
    for frame in frames[:num_frames]:
        array = np.asarray(frame)
        if array.ndim == 3:
            array = array[..., 0]
        if box is None:
            left, top, right, bottom = _crop_box(array.shape[1], array.shape[0], crop)
            box = (top, bottom, left, right)
        top, bottom, left, right = box
        array = array[top:bottom, left:right].astype(np.float32)
        if array.max() > 1.5:
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
    return torch.stack(tensors, dim=1).unsqueeze(0) * 2.0 - 1.0


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
    def _crop(batch: ForwardBatch) -> dict[str, float]:
        """Read the crop the cache was built with off the validation record.

        The crop belongs to the dataset encoding, not to the model, so it
        travels with the data rather than with the pipeline config.
        """
        crop = {
            name: float(batch.extra.get(name, 0.0))
            for name in ("crop_top", "crop_bottom", "crop_left", "crop_right")
        }
        for name, value in crop.items():
            if not 0.0 <= value < 0.5:
                raise ValueError(f"{name} must be in [0, 0.5), got {value!r}")
        return crop

    def _source_frames(self, batch: ForwardBatch, num_frames: int) -> list[Any]:
        """``ValidationDataset`` decodes ``control_video_path`` for us; fall back to a path."""
        frames = batch.extra.get("control_video")
        if frames:
            return list(frames)
        path = batch.extra.get("control_video_path") or batch.video_path
        if not path:
            raise ValueError("video-to-video validation needs a source clip: set 'control_video_path' (or "
                             "'video_path') on the validation record.")
        return _read_frames(str(path), num_frames)

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
        height, width, num_frames = batch.height, batch.width, batch.num_frames
        crop = self._crop(batch)

        self.vae = self.vae.to(get_local_torch_device())
        try:
            source_latent = self._encode(
                _rgb_clip(self._source_frames(batch, num_frames), num_frames, height, width, crop),
                fastvideo_args,
            )

            depth_latents: dict[str, torch.Tensor] = {}
            for key, kwarg in (("depth", "depth_latent"), ("depth_wide", "depth_wide_latent")):
                path = batch.extra.get(f"{key}_video_path")
                if not path:
                    continue
                clip = _depth_clip(
                    _read_frames(str(path), num_frames),
                    num_frames,
                    height,
                    width,
                    crop,
                    near=float(batch.extra.get("depth_near", DEFAULT_DEPTH_NEAR)),
                    far=float(batch.extra.get("depth_far", DEFAULT_DEPTH_FAR)),
                    encoding=str(batch.extra.get("depth_encoding", DEFAULT_DEPTH_ENCODING)),
                )
                depth_latents[kwarg] = self._encode(clip, fastvideo_args)
        finally:
            self.vae.to("cpu")

        target_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        source_latent = source_latent.to(target_dtype)

        # The transformer's in_channels decides how wide the mask band is; Wan
        # I2V and Fun-InP ship 36 = 16 noise + 4 mask + 16 condition. This reads
        # the same attribute the training plugin does, so the two cannot
        # disagree.
        in_channels = int(self.transformer.in_channels)
        latent_channels = source_latent.shape[1]
        mask_channels = in_channels - 2 * latent_channels
        if mask_channels < 0:
            raise ValueError(f"transformer in_channels={in_channels} is too small for {latent_channels} noise + "
                             f"{latent_channels} condition channels. Video-to-video needs an I2V checkpoint, "
                             "not T2V.")

        parts = [source_latent]
        if mask_channels:
            parts.insert(
                0,
                torch.ones(
                    source_latent.shape[0],
                    mask_channels,
                    *source_latent.shape[2:],
                    device=source_latent.device,
                    dtype=source_latent.dtype,
                ))
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
