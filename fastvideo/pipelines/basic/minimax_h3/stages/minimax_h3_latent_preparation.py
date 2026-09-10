# SPDX-License-Identifier: Apache-2.0
"""Condition encoding, packed layout, and target noise for MiniMax H3."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3PackedLayout,
    audio_latent_num_frames,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    keyframe_condition_noise,
    pad_video_latents_to_patch,
    patchify_video_latents,
    target_rows_per_latent_frame,
)
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MiniMaxH3PreparedReference,
    trim_reference_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import (
    MINIMAX_H3_FIXED_FIRST_FRAME_KEY,
    MINIMAX_H3_KEYFRAME_ANCHORS_KEY,
    MINIMAX_H3_KEYFRAMES_KEY,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

MINIMAX_H3_LAYOUT_KEY = "minimax_h3_layout"
# How many leading rows of the target video are a given rather than something to denoise. Read by
# the denoising stage, which holds them at the reference prefix's noise amount and leaves them out
# of every scheduler step.
MINIMAX_H3_NUM_FIXED_VIDEO_ROWS_KEY = "minimax_h3_num_fixed_video_rows"


def _video_geometry(batch: ForwardBatch) -> tuple[int, int, int, int]:
    shape = batch.raw_latent_shape
    if shape is None or len(shape) != 5:
        raise ValueError("MiniMax-H3 input preparation must set a five-dimensional raw latent shape.")
    _, channels, num_frames, height, width = shape
    if min(channels, num_frames, height, width) <= 0:
        raise ValueError(f"MiniMax-H3 raw latent geometry must be positive, got {shape}.")
    return channels, num_frames, height, width


def _sample_visual_posterior(posterior: Any) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
    return posterior.sample(generator=generator)


class MiniMaxH3LatentPreparationStage(PipelineStage):
    """Encode fixed conditions, build the row layout, then draw target noise."""

    performance_component_metric = "vae_encode_time_s"

    def __init__(
        self,
        transformer: Any,
        vae: Any,
        audio_vae: Any,
        scheduler: Any,
        *,
        ref2va: bool = False,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.audio_vae = audio_vae
        self.scheduler = scheduler
        self.ref2va = ref2va

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(3))
        result.add_check("text_token_tags", batch.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY), V.with_dims(1))
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.not_none)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        if self.ref2va:
            result.add_check("references", batch.references, V.list_not_empty)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("layout", batch.extra.get(MINIMAX_H3_LAYOUT_KEY), V.not_none)
        result.add_check("latents", batch.latents, V.with_dims(2))
        result.add_check("audio_latents", batch.audio_latents, V.with_dims(2))
        return result

    def _encode_keyframe_latents(self, image, device: torch.device) -> torch.Tensor:
        """Encode one keyframe image to normalized latents (fp16 round-trip
        preserved for parity with the seeded references)."""
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None]
        pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
        posterior = self.vae.encode_keyframe(self.vae.normalize_pixels(pixels)).latent_dist
        return self.vae.normalize_latents(_sample_visual_posterior(posterior).to(torch.float16).float()).cpu()

    def _encode_fixed_first_frame(self, image, device: torch.device) -> torch.Tensor:
        """Encode the anchor into the target's own latent space, keeping one latent frame.

        ``encode``, not ``encode_keyframe``: this lands in a *target* row, and target latents come
        from the video encode path. That path splits its input into fixed-length clips with no
        cross-clip state, so the first latent of a still image and the first latent of the real clip
        describe the same moment through the same weights -- which is what makes a standalone anchor
        interchangeable with the clip's own first latent frame.
        """
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None]
        pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
        posterior = self.vae.encode(self.vae.normalize_pixels(pixels)).latent_dist
        latents = self.vae.normalize_latents(_sample_visual_posterior(posterior).to(torch.float16).float())
        if latents.ndim != 5 or latents.shape[2] < 1:
            raise ValueError(f"Encoding the fixed first frame produced no latent frame: {tuple(latents.shape)}.")
        return latents[:, :, :1].cpu()

    def _encode_visual_rows(
        self,
        references: list[MiniMaxH3PreparedReference],
        device: torch.device,
    ) -> list[torch.Tensor]:
        patch_size = self.transformer.patch_size
        rows: list[torch.Tensor] = []
        for reference in references:
            if reference.media_type == "audio":
                continue
            if reference.media_type == "image":
                if reference.image is None:
                    raise ValueError("MiniMax-H3 reference image pixels are missing.")
                latents = self._encode_keyframe_latents(reference.image, device)
            else:
                if reference.frames is None:
                    raise ValueError("MiniMax-H3 reference video frames are missing.")
                frames = reference.frames[:trim_reference_num_frames(reference.frames.shape[0])]
                pixels = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None]
                pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
                posterior = self.vae.encode(self.vae.normalize_pixels(pixels)).latent_dist
                latents = self.vae.normalize_latents(_sample_visual_posterior(posterior).to(
                    torch.float16).float()).cpu()
            # Same pad the training cache applies: a 336x192 proxy is 21 x 12 after the VAE, and
            # 21 is not a multiple of the 2x2 patch. Geometry on the reference has to be the padded
            # size so the packed layout and the condition noise agree with the rows.
            latents = pad_video_latents_to_patch(latents, patch_size)
            reference.num_latent_frames = int(latents.shape[2])
            reference.latent_height = int(latents.shape[3])
            reference.latent_width = int(latents.shape[4])
            rows.append(patchify_video_latents(latents, patch_size))
        return rows

    def _encode_audio_rows(
        self,
        references: list[MiniMaxH3PreparedReference],
        device: torch.device,
    ) -> list[torch.Tensor]:
        rows: list[torch.Tensor] = []
        for reference in references:
            if not reference.has_audio:
                continue
            if reference.waveform is None:
                raise ValueError("MiniMax-H3 reference waveform is missing.")
            posterior = self.audio_vae.encode(reference.waveform.to(device)[:, None]).latent_dist
            latents = self.audio_vae.normalize_latents(posterior.mode().float()).cpu().transpose(1, 2)
            reference.num_audio_latents = int(latents.shape[1])
            rows.append(latents.reshape(-1, self.audio_vae.latent_channels))
        return rows

    def _encode_fl2va_conditions(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        keyframes = batch.extra.get(MINIMAX_H3_KEYFRAMES_KEY, [])
        if not isinstance(keyframes, list):
            raise TypeError("MiniMax-H3 keyframes must be a list.")
        if not keyframes:
            return None, None

        vae_device = get_local_torch_device()
        self.vae.to(vae_device)
        try:
            clean_rows: list[torch.Tensor] = []
            for image in keyframes:
                clean_rows.append(
                    patchify_video_latents(self._encode_keyframe_latents(image, vae_device),
                                           self.transformer.patch_size))
        finally:
            if fastvideo_args.vae_cpu_offload:
                self.vae.to("cpu")

        _, _, latent_height, latent_width = _video_geometry(batch)
        shapes = ((1, latent_height, latent_width), ) * len(keyframes)
        noise = keyframe_condition_noise(
            shapes,
            self.transformer.patch_size,
            self.vae.latent_channels,
            generator=batch.generator,
            device=device,
        )
        return (
            self.scheduler.scale_noise(
                torch.cat(clean_rows).to(device),
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                noise,
            ),
            None,
        )

    def _encode_ref2va_conditions(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        references = list(batch.references or [])
        if not references or not all(isinstance(item, MiniMaxH3PreparedReference) for item in references):
            raise TypeError("MiniMax-H3 Ref2VA latent preparation requires prepared references.")

        fixed_frame = batch.extra.get(MINIMAX_H3_FIXED_FIRST_FRAME_KEY)
        vae_device = get_local_torch_device()
        self.vae.to(vae_device)
        try:
            video_rows = self._encode_visual_rows(references, vae_device)
            fixed_latents = None if fixed_frame is None else self._encode_fixed_first_frame(fixed_frame, vae_device)
        finally:
            if fastvideo_args.vae_cpu_offload:
                self.vae.to("cpu")

        audio_rows: list[torch.Tensor] = []
        if any(reference.has_audio for reference in references):
            audio_device = get_local_torch_device()
            self.audio_vae.to(audio_device)
            try:
                audio_rows = self._encode_audio_rows(references, audio_device)
            finally:
                if fastvideo_args.vae_cpu_offload:
                    self.audio_vae.to("cpu")

        if not video_rows:
            raise ValueError("MiniMax-H3 Ref2VA requires at least one visual reference.")
        shapes = tuple((reference.num_latent_frames, reference.latent_height, reference.latent_width)
                       for reference in references if reference.media_type != "audio")
        # The locked frame's shape goes last so that turning the lock on does not shift the noise
        # the reference prefix draws, and a run with and without it stays comparable.
        if fixed_latents is not None:
            shapes += (tuple(int(size) for size in fixed_latents.shape[2:]), )
        noise = keyframe_condition_noise(
            shapes,
            self.transformer.patch_size,
            self.vae.latent_channels,
            generator=batch.generator,
            device=device,
        )
        reference_rows = torch.cat(video_rows)
        num_reference_rows = reference_rows.shape[0]
        # Held at the same near-clean amount as the reference prefix, and by the same draw: a locked
        # frame is a shown row, so training and sampling both have to present it the way the
        # references are presented rather than perfectly clean.
        augmented = self.scheduler.scale_noise(
            torch.cat([reference_rows] + ([patchify_video_latents(fixed_latents, self.transformer.patch_size)]
                                          if fixed_latents is not None else [])).to(device),
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            noise,
        )
        video_conditions = augmented[:num_reference_rows]
        fixed_video_rows = augmented[num_reference_rows:] if fixed_latents is not None else None
        audio_conditions = torch.cat(audio_rows).to(device) if audio_rows else None

        for reference in references:
            reference.image = None
            reference.frames = None
            reference.waveform = None
        return video_conditions, audio_conditions, fixed_video_rows

    def _build_layout(self, batch: ForwardBatch) -> MiniMaxH3PackedLayout:
        text_token_tags = batch.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY)
        if not isinstance(text_token_tags, torch.Tensor):
            raise ValueError("MiniMax-H3 conditioning must produce text token tags.")
        _, num_frames, height, width = _video_geometry(batch)
        if not isinstance(batch.num_frames, int):
            raise TypeError("MiniMax-H3 num_frames must be an integer.")
        num_audio_latents = audio_latent_num_frames(batch.num_frames)
        if self.ref2va:
            references = list(batch.references or [])
            return build_ref2va_packed_sequence(
                text_token_tags,
                references,
                num_frames,
                height,
                width,
                num_audio_latents,
                self.transformer.patch_size,
            )
        anchors = batch.extra.get(MINIMAX_H3_KEYFRAME_ANCHORS_KEY, ())
        if not isinstance(anchors, tuple):
            raise TypeError("MiniMax-H3 keyframe anchors must be a tuple.")
        return build_packed_sequence(
            text_token_tags,
            num_frames,
            height,
            width,
            num_audio_latents,
            self.transformer.patch_size,
            anchors,
        )

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        video_noise = batch.latents
        audio_noise = batch.audio_latents
        device = get_local_torch_device()
        fixed_video_rows = None
        if self.ref2va:
            condition_video, condition_audio, fixed_video_rows = self._encode_ref2va_conditions(
                batch, fastvideo_args, device)
        else:
            condition_video, condition_audio = self._encode_fl2va_conditions(batch, fastvideo_args, device)

        layout = self._build_layout(batch)
        video_channels, num_frames, height, width = _video_geometry(batch)
        expected_video_shape = (1, video_channels, num_frames, height, width)
        if video_noise is None:
            video_noise = randn_tensor(
                expected_video_shape,
                generator=batch.generator,
                device=device,
                dtype=torch.float32,
            )
        elif tuple(video_noise.shape) != expected_video_shape:
            raise ValueError(f"MiniMax-H3 injected video latents must have shape {expected_video_shape}, "
                             f"got {tuple(video_noise.shape)}.")
        video_rows = patchify_video_latents(video_noise.to(device=device, dtype=torch.float32),
                                            self.transformer.patch_size)

        num_audio_latents = layout.num_audio_latents
        expected_audio_shape = (MINIMAX_H3_AUDIO_CHANNELS, self.audio_vae.latent_channels, num_audio_latents)
        if audio_noise is None:
            audio_rows = randn_tensor(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, self.audio_vae.latent_channels),
                generator=batch.generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            if tuple(audio_noise.shape) != expected_audio_shape:
                raise ValueError(f"MiniMax-H3 injected audio latents must have shape {expected_audio_shape}, "
                                 f"got {tuple(audio_noise.shape)}.")
            audio_rows = audio_noise.to(device=device,
                                        dtype=torch.float32).permute(0, 2,
                                                                     1).reshape(-1, self.audio_vae.latent_channels)

        if condition_video is not None:
            video_rows = torch.cat((condition_video.to(device), video_rows))
        if condition_audio is not None:
            audio_rows = torch.cat((condition_audio.to(device), audio_rows))
        if video_rows.shape[0] != layout.video_indices.numel():
            raise ValueError("MiniMax-H3 packed video row count does not match its layout.")
        if audio_rows.shape[0] != layout.audio_indices.numel():
            raise ValueError("MiniMax-H3 packed audio row count does not match its layout.")

        num_fixed_video_rows = 0
        if fixed_video_rows is not None:
            # Overwrite the target's first latent frame in place, after the reference prefix has
            # been prepended, so the locked rows sit at the target's own rotary coordinates rather
            # than in the prefix. That position is the whole point: the prefix is read once for the
            # clip, while these rows say "at t=0 the scene looks exactly like this".
            num_fixed_video_rows = target_rows_per_latent_frame(layout, self.transformer.patch_size)
            if fixed_video_rows.shape[0] != num_fixed_video_rows:
                raise ValueError(f"The fixed first frame occupies {fixed_video_rows.shape[0]} rows but one target "
                                 f"latent frame is {num_fixed_video_rows} rows; the anchor was encoded on a canvas "
                                 "other than the target's.")
            start = layout.num_condition_video_rows
            video_rows[start:start + num_fixed_video_rows] = fixed_video_rows.to(video_rows)

        batch.latents = video_rows
        batch.audio_latents = audio_rows
        batch.extra[MINIMAX_H3_LAYOUT_KEY] = layout
        batch.extra[MINIMAX_H3_NUM_FIXED_VIDEO_ROWS_KEY] = num_fixed_video_rows
        batch.extra.pop(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY, None)
        batch.extra.pop(MINIMAX_H3_KEYFRAMES_KEY, None)
        batch.extra.pop(MINIMAX_H3_KEYFRAME_ANCHORS_KEY, None)
        batch.extra.pop(MINIMAX_H3_FIXED_FIRST_FRAME_KEY, None)
        batch.references = None
        return batch


__all__ = [
    "MINIMAX_H3_LAYOUT_KEY",
    "MINIMAX_H3_NUM_FIXED_VIDEO_ROWS_KEY",
    "MiniMaxH3LatentPreparationStage",
]
