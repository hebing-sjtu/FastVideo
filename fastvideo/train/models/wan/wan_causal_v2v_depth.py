# SPDX-License-Identifier: Apache-2.0
"""Causal Wan video-to-video + depth ControlNet plugin (AR and DMD stages).

Adds to the bidirectional plugin the two things streaming needs: the ControlNet
gets its own KV caches (its blocks are narrower than the backbone's, so it
cannot share them), and every control latent is sliced to the chunk currently
being denoised, because the causal backbone only ever sees one chunk at a time.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch

from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.wan.wan_causal import WanCausalModel
from fastvideo.train.models.wan.wan_v2v_depth import WanV2VDepthModel

logger = init_logger(__name__)


class WanCausalV2VDepthModel(WanV2VDepthModel, WanCausalModel):
    """Block-causal Wan video-to-video model with a depth ControlNet branch."""

    _transformer_cls_name: str = "CausalWanV2VDepthTransformer3DModel"

    def __init__(self, **kwargs: Any) -> None:
        # (cache_tag, control kind) -> per-control-block cache dicts.
        self._control_caches: dict[tuple[str, str], list[dict[str, Any]]] = {}
        # Set for the duration of one streaming call so the shared
        # `_build_distill_input_kwargs` can find the control latents; the base
        # class's streaming entry point does not thread the batch through.
        self._control_window: tuple[TrainingBatch, int, int] | None = None
        self._pending_control_kwargs: dict[str, Any] = {}
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Cache lifecycle
    # ------------------------------------------------------------------

    def clear_caches(self, *, cache_tag: str = "pos") -> None:
        super().clear_caches(cache_tag=cache_tag)
        tag = str(cache_tag)
        for kind in ("control", "control_wide"):
            self._control_caches.pop((tag, kind), None)

    def _get_control_caches(
        self,
        *,
        cache_tag: str,
        kind: str,
        noisy_latents: torch.Tensor,
        frame_seq_length: int,
        local_attn_size: int,
        sliding_window_num_frames: int,
    ) -> list[dict[str, Any]] | None:
        controlnet = getattr(self.transformer, "depth_controlnet", None)
        if controlnet is None:
            return None
        if kind == "control_wide" and not self._enable_wide_fov:
            return None

        key = (str(cache_tag), kind)
        cached = self._control_caches.get(key)
        batch_size = int(noisy_latents.shape[0])
        dtype = noisy_latents.dtype
        device = noisy_latents.device
        if cached is not None:
            probe = cached[0]["k"]
            if (probe.shape[0] == batch_size and probe.dtype == dtype and probe.device == device):
                return cached

        if local_attn_size != -1:
            cache_size = int(local_attn_size) * int(frame_seq_length)
        else:
            cache_size = int(frame_seq_length) * int(sliding_window_num_frames)
        if self._should_use_checkpoint_safe_kv_cache():
            tc = self.training_config
            total_frames = int(getattr(tc.data, "num_latent_t", 0)) if tc is not None else 0
            if total_frames <= 0:
                raise ValueError("training.data.num_latent_t must be set for a checkpoint-safe control KV cache")
            cache_size = max(cache_size, int(frame_seq_length) * total_frames)

        num_heads = int(controlnet.control_num_heads)
        head_dim = int(controlnet.control_head_dim)
        caches = [{
            "k": torch.zeros([batch_size, cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "global_end_index": torch.zeros((), dtype=torch.long, device=device),
            "local_end_index": torch.zeros((), dtype=torch.long, device=device),
        } for _ in range(int(controlnet.num_control_blocks))]
        self._control_caches[key] = caches
        return caches

    # ------------------------------------------------------------------
    # Streaming forward
    # ------------------------------------------------------------------

    def predict_noise_streaming(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor | None:
        self._control_window = (batch, int(cur_start_frame), int(noisy_latents.shape[1]))
        try:
            return super().predict_noise_streaming(
                noisy_latents,
                timestep,
                batch,
                conditional=conditional,
                cache_tag=cache_tag,
                store_kv=store_kv,
                cur_start_frame=cur_start_frame,
                cfg_uncond=cfg_uncond,
                attn_kind=attn_kind,
            )
        finally:
            self._control_window = None

    def _get_or_init_streaming_caches(self, *, cache_tag: str, transformer: torch.nn.Module,
                                      noisy_latents: torch.Tensor) -> Any:
        caches = super()._get_or_init_streaming_caches(
            cache_tag=cache_tag,
            transformer=transformer,
            noisy_latents=noisy_latents,
        )
        self._pending_control_kwargs = {
            "control_kv_cache":
            self._get_control_caches(
                cache_tag=cache_tag,
                kind="control",
                noisy_latents=noisy_latents,
                frame_seq_length=caches.frame_seq_length,
                local_attn_size=caches.local_attn_size,
                sliding_window_num_frames=caches.sliding_window_num_frames,
            ),
            "control_wide_kv_cache":
            self._get_control_caches(
                cache_tag=cache_tag,
                kind="control_wide",
                noisy_latents=noisy_latents,
                frame_seq_length=caches.frame_seq_length,
                local_attn_size=caches.local_attn_size,
                sliding_window_num_frames=caches.sliding_window_num_frames,
            ),
        }
        return caches

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        batch: TrainingBatch | None = None,
    ) -> dict[str, Any]:
        window = self._control_window
        if batch is None and window is not None:
            batch, start_frame, num_frames = window
            kwargs = super()._build_distill_input_kwargs(
                noise_input,
                timestep,
                text_dict,
                clean_x=clean_x,
                aug_t=aug_t,
                batch=self._sliced_batch_view(batch, start_frame, num_frames),
            )
            kwargs["start_frame"] = start_frame
            kwargs.update(self._pending_control_kwargs)
            return kwargs
        return super()._build_distill_input_kwargs(
            noise_input,
            timestep,
            text_dict,
            clean_x=clean_x,
            aug_t=aug_t,
            batch=batch,
        )

    @staticmethod
    def _sliced_batch_view(batch: TrainingBatch, start_frame: int, num_frames: int) -> TrainingBatch:
        """Shallow view of ``batch`` whose control latents cover one chunk only.

        Control latents are cached full-length, but a causal step patch-embeds
        only the frames it is denoising; feeding the full clip would misalign
        every control token against the backbone's.
        """
        stop = start_frame + num_frames
        view = copy.copy(batch)
        for field in ("control_latent", "depth_latent", "depth_wide_latent"):
            tensor = getattr(batch, field, None)
            if tensor is None:
                continue
            if tensor.shape[2] < stop:
                raise ValueError(f"{field} has {tensor.shape[2]} latent frames but the current chunk needs frames "
                                 f"[{start_frame}, {stop}); the cached control clip is shorter than the video.")
            setattr(view, field, tensor[:, :, start_frame:stop])
        return view
