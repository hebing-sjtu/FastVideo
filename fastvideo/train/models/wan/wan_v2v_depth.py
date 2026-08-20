# SPDX-License-Identifier: Apache-2.0
"""Wan video-to-video + depth ControlNet training plugin (bidirectional stage).

This is the BD stage of the BD -> AR -> DMD recipe. The Wan I2V backbone is
frozen by default and only the depth ControlNet trains, which is what keeps an
A14B run affordable.

The video-to-video conditioning needs no architecture change. Wan I2V's extra
input channels already carry a full-length clean conditioning latent
(``[noise 16 | mask 4 | condition 16]`` for the released A14B checkpoint); image
-to-video merely leaves all but the first frame zero and marks that with the
mask. Here the conditioning slot holds the whole source clip and the mask is all
ones, so every frame is declared given.
"""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.models.wan.v2v_depth_dataset import (
    build_v2v_depth_train_dataloader, )
from fastvideo.train.utils.checkpoint import (
    dcp_model_has_key_substring,
    load_prefixed_model_weights_from_dcp,
)
from fastvideo.train.utils.moduleloader import load_module_from_path
from fastvideo.training.training_utils import normalize_dit_input

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)

# Wide-branch tensors may legitimately be absent from a checkpoint that predates
# the branch; those params are warm-started instead of failing the load.
_WIDE_OPTIONAL_SUBSTRINGS = (".wca.", ".wide_embeddings.", ".wide_fusion_proj.")


class WanV2VDepthModel(WanModel):
    """Per-role Wan video-to-video model with a depth ControlNet branch."""

    _transformer_cls_name: str = "WanV2VDepthTransformer3DModel"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 5.0,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
        lora: LoraConfig | dict[str, Any] | None = None,
        attention_backend: AttentionBackendEnum | str | None = None,
        # --- control branch ---
        enable_controlnet: bool = True,
        enable_depth: bool = True,
        controlnet_dim: int = 1024,
        controlnet_ffn_dim: int = 4096,
        controlnet_num_heads: int = 16,
        controlnet_layer_stride: int = 1,
        freeze_backbone: bool = True,
        enable_wide_fov: bool = False,
        wide_fov_scale: float = 0.5,
        # --- video-to-video conditioning ---
        control_dropout: float = 0.0,
        # --- Wan 2.2 A14B is a two-expert MoE; train one expert at a time. ---
        expert: Literal["high", "low"] = "high",
        # --- stage handoff ---
        init_checkpoint: str | None = None,
        controlnet_checkpoint: str | None = None,
        **base_kwargs: Any,
    ) -> None:
        self._enable_controlnet = bool(enable_controlnet)
        self._enable_depth = bool(enable_depth)
        if self._enable_controlnet and not self._enable_depth:
            raise ValueError("enable_controlnet=true requires enable_depth=true; depth is the only control modality.")
        self._controlnet_dim = int(controlnet_dim)
        self._controlnet_ffn_dim = int(controlnet_ffn_dim)
        self._controlnet_num_heads = int(controlnet_num_heads)
        self._controlnet_layer_stride = int(controlnet_layer_stride)
        self._freeze_backbone = bool(freeze_backbone)
        self._enable_wide_fov = bool(enable_wide_fov)
        self._wide_fov_scale = float(wide_fov_scale)
        self._control_dropout = float(control_dropout)
        self._expert = str(expert).strip().lower()
        if self._expert not in {"high", "low"}:
            raise ValueError(f"expert must be 'high' or 'low', got {expert!r}")
        if self._enable_wide_fov and not self._enable_controlnet:
            raise ValueError("enable_wide_fov=true requires enable_controlnet=true.")
        if not 0.0 <= self._control_dropout < 1.0:
            raise ValueError(f"control_dropout must be in [0, 1), got {control_dropout!r}")

        self._stamp_control_config(training_config)

        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
            transformer_override_safetensor=transformer_override_safetensor,
            lora=lora,
            attention_backend=attention_backend,
            **base_kwargs,
        )

        if init_checkpoint:
            self._load_stage_checkpoint(init_checkpoint, controlnet_only=False)
        if controlnet_checkpoint:
            self._load_stage_checkpoint(controlnet_checkpoint, controlnet_only=True)

    # ------------------------------------------------------------------
    # Config plumbing
    # ------------------------------------------------------------------

    def _stamp_control_config(self, training_config: TrainingConfig) -> None:
        """Write ``control_*`` fields onto the DiT config before the model is built.

        The released Wan config has no counterpart for these and the DiT reads
        them with ``getattr`` defaults, so stamping is what turns the branch on
        without forking the upstream config dataclass. Each role re-stamps
        immediately before its own load, so a student with the wide branch and a
        teacher without it can share one ``training_config``.
        """
        pipeline_config = getattr(training_config, "pipeline_config", None)
        dit_config = getattr(pipeline_config, "dit_config", None)
        if dit_config is None:
            raise ValueError("training_config.pipeline_config.dit_config is required to configure the "
                             "depth ControlNet")
        dit_config.control_enable_controlnet = self._enable_controlnet
        dit_config.control_enable_depth = self._enable_depth
        dit_config.control_layer_stride = self._controlnet_layer_stride
        dit_config.control_dim = self._controlnet_dim
        dit_config.control_ffn_dim = self._controlnet_ffn_dim
        dit_config.control_num_heads = self._controlnet_num_heads
        dit_config.control_freeze_backbone = self._freeze_backbone
        dit_config.control_enable_wide = self._enable_wide_fov
        dit_config.control_wide_fov_scale = self._wide_fov_scale

    def _load_transformer(self, *, init_from: str, **kwargs: Any) -> torch.nn.Module:
        """Load one MoE expert of the Wan 2.2 checkpoint.

        A14B ships two experts. Training both at once doubles the memory for no
        benefit at this stage, so a run picks one; ``expert: low`` selects
        ``transformer_2``.
        """
        if self._expert == "low":
            training_config = kwargs["training_config"]
            transformer = load_module_from_path(
                model_path=init_from,
                module_type="transformer_2",
                training_config=training_config,
                disable_custom_init_weights=kwargs.get("disable_custom_init_weights", False),
                override_transformer_cls_name=self._transformer_cls_name,
                transformer_override_safetensor=kwargs.get("transformer_override_safetensor"),
                attention_backend=kwargs.get("attention_backend"),
            )
            return self._post_load_transformer(transformer, kwargs)
        return super()._load_transformer(init_from=init_from, **kwargs)

    def _post_load_transformer(self, transformer: torch.nn.Module, kwargs: dict[str, Any]) -> torch.nn.Module:
        from fastvideo.train.utils.module_state import apply_trainable
        from fastvideo.training.activation_checkpoint import (
            apply_activation_checkpointing, )

        training_config = kwargs["training_config"]
        trainable = bool(kwargs.get("trainable", True))
        ckpt_type = (kwargs.get("enable_gradient_checkpointing_type")
                     or getattr(getattr(training_config, "model", None), "enable_gradient_checkpointing_type", None))
        if trainable and ckpt_type:
            transformer = apply_activation_checkpointing(transformer, checkpointing_type=ckpt_type)
        if self._enable_lora_if_configured(transformer):
            return transformer
        return apply_trainable(transformer, trainable=trainable)

    # ------------------------------------------------------------------
    # Stage handoff
    # ------------------------------------------------------------------

    def _load_stage_checkpoint(self, checkpoint: str, *, controlnet_only: bool) -> None:
        """Overlay weights from a previous stage's DCP training checkpoint.

        ``controlnet_only`` keeps the backbone as loaded from ``init_from`` and
        overlays just ``depth_controlnet.*``. That is the only thing a
        frozen-backbone stage leaves behind: a checkpoint stores parameters that
        had ``requires_grad``, so a stage trained with ``freeze_backbone=true``
        saved the branch alone. Optimizer, scheduler, dataloader, callback and RNG
        state are never requested.

        The source role is always ``student`` regardless of which role is loading:
        a teacher or critic inherits from whichever run produced the weights, and
        that run trained them as its student.
        """
        optional = _WIDE_OPTIONAL_SUBSTRINGS if self._enable_wide_fov else None
        had_wide = False
        if self._enable_wide_fov:
            had_wide = dcp_model_has_key_substring(
                checkpoint,
                state_key="roles.student.transformer",
                source_prefix="",
                substring=".wca.",
            )
        loaded = load_prefixed_model_weights_from_dcp(
            self.transformer,
            checkpoint,
            state_key="roles.student.transformer",
            source_prefix="",
            required_target_prefixes=("depth_controlnet.", ) if controlnet_only else None,
            optional_target_substrings=optional,
        )
        logger.info("Loaded %d tensors from %s (controlnet_only=%s)", loaded, checkpoint, controlnet_only)
        if self._enable_wide_fov and not had_wide:
            warm_start = getattr(self.transformer, "warm_start_wide_branch", None)
            if not callable(warm_start):
                raise RuntimeError(f"enable_wide_fov=true but {checkpoint} has no wide-FOV (.wca.) tensors and this "
                                   "transformer cannot warm-start them.")
            warm_start()
            logger.info("Warm-started the wide-FOV branch from self-attention (output gate zeroed).")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        from fastvideo.distributed import get_sp_group, get_world_group

        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()

        sp_world_size = int(training_config.distributed.sp_size or 1)
        world_size = self.world_group.world_size
        if world_size % sp_world_size != 0:
            raise ValueError(f"world_size={world_size} must be divisible by sp_size={sp_world_size}")
        self.dataloader = build_v2v_depth_train_dataloader(
            training_config.data,
            num_sp_groups=world_size // sp_world_size,
            sp_world_size=sp_world_size,
            global_rank=self.world_group.rank,
            include_depth=self._enable_depth and self._enable_controlnet,
            include_wide=self._enable_wide_fov,
        )
        self.start_step = 0

    # ------------------------------------------------------------------
    # Runtime primitives
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        training_batch = super().prepare_batch(raw_batch, generator=generator, latents_source=latents_source)

        tc = self.training_config
        assert tc is not None
        dtype = self._get_training_dtype()
        device = self.device
        num_latent_t = int(tc.data.num_latent_t)

        def _take(key: str) -> torch.Tensor:
            tensor = raw_batch[key][:, :, :num_latent_t].to(device=device, dtype=dtype)
            return normalize_dit_input("wan", tensor, self.vae)

        if "control_latent" not in raw_batch:
            raise KeyError("v2v training requires 'control_latent' (the source clip's VAE latent) in the batch")
        training_batch.control_latent = _take("control_latent")
        if self._enable_controlnet and self._enable_depth:
            training_batch.depth_latent = _take("depth_latent")
            if self._enable_wide_fov:
                training_batch.depth_wide_latent = _take("depth_wide_latent")

        # Classifier-free guidance over the *control* channel: dropping the
        # source clip on a fraction of steps is what lets inference trade off
        # control strength. The text dropout is handled by the training method.
        if self._control_dropout > 0.0 and self.training_config is not None:
            drop = torch.rand((), generator=generator, device="cpu").item() < self._control_dropout
            if drop:
                training_batch.control_latent = torch.zeros_like(training_batch.control_latent)
        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, torch.Tensor] | None,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        batch: TrainingBatch | None = None,
    ) -> dict[str, Any]:
        kwargs = super()._build_distill_input_kwargs(noise_input, timestep, text_dict, clean_x=clean_x, aug_t=aug_t)
        if batch is None:
            return kwargs
        kwargs["hidden_states"] = self._concat_conditioning(kwargs["hidden_states"], batch)
        if batch.depth_latent is not None:
            kwargs["depth_latent"] = batch.depth_latent
        if batch.depth_wide_latent is not None:
            kwargs["depth_wide_latent"] = batch.depth_wide_latent
        return kwargs

    def _concat_conditioning(self, hidden_states: torch.Tensor, batch: TrainingBatch) -> torch.Tensor:
        """Fill Wan I2V's conditioning channels with the source clip.

        ``hidden_states`` arrives as ``[B, C, T, H, W]``. The released A14B
        checkpoint expects ``[noise | mask | condition]``; the mask is all ones
        because in video-to-video every frame is conditioned, unlike
        image-to-video where only frame 0 is.
        """
        control = batch.control_latent
        if control is None:
            raise RuntimeError("control_latent is missing; prepare_batch must run first")
        expected = int(self.transformer.in_channels)
        noise_channels = hidden_states.shape[1]
        mask_channels = expected - noise_channels - control.shape[1]
        if mask_channels < 0:
            raise ValueError(f"Wan transformer in_channels={expected} is too small for "
                             f"{noise_channels} noise + {control.shape[1]} condition channels. A "
                             "video-to-video run needs an I2V checkpoint, not T2V.")
        parts = [hidden_states]
        if mask_channels:
            parts.append(
                torch.ones(
                    hidden_states.shape[0],
                    mask_channels,
                    *hidden_states.shape[2:],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ))
        parts.append(control.to(device=hidden_states.device, dtype=hidden_states.dtype))
        return torch.cat(parts, dim=1)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from fastvideo.forward_context import set_forward_context

        device_type = self.device.type
        dtype = self._get_training_dtype()
        if conditional:
            text_dict = batch.conditional_dict
            if text_dict is None:
                raise RuntimeError("Missing conditional_dict in TrainingBatch")
        else:
            text_dict = self._get_uncond_text_dict(batch, cfg_uncond=cfg_uncond)

        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

        if noisy_latents.is_floating_point():
            noisy_latents = noisy_latents.to(dtype=dtype)

        with torch.autocast(device_type, dtype=dtype), set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=attn_metadata,
        ):
            input_kwargs = self._build_distill_input_kwargs(
                noisy_latents,
                timestep,
                text_dict,
                clean_x=clean_x,
                aug_t=aug_t,
                batch=batch,
            )
            transformer = self._get_transformer(timestep)
            return transformer(**input_kwargs).permute(0, 2, 1, 3, 4)
