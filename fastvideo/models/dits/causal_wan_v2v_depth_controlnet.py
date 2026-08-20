# SPDX-License-Identifier: Apache-2.0
"""Autoregressive (block-causal + KV cache) Wan video-to-video DiT with depth ControlNet.

Same control trunk as ``wan_v2v_depth_controlnet``, rebuilt on the block-causal
attention so the model can generate one chunk at a time against a committed KV
cache. Every parameter name in the trunk matches the bidirectional model's, which
is what lets the autoregressive stage overlay a bidirectional-trained
``depth_controlnet.*`` onto a fresh backbone.

Two paths, distinguished the same way the backbone distinguishes them:

* ``kv_cache is None`` — training. One masked pass over the whole window using
  the backbone's block-causal (or teacher-forcing) block mask.
* ``kv_cache`` given — inference. One chunk at a time against the cache.

Depth is known for every frame in both paths, including at inference: the engine
renders the current chunk's depth before it is generated. The trunk still runs
block-causally so its token positions and cache layout stay aligned with the
backbone's, and so the wide branch cannot read future chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask

from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.layers.layernorm import (
    RMSNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.models.dits.causal_wanvideo import (
    GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES,
    CausalWanSelfAttention,
    CausalWanTransformer3DModel,
)
from fastvideo.models.dits.wan_v2v_depth_controlnet import (
    CONTROL_MODALITIES,
    wide_rope_dim_list,
    zero_module,
)
from fastvideo.platforms import current_platform


class CausalWanDepthWideCrossAttention(nn.Module):
    """Block-causal wide-FOV cross-attention (``wca``) for a control block.

    Query is the narrow control tokens, key/value the wide-FOV depth tokens at
    the same positions. Reusing :class:`CausalWanSelfAttention` for the attention
    itself is what keeps this causal: it takes q, k and v separately, so a
    same-length wide key/value stream inherits the block mask during training and
    the sliding KV window at inference without any extra masking logic.

    ``to_out`` is zero-init so the branch is a no-op until trained. That is what
    makes it safe to warm-start an autoregressive run from a bidirectional
    checkpoint that predates the wide branch.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: str,
        eps: float,
        local_attn_size: int = -1,
        sink_size: int = 0,
        rope_cache_policy: str = "absolute",
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_heads
        head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = zero_module(nn.Linear(dim, dim, bias=True))
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(head_dim, eps=eps)
            self.norm_k = RMSNorm(head_dim, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unsupported qk_norm: {qk_norm!r}")
        self.attn = CausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
            rope_cache_policy=rope_cache_policy,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        wide_tokens: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | None,
        kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        frame_seqlen: int = 1560,
    ) -> torch.Tensor:
        query = self.norm_q(self.to_q(query_tokens))
        key = self.norm_k(self.to_k(wide_tokens))
        value = self.to_v(wide_tokens)
        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        attn_output = self.attn(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            frame_seqlen=frame_seqlen,
        )
        return self.to_out(attn_output.flatten(2)).squeeze(1)


class CausalWanDepthControlBlock(nn.Module):
    """Block-causal self-attention + MLP emitting a zero-init residual.

    Modulation follows the backbone's causal block exactly -- per-frame
    ``[B, T, 6, dim]`` modulation unflattened over tokens-per-frame -- so control
    and backbone stay phase-aligned frame by frame.
    """

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str,
        eps: float,
        out_dim: int,
        temb_dim: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        rope_cache_policy: str = "absolute",
        enable_wide: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.temb_proj = nn.Linear(temb_dim, dim) if temb_dim != dim else nn.Identity()
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.attn1 = CausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
            rope_cache_policy=rope_cache_policy,
        )
        head_dim = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(head_dim, eps=eps)
            self.norm_k = RMSNorm(head_dim, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unsupported qk_norm: {qk_norm!r}")
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.proj_out = zero_module(nn.Linear(dim, out_dim))
        self.wca: CausalWanDepthWideCrossAttention | None = None
        if enable_wide:
            self.wca = CausalWanDepthWideCrossAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=qk_norm,
                eps=eps,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                rope_cache_policy=rope_cache_policy,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | None,
        kv_cache: dict | None = None,
        wide_kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        frame_seqlen: int | None = None,
        wide_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        if not isinstance(self.temb_proj, nn.Identity):
            temb = self.temb_proj(temb)
        temb_seq_len = temb.shape[1]
        tokens_per_temb = hidden_states.shape[1] // temb_seq_len
        frame_seqlen = tokens_per_temb if frame_seqlen is None else int(frame_seqlen)
        bs, _seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        e = self.scale_shift_table + temb
        assert e.shape == (bs, temb_seq_len, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, _c_shift, _c_scale, c_gate_msa = e.chunk(6, dim=2)

        norm_hidden_states = (self.norm1(hidden_states).unflatten(dim=1, sizes=(temb_seq_len, tokens_per_temb)) *
                              (1 + scale_msa) + shift_msa).flatten(1, 2)
        query = self.norm_q(self.to_q(norm_hidden_states))
        key = self.norm_k(self.to_k(norm_hidden_states))
        value = self.to_v(norm_hidden_states)
        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            frame_seqlen=frame_seqlen,
        )
        attn_output = self.to_out(attn_output.flatten(2)).squeeze(1)

        wide_residual = None
        if self.wca is not None and wide_states is not None:
            norm_wide = (self.norm1(wide_states).unflatten(dim=1, sizes=(temb_seq_len, tokens_per_temb)) *
                         (1 + scale_msa) + shift_msa).flatten(1, 2)
            wide_residual = self.wca(
                norm_hidden_states,
                norm_wide,
                freqs_cis,
                block_mask,
                wide_kv_cache,
                current_start,
                cache_start,
                frame_seqlen=frame_seqlen,
            )

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states,
            attn_output,
            gate_msa,
            null_shift,
            null_scale,
        )
        norm_hidden_states = norm_hidden_states.to(orig_dtype)
        hidden_states = hidden_states.to(orig_dtype)
        if wide_residual is not None:
            hidden_states = (hidden_states + wide_residual).to(orig_dtype)
        hidden_states = self.mlp_residual(hidden_states, self.ffn(norm_hidden_states), c_gate_msa).to(orig_dtype)
        return hidden_states, self.proj_out(hidden_states)


class CausalWanDepthControlNet(nn.Module):
    """Block-causal control trunk over VAE-encoded depth latents."""

    def __init__(
        self,
        base_transformer: nn.Module,
        *,
        enabled_modalities: Sequence[str] = ("depth", ),
        layer_stride: int = 1,
        latent_channels: int = 16,
        control_dim: int = 1024,
        control_ffn_dim: int = 4096,
        control_num_heads: int = 16,
        enable_wide: bool = False,
    ) -> None:
        super().__init__()
        self.enabled_modalities = tuple(enabled_modalities)
        if not self.enabled_modalities:
            raise ValueError("CausalWanDepthControlNet requires at least one enabled modality.")
        unknown = [name for name in self.enabled_modalities if name not in CONTROL_MODALITIES]
        if unknown:
            raise ValueError(f"Unknown control modalities {unknown}; supported: {list(CONTROL_MODALITIES)}")
        self.layer_stride = max(1, int(layer_stride))

        config = base_transformer.config
        wan_dim = int(config.num_attention_heads * config.attention_head_dim)
        control_dim = int(control_dim)
        control_num_heads = int(control_num_heads)
        if control_dim % control_num_heads != 0:
            raise ValueError(f"controlnet_dim={control_dim} must be divisible by "
                             f"controlnet_num_heads={control_num_heads}")
        self.control_dim = control_dim
        self.control_num_heads = control_num_heads
        self.control_head_dim = control_dim // control_num_heads
        self.enable_wide = bool(enable_wide)

        self.embeddings = nn.ModuleDict({
            name:
            PatchEmbed(
                in_chans=latent_channels,
                embed_dim=control_dim,
                patch_size=config.patch_size,
                flatten=False,
            )
            for name in self.enabled_modalities
        })
        num_modalities = len(self.enabled_modalities)
        self.fusion_proj: nn.Module = (nn.Linear(control_dim *
                                                 num_modalities, control_dim) if num_modalities > 1 else nn.Identity())

        self.control_blocks = nn.ModuleList()
        self.block_to_control: dict[int, int] = {}
        for block_idx in range(int(config.num_layers)):
            if block_idx % self.layer_stride != 0:
                continue
            self.block_to_control[block_idx] = len(self.control_blocks)
            self.control_blocks.append(
                CausalWanDepthControlBlock(
                    dim=control_dim,
                    ffn_dim=int(control_ffn_dim),
                    num_heads=control_num_heads,
                    qk_norm=config.qk_norm,
                    eps=config.eps,
                    out_dim=wan_dim,
                    temb_dim=wan_dim,
                    local_attn_size=config.local_attn_size,
                    sink_size=config.arch_config.sink_size,
                    rope_cache_policy=config.arch_config.rope_cache_policy,
                    enable_wide=self.enable_wide,
                ))

        if self.enable_wide:
            self.wide_embeddings = nn.ModuleDict({
                name:
                PatchEmbed(
                    in_chans=latent_channels,
                    embed_dim=control_dim,
                    patch_size=config.patch_size,
                    flatten=False,
                )
                for name in self.enabled_modalities
            })
            self.wide_fusion_proj: nn.Module = (nn.Linear(control_dim * num_modalities, control_dim)
                                                if num_modalities > 1 else nn.Identity())

    def prepare(self, *, depth_latent: torch.Tensor | None = None) -> torch.Tensor | None:
        tensors = {"depth": depth_latent}
        embeds: list[torch.Tensor] = []
        for name in self.enabled_modalities:
            value = tensors.get(name)
            if value is None:
                return None
            embeds.append(self.embeddings[name](value).flatten(2).transpose(1, 2))
        fused = embeds[0] if len(embeds) == 1 else torch.cat(embeds, dim=-1)
        return self.fusion_proj(fused)

    def prepare_wide(self, *, depth_wide_latent: torch.Tensor | None = None) -> torch.Tensor | None:
        if not self.enable_wide:
            return None
        tensors = {"depth": depth_wide_latent}
        embeds: list[torch.Tensor] = []
        for name in self.enabled_modalities:
            value = tensors.get(name)
            if value is None:
                return None
            embeds.append(self.wide_embeddings[name](value).flatten(2).transpose(1, 2))
        fused = embeds[0] if len(embeds) == 1 else torch.cat(embeds, dim=-1)
        return self.wide_fusion_proj(fused)

    def compute_freqs_cis(
        self,
        grid_sizes: tuple[int, int, int],
        device: torch.device,
        *,
        start_frame: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE table at the trunk's own head_dim over the same (T, H, W) grid."""
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            grid_sizes,
            self.control_dim,
            self.control_num_heads,
            wide_rope_dim_list(self.control_head_dim),
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame,
        )
        return freqs_cos.to(device), freqs_sin.to(device)

    def forward_block(
        self,
        block_idx: int,
        control_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | None,
        *,
        kv_cache: dict | None = None,
        wide_kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        frame_seqlen: int | None = None,
        wide_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        control_idx = self.block_to_control.get(block_idx)
        if control_idx is None:
            return control_states, None
        return self.control_blocks[control_idx](
            control_states,
            temb,
            freqs_cis,
            block_mask,
            kv_cache=kv_cache,
            wide_kv_cache=wide_kv_cache,
            current_start=current_start,
            cache_start=cache_start,
            frame_seqlen=frame_seqlen,
            wide_states=wide_states,
        )

    def warm_start_wide_branch(self) -> None:
        """Seed each ``wca`` from its block's self-attention, keeping ``to_out`` zeroed."""
        for block in self.control_blocks:
            if block.wca is None:
                continue
            with torch.no_grad():
                block.wca.to_q.weight.copy_(block.to_q.weight)
                block.wca.to_q.bias.copy_(block.to_q.bias)
                block.wca.to_k.weight.copy_(block.to_k.weight)
                block.wca.to_k.bias.copy_(block.to_k.bias)
                block.wca.to_v.weight.copy_(block.to_v.weight)
                block.wca.to_v.bias.copy_(block.to_v.bias)
                block.wca.to_out.weight.zero_()
                block.wca.to_out.bias.zero_()

    @property
    def num_control_blocks(self) -> int:
        return len(self.control_blocks)


class CausalWanV2VDepthTransformer3DModel(CausalWanTransformer3DModel):
    """Block-causal Wan transformer plus a depth ControlNet branch."""

    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self.depth_controlnet: CausalWanDepthControlNet | None = None
        if bool(getattr(config, "control_enable_controlnet", False)):
            enabled = [name for name in CONTROL_MODALITIES if bool(getattr(config, f"control_enable_{name}", False))]
            if not enabled:
                raise ValueError("control_enable_controlnet=True but no control modality is enabled")
            self.depth_controlnet = CausalWanDepthControlNet(
                self,
                enabled_modalities=enabled,
                layer_stride=int(getattr(config, "control_layer_stride", 1)),
                control_dim=int(getattr(config, "control_dim", 1024)),
                control_ffn_dim=int(getattr(config, "control_ffn_dim", 4096)),
                control_num_heads=int(getattr(config, "control_num_heads", 16)),
                enable_wide=bool(getattr(config, "control_enable_wide", False)),
            )
            if bool(getattr(config, "control_freeze_backbone", False)):
                for name, param in self.named_parameters():
                    param.requires_grad_(name.startswith("depth_controlnet."))

    def warm_start_wide_branch(self) -> None:
        if self.depth_controlnet is not None:
            self.depth_controlnet.warm_start_wide_branch()

    def _prepare_control_states(
        self,
        depth_latent: torch.Tensor | None,
        depth_wide_latent: torch.Tensor | None,
        *,
        teacher_forcing: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Embed the control latents, tiling them for the teacher-forcing layout.

        Teacher forcing feeds the backbone ``[clean | noisy]`` over the same
        positions. Depth at frame *i* is the same regardless of which copy of
        frame *i* reads it, so the control stream is simply repeated to match.
        """
        assert self.depth_controlnet is not None
        control_states = self.depth_controlnet.prepare(depth_latent=depth_latent)
        if control_states is None:
            return None, None
        wide_states = self.depth_controlnet.prepare_wide(depth_wide_latent=depth_wide_latent)
        if teacher_forcing:
            control_states = torch.cat([control_states, control_states], dim=1)
            if wide_states is not None:
                wide_states = torch.cat([wide_states, wide_states], dim=1)
        return control_states, wide_states

    def _control_freqs(
        self,
        grid_sizes: tuple[int, int, int],
        device: torch.device,
        *,
        start_frame: int,
        teacher_forcing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.depth_controlnet is not None
        cos, sin = self.depth_controlnet.compute_freqs_cis(grid_sizes, device, start_frame=start_frame)
        if teacher_forcing:
            cos = torch.cat([cos, cos], dim=0)
            sin = torch.cat([sin, sin], dim=0)
        return cos, sin

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        start_frame: int = 0,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        depth_latent: torch.Tensor | None = None,
        depth_wide_latent: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.depth_controlnet is None:
            return super()._forward_train(
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_hidden_states_image,
                start_frame=start_frame,
                clean_x=clean_x,
                aug_t=aug_t,
                **kwargs,
            )
        return self._forward_with_control(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_image,
            start_frame=start_frame,
            clean_x=clean_x,
            aug_t=aug_t,
            depth_latent=depth_latent,
            depth_wide_latent=depth_wide_latent,
            kv_cache=None,
            control_kv_cache=None,
            control_wide_kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            cache_start=None,
        )

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        start_frame: int = 0,
        depth_latent: torch.Tensor | None = None,
        depth_wide_latent: torch.Tensor | None = None,
        control_kv_cache: list[dict] | None = None,
        control_wide_kv_cache: list[dict] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.depth_controlnet is None:
            return super()._forward_inference(
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_hidden_states_image,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                start_frame=start_frame,
                **kwargs,
            )
        return self._forward_with_control(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_image,
            start_frame=start_frame,
            clean_x=None,
            aug_t=None,
            depth_latent=depth_latent,
            depth_wide_latent=depth_wide_latent,
            kv_cache=kv_cache,
            control_kv_cache=control_kv_cache,
            control_wide_kv_cache=control_wide_kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
        )

    def _forward_with_control(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None,
        *,
        start_frame: int,
        clean_x: torch.Tensor | None,
        aug_t: torch.Tensor | None,
        depth_latent: torch.Tensor | None,
        depth_wide_latent: torch.Tensor | None,
        kv_cache: list[dict] | None,
        control_kv_cache: list[dict] | None,
        control_wide_kv_cache: list[dict] | None,
        crossattn_cache: list[dict] | None,
        current_start: int,
        cache_start: int | None,
    ) -> torch.Tensor:
        assert self.depth_controlnet is not None
        from fastvideo.distributed.parallel_state import get_sp_world_size

        orig_dtype = hidden_states.dtype
        teacher_forcing = clean_x is not None
        streaming = kv_cache is not None
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = (encoder_hidden_states_image[0]
                                           if len(encoder_hidden_states_image) > 0 else None)

        _batch_size, _num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        frame_seqlen = post_patch_height * post_patch_width

        # Mirror the backbone's RoPE window exactly: the relativistic policy
        # builds one fixed table over the whole attention window and lets
        # attention slice it per step, so it must not be re-based per chunk.
        if self.rope_cache_policy == "relativistic":
            max_attention_frames = (GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES
                                    if self.local_attn_size == -1 else self.local_attn_size)
            rope_num_frames = max_attention_frames * get_sp_world_size()
            rope_start_frame = 0
        else:
            rope_num_frames = post_patch_num_frames * get_sp_world_size()
            rope_start_frame = start_frame
        grid_sizes_rope = (rope_num_frames, post_patch_height, post_patch_width)

        d = self.hidden_size // self.num_attention_heads
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            grid_sizes_rope,
            self.hidden_size,
            self.num_attention_heads,
            wide_rope_dim_list(d),
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=rope_start_frame,
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)

        block_mask = None
        if not streaming:
            if teacher_forcing:
                if self.teacher_forcing_block_mask is None:
                    self.teacher_forcing_block_mask = self._prepare_teacher_forcing_mask(
                        device=hidden_states.device,
                        num_frames=num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                block_mask = self.teacher_forcing_block_mask
            else:
                if self.block_mask is None:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device=hidden_states.device,
                        num_frames=num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                block_mask = self.block_mask

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack([torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        control_states, wide_states = self._prepare_control_states(
            depth_latent,
            depth_wide_latent,
            teacher_forcing=teacher_forcing,
        )
        control_freqs = None
        if control_states is not None:
            control_freqs = self._control_freqs(
                grid_sizes_rope,
                hidden_states.device,
                start_frame=rope_start_frame,
                teacher_forcing=teacher_forcing,
            )

        encoder_hidden_states = torch.cat([
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(1, self.text_len - encoder_hidden_states.size(1),
                                            encoder_hidden_states.size(2)),
        ],
                                          dim=1)
        encoder_hidden_states_text = encoder_hidden_states

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        if current_platform.is_mps():
            encoder_hidden_states = encoder_hidden_states.to(orig_dtype)
        assert encoder_hidden_states.dtype == orig_dtype

        if teacher_forcing:
            clean_tokens = self.patch_embedding(clean_x).flatten(2).transpose(1, 2)
            hidden_states = torch.cat([clean_tokens, hidden_states], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(timestep)
            _, timestep_proj_clean, _, _ = self.condition_embedder(aug_t.flatten(), encoder_hidden_states_text, None)
            timestep_proj_clean = timestep_proj_clean.unflatten(1,
                                                                (6, self.hidden_size)).unflatten(dim=0,
                                                                                                 sizes=timestep.shape)
            timestep_proj = torch.cat([timestep_proj_clean, timestep_proj], dim=1)
            freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=0)
            freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=0)

        freqs_cis = (freqs_cos, freqs_sin)

        for block_index, block in enumerate(self.blocks):
            backbone_kwargs: dict[str, Any] = {"block_mask": block_mask, "frame_seqlen": frame_seqlen}
            if streaming:
                backbone_kwargs.update({
                    "kv_cache":
                    kv_cache[block_index],
                    "crossattn_cache": (crossattn_cache[block_index] if crossattn_cache is not None else None),
                    "current_start":
                    current_start,
                    "cache_start":
                    cache_start,
                })
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                backbone_kwargs.pop("crossattn_cache", None)
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states,
                                                                  timestep_proj, freqs_cis, **backbone_kwargs)
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis, **backbone_kwargs)

            if control_states is None:
                continue
            control_idx = self.depth_controlnet.block_to_control.get(block_index)
            control_states, residual = self.depth_controlnet.forward_block(
                block_index,
                control_states,
                timestep_proj,
                control_freqs,
                block_mask,
                kv_cache=(control_kv_cache[control_idx]
                          if control_kv_cache is not None and control_idx is not None else None),
                wide_kv_cache=(control_wide_kv_cache[control_idx]
                               if control_wide_kv_cache is not None and control_idx is not None else None),
                current_start=current_start,
                cache_start=cache_start,
                frame_seqlen=frame_seqlen,
                wide_states=wide_states,
            )
            if residual is not None:
                hidden_states = hidden_states + residual

        if teacher_forcing:
            hidden_states = hidden_states[:, hidden_states.shape[1] // 2:]

        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)
        return torch.stack(self.unpatchify(hidden_states, grid_sizes))

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache") is not None:
            return self._forward_inference(*args, **kwargs)
        return self._forward_train(*args, **kwargs)


EntryClass = CausalWanV2VDepthTransformer3DModel
