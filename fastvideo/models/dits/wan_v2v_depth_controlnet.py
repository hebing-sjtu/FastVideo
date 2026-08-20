# SPDX-License-Identifier: Apache-2.0
"""Wan video-to-video DiT with a depth ControlNet branch.

The backbone is an unmodified Wan I2V transformer. Wan I2V already accepts a
full-length clean conditioning latent in its extra input channels (the released
A14B checkpoint takes ``[noise 16 | mask 4 | condition 16]``), so video-to-video
needs no channel-count change: the caller fills the conditioning slot with the
source clip instead of a zero-padded first frame. See
``fastvideo/train/models/wan/wan_v2v_depth.py``.

Depth enters separately, through a parallel trunk of narrow transformer blocks
that reads the VAE-encoded depth of the target clip and adds a zero-initialised
residual into the backbone's residual stream every ``layer_stride`` blocks. The
zero init is what makes depth optional in both directions: the branch is a no-op
at step 0, and a forward that supplies no depth latent skips it entirely.

An opt-in wide-FOV branch gives the control trunk peripheral context. The wide
render keeps the narrow render's resolution and frame count and only scales
``fx,fy`` by ``k = wide_fov_scale < 1``, so wide and narrow tokens share a ray
coordinate once the wide spatial positions are divided by ``k``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_all_to_all_4D,
    sequence_model_parallel_shard,
)
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import (
    FP32LayerNorm,
    RMSNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import (
    _apply_rotary_emb,
    get_1d_rotary_pos_embed,
    get_rotary_pos_embed,
)
from fastvideo.layers.visual_embedding import PatchEmbed
from fastvideo.models.dits.wanvideo import WanTransformer3DModel
from fastvideo.platforms import AttentionBackendEnum, current_platform

# Control modalities this trunk can consume. Depth is the only geometry signal
# the video-to-video recipe uses; the source clip rides the backbone's own
# conditioning channels rather than the control trunk.
CONTROL_MODALITIES: tuple[str, ...] = ("depth", )


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.detach().zero_()
    return module


def wide_rope_dim_list(head_dim: int) -> list[int]:
    """Per-axis rotary dims for a (t, h, w) grid at ``head_dim``."""
    return [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]


class WanDepthWideCrossAttention(nn.Module):
    """Wide-FOV peripheral cross-attention sublayer (``wca``) for a control block.

    Query is the block's narrow control tokens; key/value are the same-scene
    wide-FOV depth tokens, used as a KV-only reference. Runs on the
    sequence-parallel data layout: q/k/v arrive sequence-sharded
    ``[B, L_local, H, D]``, one all-to-all gives each rank the full sequence with
    a head subset, and a second restores the sharded layout.

    Trained from scratch alongside the rest of the trunk, so ``to_out`` is a
    plain projection rather than zero-init: the branch contributes from step 0.
    Attention is full over the clip; the shared-ray spatial RoPE plus per-frame
    time RoPE are what let the model localize the relevant wide tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: str,
        eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_heads
        head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(head_dim, eps=eps)
            self.norm_k = RMSNorm(head_dim, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unsupported qk_norm: {qk_norm!r}")
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends
            or (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        wide_tokens: torch.Tensor,
        *,
        original_seq_len: int,
        wide_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """``query_tokens`` / ``wide_tokens``: sequence-sharded ``[B, L_local, dim]`` of equal length."""
        cos_q, sin_q, cos_k, sin_k = wide_freqs
        q = self.norm_q(self.to_q(query_tokens))
        k = self.norm_k(self.to_k(wide_tokens))
        v = self.to_v(wide_tokens)
        q = q.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        k = k.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        v = v.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        qkv = torch.cat([q, k, v], dim=0)
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
        pad_seq_len = qkv.shape[1] - original_seq_len
        qkv = qkv[:, :original_seq_len, :, :]
        q_f, k_f, v_f = qkv.chunk(3, dim=0)

        roped_q = _apply_rotary_emb(q_f, cos_q, sin_q, is_neox_style=False)
        roped_k = _apply_rotary_emb(k_f, cos_k, sin_k, is_neox_style=False)
        output = self.attn(roped_q, roped_k, v_f, freqs_cis=None)

        output = torch.nn.functional.pad(output, (0, 0, 0, 0, 0, pad_seq_len))
        output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)
        return self.to_out(output.flatten(2))


class WanDepthControlBlock(nn.Module):
    """Self-attention + MLP block that emits a zero-init residual for one Wan block."""

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str,
        eps: float,
        supported_attention_backends: tuple[Any, ...] | None,
        out_dim: int,
        temb_dim: int,
        enable_wide: bool = False,
    ) -> None:
        super().__init__()
        # The trunk runs narrower than the backbone, so the shared timestep
        # embedding needs projecting down before it can modulate these blocks.
        self.temb_proj = nn.Linear(temb_dim, dim) if temb_dim != dim else nn.Identity()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.attn1 = DistributedAttention(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix="WanDepthControlNet.attn1",
        )
        self.num_attention_heads = num_heads
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
            compute_dtype=torch.float32,
        )
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.proj_out = zero_module(nn.Linear(dim, out_dim))
        self.wca: WanDepthWideCrossAttention | None = None
        if enable_wide:
            self.wca = WanDepthWideCrossAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=supported_attention_backends,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
        *,
        wide_states: torch.Tensor | None = None,
        wide_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = hidden_states.dtype
        if not isinstance(self.temb_proj, nn.Identity):
            temb = self.temb_proj(temb)
        if temb.dim() == 4:
            shift_msa, scale_msa, gate_msa, _c_shift, _c_scale, c_gate_msa = (self.scale_shift_table.unsqueeze(0) +
                                                                              temb.float()).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, _c_shift, _c_scale, c_gate_msa = (self.scale_shift_table +
                                                                              temb.float()).chunk(6, dim=1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).to(orig_dtype)
        query = self.norm_q(self.to_q(norm_hidden_states))
        key = self.norm_k(self.to_k(norm_hidden_states))
        value = self.to_v(norm_hidden_states)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output, _ = self.attn1(query, key, value, original_seq_len, freqs_cis=freqs_cis)
        attn_output = self.to_out(attn_output.flatten(2)).squeeze(1)

        # Wide-FOV peripheral context: the query is the same normalized narrow
        # tokens, keys/values are the wide tokens under the same per-frame
        # modulation, added as an independent residual before the FFN.
        wide_residual = None
        if self.wca is not None and wide_states is not None:
            if wide_freqs is None:
                raise ValueError("wide_freqs is required when the wide branch is enabled")
            norm_wide = (self.norm1(wide_states.float()) * (1 + scale_msa) + shift_msa).to(orig_dtype)
            wide_residual = self.wca(
                norm_hidden_states,
                norm_wide,
                original_seq_len=original_seq_len,
                wide_freqs=wide_freqs,
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


class WanDepthControlNet(nn.Module):
    """Parallel control trunk over VAE-encoded depth latents."""

    def _validate_control_sp_head_split(self, control_num_heads: int, backbone_heads: int) -> None:
        """Control-block self-attention shards heads with an exact all-to-all split.

        The trunk's head count is independent of the backbone's, so a world size
        that divides the backbone heads can still leave the trunk unable to split
        its own. Without this check the failure surfaces as an opaque NCCL
        "does not divide equally" abort at the first forward.
        """
        sp_world_size = get_sp_world_size()
        if control_num_heads % sp_world_size != 0:
            common = math.gcd(control_num_heads, backbone_heads)
            valid = [n for n in range(1, common + 1) if common % n == 0]
            raise ValueError(
                f"controlnet_num_heads={control_num_heads} must be divisible by the sequence parallel size "
                f"({sp_world_size}); the control trunk shards its heads across the SP group just like the "
                f"backbone ({backbone_heads} heads). Valid sequence parallel sizes for this model: {valid}.")

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
        wide_fov_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.enabled_modalities = tuple(enabled_modalities)
        if not self.enabled_modalities:
            raise ValueError("WanDepthControlNet requires at least one enabled modality.")
        unknown = [name for name in self.enabled_modalities if name not in CONTROL_MODALITIES]
        if unknown:
            raise ValueError(f"Unknown control modalities {unknown}; supported: {list(CONTROL_MODALITIES)}")
        self.layer_stride = max(1, int(layer_stride))

        config = base_transformer.config
        wan_dim = int(config.num_attention_heads * config.attention_head_dim)
        control_dim = int(control_dim)
        control_ffn_dim = int(control_ffn_dim)
        control_num_heads = int(control_num_heads)
        if control_dim % control_num_heads != 0:
            raise ValueError(f"controlnet_dim={control_dim} must be divisible by "
                             f"controlnet_num_heads={control_num_heads}")
        self._validate_control_sp_head_split(control_num_heads, int(config.num_attention_heads))
        self.control_dim = control_dim
        self.control_num_heads = control_num_heads
        self.control_head_dim = control_dim // control_num_heads
        self.enable_wide = bool(enable_wide)
        self.wide_fov_scale = float(wide_fov_scale)

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
        # Concatenate per-modality embeddings on the feature axis and project
        # back, so each modality's signal stays distinct and the projection
        # learns the combination. One modality needs no fusion.
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
                WanDepthControlBlock(
                    dim=control_dim,
                    ffn_dim=control_ffn_dim,
                    num_heads=control_num_heads,
                    qk_norm=config.qk_norm,
                    eps=config.eps,
                    supported_attention_backends=base_transformer._supported_attention_backends,
                    out_dim=wan_dim,
                    temb_dim=wan_dim,
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
        """Embed + fuse the narrow control latents into control tokens.

        Returns ``None`` when any enabled modality is absent, which is how the
        caller skips the control path for that forward.
        """
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
        """Embed + fuse the wide-FOV latents, mirroring :meth:`prepare`.

        Returns ``None`` when the wide branch is disabled or any wide modality is
        missing, so the caller skips the wide path entirely.
        """
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE table sized for the trunk's own head_dim.

        Control blocks attend at ``control_dim / control_num_heads``, which
        usually differs from the backbone head_dim, so reusing the backbone's
        table would mismatch inside ``_apply_rotary_emb``.
        """
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            grid_sizes,
            self.control_dim,
            self.control_num_heads,
            wide_rope_dim_list(self.control_head_dim),
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
        )
        return freqs_cos.to(device).float(), freqs_sin.to(device).float()

    def compute_wide_freqs_cis(
        self,
        grid_sizes: tuple[int, int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared-ray RoPE for the wide cross-attention: narrow Q, wide K.

        Spatial axes are centered on the latent grid, since the wide render keeps
        the narrow principal point: narrow Q sits at ``idx - c`` and wide K at
        ``(idx - c) / k``, so the same 3D ray lands at the same rotary position.
        The time axis is the unscaled frame index for both. Wide K positions are
        fractional, so cos/sin come straight from ``pos * inv_freq``.
        """
        num_frames, height, width = grid_sizes
        rope_dim_list = wide_rope_dim_list(self.control_head_dim)
        dtype = torch.float32 if current_platform.is_mps() else torch.float64
        k = self.wide_fov_scale
        center_h = (height - 1) / 2.0
        center_w = (width - 1) / 2.0
        t_pos = torch.arange(num_frames, dtype=dtype)
        h_narrow = torch.arange(height, dtype=dtype) - center_h
        w_narrow = torch.arange(width, dtype=dtype) - center_w
        cos_q, sin_q = self._wide_axis_rope(rope_dim_list, t_pos, h_narrow, w_narrow, dtype)
        cos_k, sin_k = self._wide_axis_rope(rope_dim_list, t_pos, h_narrow / k, w_narrow / k, dtype)
        return (cos_q.to(device).float(), sin_q.to(device).float(), cos_k.to(device).float(), sin_k.to(device).float())

    @staticmethod
    def _wide_axis_rope(
        rope_dim_list: list[int],
        t_pos: torch.Tensor,
        h_pos: torch.Tensor,
        w_pos: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (cos, sin) over the (t, h, w) grid for possibly-fractional positions.

        Matches ``get_nd_rotary_pos_embed``'s token layout: (t, h, w) meshgrid
        flattened with w fastest, per-axis rope dims concatenated on the feature
        axis.
        """
        num_frames, height, width = t_pos.numel(), h_pos.numel(), w_pos.numel()
        tt = t_pos.view(num_frames, 1, 1).expand(num_frames, height, width).reshape(-1)
        hh = h_pos.view(1, height, 1).expand(num_frames, height, width).reshape(-1)
        ww = w_pos.view(1, 1, width).expand(num_frames, height, width).reshape(-1)
        coss: list[torch.Tensor] = []
        sins: list[torch.Tensor] = []
        for dim_i, pos in zip(rope_dim_list, (tt, hh, ww), strict=True):
            emb_cos, emb_sin = get_1d_rotary_pos_embed(dim_i, pos.to(dtype), theta=10000, dtype=dtype, use_real=True)
            coss.append(emb_cos)
            sins.append(emb_sin)
        return torch.cat(coss, dim=1), torch.cat(sins, dim=1)

    def forward_block(
        self,
        block_idx: int,
        control_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
        *,
        wide_states: torch.Tensor | None = None,
        wide_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        control_idx = self.block_to_control.get(block_idx)
        if control_idx is None:
            return control_states, None
        return self.control_blocks[control_idx](
            control_states,
            temb,
            freqs_cis,
            original_seq_len,
            wide_states=wide_states,
            wide_freqs=wide_freqs,
        )

    def warm_start_wide_branch(self) -> None:
        """Seed each ``wca`` from its block's self-attention, keeping ``to_out`` zeroed.

        Used when resuming from a checkpoint that predates the wide branch: the
        projections start somewhere sensible rather than at a fresh random init,
        while the zeroed output keeps the branch a no-op until it trains.
        """
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


def build_depth_controlnet(base_transformer: nn.Module, config: Any) -> WanDepthControlNet | None:
    """Build the control trunk from ``control_*`` fields stamped onto the DiT config.

    The training plugin sets these at load time; the released Wan config has no
    counterpart, so every field falls back to a default that keeps the branch off.
    """
    if not bool(getattr(config, "control_enable_controlnet", False)):
        return None
    enabled = [name for name in CONTROL_MODALITIES if bool(getattr(config, f"control_enable_{name}", False))]
    if not enabled:
        raise ValueError("control_enable_controlnet=True but no control modality is enabled")
    return WanDepthControlNet(
        base_transformer,
        enabled_modalities=enabled,
        layer_stride=int(getattr(config, "control_layer_stride", 1)),
        control_dim=int(getattr(config, "control_dim", 1024)),
        control_ffn_dim=int(getattr(config, "control_ffn_dim", 4096)),
        control_num_heads=int(getattr(config, "control_num_heads", 16)),
        enable_wide=bool(getattr(config, "control_enable_wide", False)),
        wide_fov_scale=float(getattr(config, "control_wide_fov_scale", 0.5)),
    )


class WanV2VDepthTransformer3DModel(WanTransformer3DModel):
    """Wan transformer plus a depth ControlNet branch.

    Subclassing keeps every backbone parameter name identical to the released
    checkpoint, so strict loading is unchanged and only the new
    ``depth_controlnet.*`` keys need initializing (see
    ``fastvideo/models/loader/fsdp_load.py``).
    """

    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self.depth_controlnet = build_depth_controlnet(self, config)
        if self.depth_controlnet is not None and bool(getattr(config, "control_freeze_backbone", False)):
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze everything except the control trunk."""
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith("depth_controlnet."))

    def warm_start_wide_branch(self) -> None:
        if self.depth_controlnet is not None:
            self.depth_controlnet.warm_start_wide_branch()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        depth_latent: torch.Tensor | None = None,
        depth_wide_latent: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.depth_controlnet is None:
            return super().forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_hidden_states_image,
                guidance,
                **kwargs,
            )
        del guidance, kwargs

        orig_dtype = hidden_states.dtype
        if encoder_hidden_states is not None and not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = (encoder_hidden_states_image[0]
                                           if len(encoder_hidden_states_image) > 0 else None)

        batch_size, _num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        grid_sizes = (post_patch_num_frames, post_patch_height, post_patch_width)

        d = self.hidden_size // self.num_attention_heads
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            grid_sizes,
            self.hidden_size,
            self.num_attention_heads,
            wide_rope_dim_list(d),
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
        )
        freqs_cis = (freqs_cos.to(hidden_states.device).float(), freqs_sin.to(hidden_states.device).float())

        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        control_states = self.depth_controlnet.prepare(depth_latent=depth_latent)
        wide_states = (self.depth_controlnet.prepare_wide(
            depth_wide_latent=depth_wide_latent) if control_states is not None else None)

        hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)
        control_freqs_cis = None
        wide_freqs = None
        if control_states is not None:
            control_states, _ = sequence_model_parallel_shard(control_states, dim=1)
            control_freqs_cis = self.depth_controlnet.compute_freqs_cis(grid_sizes, hidden_states.device)
            if wide_states is not None:
                wide_states, _ = sequence_model_parallel_shard(wide_states, dim=1)
                wide_freqs = self.depth_controlnet.compute_wide_freqs_cis(grid_sizes, hidden_states.device)

        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=ts_seq_len,
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states_image

        if current_platform.is_mps() or current_platform.is_npu():
            encoder_hidden_states = encoder_hidden_states.to(orig_dtype)
        assert encoder_hidden_states.dtype == orig_dtype

        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    original_seq_len,
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis, original_seq_len)
            if control_states is not None:
                control_states, residual = self.depth_controlnet.forward_block(
                    i,
                    control_states,
                    timestep_proj,
                    control_freqs_cis,
                    original_seq_len,
                    wide_states=wide_states,
                    wide_freqs=wide_freqs,
                )
                if residual is not None:
                    hidden_states = hidden_states + residual

        if temb.dim() == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
                                              p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = WanV2VDepthTransformer3DModel
