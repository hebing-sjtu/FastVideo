# SPDX-License-Identifier: Apache-2.0
"""Camera ControlNet for MiniMax-H3.

A parallel trunk of narrow transformer blocks reads a dense camera signal — the Plücker ray field of
the requested trajectory, optionally alongside proxy depth — and adds a zero-initialised residual
into the backbone's residual stream every ``layer_stride`` blocks. The backbone is normally frozen,
so this trunk is the only route by which a trajectory reaches the sampler.

Why a ControlNet and not another Ref2VA reference. A reference is *content*: the model reads it,
decides how much of it to believe, and the decision is made once for the whole clip. Camera motion
is not content, it is a per-token constraint — token ``(t, h, w)`` must show whatever the world puts
along one specific ray — and it has to bind tightly enough that the same proxy under two
trajectories produces two different videos. A residual added into every block at the exact token
that ray belongs to gives that binding; a reference in the prefix does not.

Three properties are inherited from how MiniMax-H3 packs its sequence, and all three are
load-bearing:

**The trunk mirrors the packed row layout.** H3's residual stream holds text, reference, audio and
target-video rows in one sequence, and sequence parallelism splits it into one contiguous block of
rows per rank. The control tokens are laid over the same rows — zero everywhere except the target
video rows — so the two streams shard identically and the residual add stays a local elementwise
operation. A trunk built over the video rows alone would be marginally cheaper and would cost two
extra all-gathers per block, because the video rows do not divide evenly across ranks.

**The residual is masked to the target video rows.** Every other row is either a reference the
caller fixed or an audio row this branch has no business touching. The trunk still *attends* over
all of its rows, which is what lets one frame's tokens see the rest of the trajectory, but only
target video rows receive the residual.

**The control blocks reuse the backbone's rotary table verbatim.**
:meth:`~fastvideo.models.dits.minimax_h3.MiniMaxH3Attention._apply_rotary_emb` rotates the leading
``2 * 3 * rope_freq_dim`` channels of each head and passes the rest through, so the same
``(cos, sin)`` applies at any head width at or above that. Keeping the control head dimension there
removes a second rotary table and any chance of the two streams disagreeing about where a row sits.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
    model_parallel_is_initialized,
)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.logger import init_logger
from fastvideo.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3FeedForward,
    MiniMaxH3Transformer3DModel,
    local_shard_rows,
)
from fastvideo.platforms import AttentionBackendEnum

logger = init_logger(__name__)

# Modality order is fixed: it decides the concatenation order into `fusion_proj` and is therefore
# part of the ControlNet's state-dict contract.
CAMERA_CONTROL_MODALITIES: tuple[str, ...] = ("camera", "depth")


class MiniMaxH3CameraControlBlock(nn.Module):
    """One block of the control trunk, plus its projection into the backbone's width.

    Structurally a narrower :class:`~fastvideo.models.dits.minimax_h3.MiniMaxH3TransformerBlock` with
    two differences. The AdaLN table is indexed by timestep alone rather than by
    ``(timestep, modality)``: the trunk spans every row of the packed sequence but carries only one
    kind of content, so a per-modality table would be three copies of the same thing. And the block
    returns a second tensor, ``proj_out(hidden_states)``, which the caller adds into the backbone.
    ``proj_out`` is zero-initialised by the loader, so an untrained ControlNet is exactly a no-op and
    step 0 reproduces the released model.

    The pre-norms are affine-free, unlike the backbone's. The backbone inherits its weights from the
    released checkpoint; here they would be new parameters whose effect the AdaLN scale immediately
    after can already express, so they would only add something for the loader to have a rule for.
    """

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        temb_dim: int,
        out_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.attn = MiniMaxH3Attention(
            dim,
            num_heads,
            head_dim,
            qk_norm_eps,
            supported_attention_backends,
            None,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.ff = MiniMaxH3FeedForward(dim, ffn_dim, quant_config=None, prefix=f"{prefix}.ff")
        self.adaln_proj = ReplicatedLinear(temb_dim, 6 * dim, bias=True, prefix=f"{prefix}.adaln_proj")
        # Zero-initialised by the loader, which keys on the name `proj_out`. Renaming this silently
        # turns the ControlNet on at step 0 with random weights.
        self.proj_out = ReplicatedLinear(dim, out_dim, bias=True, prefix=f"{prefix}.proj_out")

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modulation, _ = self.adaln_proj(F.silu(temb).to(self.adaln_proj.weight.dtype))
        modulation = modulation.to(hidden_states.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_msa.index_select(0, timestep_indices)) + shift_msa.index_select(0, timestep_indices)
        attention_output = self.attn(norm_hidden_states, rotary_emb, original_seq_len)
        hidden_states = residual + gate_msa.index_select(0, timestep_indices) * attention_output

        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_mlp.index_select(0, timestep_indices)) + shift_mlp.index_select(0, timestep_indices)
        hidden_states = residual + gate_mlp.index_select(0, timestep_indices) * self.ff(norm_hidden_states)

        control_residual, _ = self.proj_out(hidden_states)
        return hidden_states, control_residual


class MiniMaxH3CameraControlNet(nn.Module):
    """The control trunk: modality embedding, fusion, and one block per injection point."""

    def __init__(
        self,
        arch: Any,
        *,
        enabled_modalities: Sequence[str],
        layer_stride: int,
        control_dim: int,
        control_ffn_dim: int,
        control_num_heads: int,
        control_temb_dim: int,
        temb_in_dim: int,
        supported_attention_backends: tuple[AttentionBackendEnum, ...],
    ) -> None:
        # Imported here rather than at module scope: `fastvideo.pipelines.basic.minimax_h3` runs a
        # package __init__ that pulls in the whole inference pipeline, and a DiT module has no
        # business dragging that in just to read a channel count.
        from fastvideo.pipelines.basic.minimax_h3.camera import MINIMAX_H3_CAMERA_CHANNELS

        super().__init__()
        self.enabled_modalities = tuple(enabled_modalities)
        if not self.enabled_modalities:
            raise ValueError("MiniMaxH3CameraControlNet needs at least one enabled modality.")
        unknown = [name for name in self.enabled_modalities if name not in CAMERA_CONTROL_MODALITIES]
        if unknown:
            raise ValueError(f"Unknown camera-control modalities {unknown}; supported: "
                             f"{list(CAMERA_CONTROL_MODALITIES)}.")

        self.control_dim = int(control_dim)
        self.control_num_heads = int(control_num_heads)
        if self.control_dim % self.control_num_heads != 0:
            raise ValueError(f"camera_control_dim={self.control_dim} must be divisible by "
                             f"camera_control_num_heads={self.control_num_heads}.")
        self.control_head_dim = self.control_dim // self.control_num_heads
        self.layer_stride = max(1, int(layer_stride))

        self._validate_rope_width(int(arch.rope_freq_dim))
        self._validate_sp_head_split()

        patch_volume = math.prod(int(size) for size in arch.patch_size)
        # Both modalities arrive already patchified into rows, exactly like the backbone's video
        # rows, so these are linears over a patch and not convolutions over a latent grid. They
        # differ in width because a Plücker field is six raw channels while depth is a VAE latent.
        self._modality_widths = {
            "camera": MINIMAX_H3_CAMERA_CHANNELS * patch_volume,
            "depth": int(arch.in_channels) * patch_volume,
        }
        self.embeddings = nn.ModuleDict({
            name:
            ReplicatedLinear(self._modality_widths[name],
                             self.control_dim,
                             bias=True,
                             prefix=f"camera_controlnet.embeddings.{name}")
            for name in self.enabled_modalities
        })
        # Concatenating on the feature axis and projecting keeps each modality separable, which
        # summing them would not. One modality needs no fusion.
        self.fusion_proj: nn.Module | None = None
        if len(self.enabled_modalities) > 1:
            self.fusion_proj = ReplicatedLinear(self.control_dim * len(self.enabled_modalities),
                                                self.control_dim,
                                                bias=True,
                                                prefix="camera_controlnet.fusion_proj")

        self.temb_proj = ReplicatedLinear(int(temb_in_dim),
                                          int(control_temb_dim),
                                          bias=True,
                                          prefix="camera_controlnet.temb_proj")

        self.control_blocks = nn.ModuleList()
        self.block_to_control: dict[int, int] = {}
        for block_idx in range(int(arch.num_layers)):
            if block_idx % self.layer_stride:
                continue
            self.block_to_control[block_idx] = len(self.control_blocks)
            self.control_blocks.append(
                MiniMaxH3CameraControlBlock(
                    dim=self.control_dim,
                    ffn_dim=int(control_ffn_dim),
                    num_heads=self.control_num_heads,
                    head_dim=self.control_head_dim,
                    temb_dim=int(control_temb_dim),
                    out_dim=int(arch.hidden_size),
                    norm_eps=float(arch.norm_eps),
                    qk_norm_eps=float(arch.qk_norm_eps),
                    supported_attention_backends=supported_attention_backends,
                    prefix=f"camera_controlnet.control_blocks.{len(self.control_blocks)}",
                ))

    def _validate_rope_width(self, rope_freq_dim: int) -> None:
        """The control head has to be wide enough for the backbone's rotary table.

        The table rotates ``2 * 3 * rope_freq_dim`` channels per head. Reusing it is what keeps the
        two streams on one coordinate system; a narrower control head would need its own table at a
        reduced ``rope_freq_dim``, which is a different notion of position.
        """
        required = 2 * 3 * rope_freq_dim
        if self.control_head_dim < required:
            raise ValueError(
                f"camera_control_dim / camera_control_num_heads = {self.control_head_dim} is below the {required} "
                f"channels MiniMax-H3's rotary embedding rotates per head. Raise camera_control_dim or lower "
                f"camera_control_num_heads so the ratio is at least {required} (e.g. dim 1024 with 8 heads).")

    def _validate_sp_head_split(self) -> None:
        """The Ulysses all-to-all scatters heads across the SP group, so the split has to be exact.

        The trunk picks its head count independently of the backbone's, so a world size that divides
        the backbone's 56 heads can still leave the trunk unable to split its own. Without this the
        failure surfaces as an opaque NCCL abort at the first forward.
        """
        sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if self.control_num_heads % sp_world_size:
            valid = [n for n in range(1, self.control_num_heads + 1) if not self.control_num_heads % n]
            raise ValueError(
                f"camera_control_num_heads={self.control_num_heads} must be divisible by the sequence parallel "
                f"size ({sp_world_size}); the ControlNet shards its heads across the SP group just like the "
                f"backbone. Valid sequence parallel sizes for this head count: {valid}.")

    def embed(self, control_latents: dict[str, torch.Tensor | None]) -> torch.Tensor | None:
        """Embed and fuse the control rows.

        Returns ``None`` when any enabled modality is missing, which is how an unconditional forward
        pass opts out of the branch entirely.
        """
        embeds: list[torch.Tensor] = []
        for name in self.enabled_modalities:
            value = control_latents.get(name)
            if value is None:
                return None
            projection = self.embeddings[name]
            expected = self._modality_widths[name]
            if value.ndim != 3 or value.shape[-1] != expected:
                raise ValueError(f"The {name} control rows must have shape [batch, rows, {expected}], got "
                                 f"{tuple(value.shape)}.")
            embedded, _ = projection(value.to(projection.weight.dtype))
            embeds.append(embedded)
        fused = embeds[0] if len(embeds) == 1 else torch.cat(embeds, dim=-1)
        if self.fusion_proj is not None:
            fused, _ = self.fusion_proj(fused)
        return fused

    def scatter(
        self,
        control_latents: dict[str, torch.Tensor | None],
        *,
        target_row_indices: torch.Tensor | None,
        row_start: int,
        row_stop: int,
        local_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Place the control rows into this rank's slice of the packed layout.

        Returns the control stream ``(batch, local_len, control_dim)`` and the residual mask
        ``(1, local_len, 1)``. Both are returned even when this rank owns no target video row: the
        control blocks run collectives, so every rank has to walk the same number of them.
        """
        fused = self.embed(control_latents)
        if fused is None or target_row_indices is None:
            return None, None
        if fused.shape[1] != target_row_indices.numel():
            raise ValueError(f"The control trunk was given {fused.shape[1]} rows for "
                             f"{target_row_indices.numel()} target video rows.")

        states = fused.new_zeros((batch_size, local_len, self.control_dim))
        mask = torch.zeros((1, local_len, 1), device=device, dtype=dtype)
        local_rows, source_rows = local_shard_rows(target_row_indices, row_start, row_stop)
        if local_rows.numel():
            states = states.index_copy(1, local_rows, fused.index_select(1, source_rows))
            mask = mask.index_fill(1, local_rows, 1.0)
        return states.to(dtype), mask

    def project_temb(self, temb: torch.Tensor) -> torch.Tensor:
        """Bottleneck the shared timestep embedding once for the whole trunk."""
        projected, _ = self.temb_proj(F.silu(temb).to(self.temb_proj.weight.dtype))
        return projected

    def forward_block(
        self,
        block_idx: int,
        control_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance the control stream past backbone block ``block_idx``.

        Returns the stream unchanged and no residual for the blocks ``layer_stride`` skips.
        """
        control_idx = self.block_to_control.get(block_idx)
        if control_idx is None:
            return control_states, None
        return self.control_blocks[control_idx](control_states, temb, timestep_indices, rotary_emb, original_seq_len)


class MiniMaxH3CameraTransformer3DModel(MiniMaxH3Transformer3DModel):
    """MiniMax-H3 with a camera ControlNet as a child module.

    The ControlNet is built during construction rather than attached afterwards, so FastVideo's
    loader and FSDP see its parameters on the normal path. They are absent from the released
    checkpoint; the loader initialises them by matching ``camera_controlnet`` in the parameter name.
    """

    def __init__(self, config: MiniMaxH3Config, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config

        self.enable_camera_controlnet = bool(getattr(arch, "camera_enable_controlnet", False))
        self.freeze_backbone_for_camera = bool(getattr(arch, "camera_freeze_backbone", True))

        enabled = ["camera"]
        if bool(getattr(arch, "camera_enable_depth", False)):
            enabled.append("depth")

        self.camera_controlnet: MiniMaxH3CameraControlNet | None = None
        if not self.enable_camera_controlnet:
            return

        self.camera_controlnet = MiniMaxH3CameraControlNet(
            arch,
            enabled_modalities=enabled,
            layer_stride=int(getattr(arch, "camera_control_layer_stride", 2)),
            control_dim=int(getattr(arch, "camera_control_dim", 1024)),
            control_ffn_dim=int(getattr(arch, "camera_control_ffn_dim", 4096)),
            control_num_heads=int(getattr(arch, "camera_control_num_heads", 8)),
            control_temb_dim=int(getattr(arch, "camera_control_temb_dim", 256)),
            temb_in_dim=self.adaln_rank or int(arch.time_embed_dim),
            # The control trunk attends over the same packed layout as the backbone, but the sparse
            # H3 kernel is tuned for the backbone's head geometry and carries a gate weight the
            # loader would have to initialise separately. Dense attention over a 1024-wide trunk is
            # a small fraction of the step, so the trunk stays dense.
            supported_attention_backends=tuple(backend for backend in self._supported_attention_backends
                                               if backend != AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3),
        )
        if self.freeze_backbone_for_camera:
            for name, param in self.named_parameters():
                param.requires_grad_(name.startswith("camera_controlnet."))

    def _run_blocks(
        self,
        packed_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        local_timestep_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Interleave the control trunk with the backbone trunk."""
        camera_latent = kwargs.pop("camera_latent", None)
        depth_latent = kwargs.pop("depth_latent", None)
        target_row_indices = kwargs.pop("camera_row_indices", None)
        if kwargs:
            raise TypeError(f"{type(self).__name__} received unsupported forward arguments {sorted(kwargs)}.")

        control_states: torch.Tensor | None = None
        control_mask: torch.Tensor | None = None
        control_temb: torch.Tensor | None = None
        if self.camera_controlnet is not None:
            sp_world_size = get_sp_world_size() if model_parallel_is_initialized() else 1
            local_len = int(packed_hidden_states.shape[1])
            row_start = get_sp_parallel_rank() * local_len if sp_world_size > 1 else 0
            control_states, control_mask = self.camera_controlnet.scatter(
                {
                    "camera": camera_latent,
                    "depth": depth_latent
                },
                target_row_indices=target_row_indices,
                row_start=row_start,
                row_stop=row_start + local_len,
                local_len=local_len,
                batch_size=int(packed_hidden_states.shape[0]),
                dtype=packed_hidden_states.dtype,
                device=packed_hidden_states.device,
            )
            if control_states is not None:
                control_temb = self.camera_controlnet.project_temb(temb)

        for block_idx, block in enumerate(self.transformer_blocks):
            packed_hidden_states = block(
                packed_hidden_states,
                temb,
                adaln_indices,
                rotary_emb,
                original_seq_len,
            )
            if control_states is None:
                continue
            assert self.camera_controlnet is not None and control_temb is not None
            control_states, residual = self.camera_controlnet.forward_block(
                block_idx,
                control_states,
                control_temb,
                local_timestep_indices,
                rotary_emb,
                original_seq_len,
            )
            if residual is not None:
                packed_hidden_states = packed_hidden_states + residual.to(packed_hidden_states.dtype) * control_mask
        return packed_hidden_states


EntryClass = MiniMaxH3CameraTransformer3DModel
