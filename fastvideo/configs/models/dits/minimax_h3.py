# SPDX-License-Identifier: Apache-2.0
"""Architecture configuration for the MiniMax H3 joint audio-video DiT."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_minimax_h3_block(name: str, module: object) -> bool:
    """Select the main, text-refiner, and control-trunk blocks for FSDP.

    The camera ControlNet's blocks are listed too: a 25-block trunk is a few hundred million
    parameters, and leaving them out of the shard plan replicates them — plus their optimizer
    state, the larger cost — on every rank. They are absent from a plain H3 model, so the extra
    pattern matches nothing there.
    """
    del module
    parts = name.split(".")
    return ((len(parts) == 2 and parts[0] == "transformer_blocks" and parts[1].isdigit())
            or (len(parts) == 3 and parts[:2] == ["token_refiner", "refiner_blocks"] and parts[2].isdigit())
            or (len(parts) == 3 and parts[:2] == ["camera_controlnet", "control_blocks"] and parts[2].isdigit()))


@dataclass
class MiniMaxH3ArchConfig(DiTArchConfig):
    """One-to-one representation of the released transformer config."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_minimax_h3_block])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FLASH_ATTN,
        # FP4-quantized QK attention (fa4_fp4 on sm_100/sm_103, cutlass on
        # sm_12x). Enabled for speed experiments; output quality against the
        # SSIM references is not yet validated.
        AttentionBackendEnum.ATTN_QAT_INFER,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
    )

    # LoRA targets are matched by substring against the full parameter path, so the leaf names in
    # the 50 main blocks (`attn.to_*`, `ff.fc_*`) also occur in places that must not be adapted:
    # the two text-refiner blocks, the timestep MLP (also named fc_in/fc_out after
    # `param_names_mapping`), and the from-scratch control trunk, whose zero-initialised `proj_out`
    # is the reason an untrained ControlNet is a no-op.
    exclude_lora_layers: list[str] = field(
        default_factory=lambda: ["token_refiner", "time_embedder", "camera_controlnet"])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^time_embedder\.linear_1\.(.*)$": r"time_embedder.fc_in.\1",
            r"^time_embedder\.linear_2\.(.*)$": r"time_embedder.fc_out.\1",
            r"^(.*)\.attn\.to_out\.0\.(.*)$": r"\1.attn.to_out.\2",
            r"^(.*)\.ff\.net\.0\.proj\.(.*)$": r"\1.ff.fc_in.\2",
            r"^(.*)\.ff\.net\.2\.(.*)$": r"\1.ff.fc_out.\2",
        })
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    adaln_rank: int | None = None
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    # Camera ControlNet. Absent from the released checkpoint: these describe a branch built on the
    # meta device and initialized by the loader, so leaving `camera_enable_controlnet` false keeps
    # the model bit-identical to the release.
    camera_enable_controlnet: bool = False
    camera_freeze_backbone: bool = True
    camera_enable_depth: bool = False
    camera_control_layer_stride: int = 2
    camera_control_dim: int = 1024
    camera_control_ffn_dim: int = 4096
    camera_control_num_heads: int = 8
    camera_control_temb_dim: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.patch_size) != 3:
            raise ValueError(f"MiniMax H3 patch_size must have three axes, got {self.patch_size}.")
        self.patch_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2])
        self.num_channels_latents = self.in_channels
        self.out_channels = self.in_channels
        if self.adaln_rank is not None and not 0 < self.adaln_rank <= self.time_embed_dim:
            raise ValueError(f"MiniMax H3 adaln_rank must be in (0, time_embed_dim={self.time_embed_dim}], "
                             f"got {self.adaln_rank}.")
        rotary_dim = 2 * 3 * self.rope_freq_dim
        if rotary_dim > self.attention_head_dim or rotary_dim % 2:
            raise ValueError(f"MiniMax H3 rotary width must be even and no larger than the head width; got "
                             f"rotary_dim={rotary_dim}, attention_head_dim={self.attention_head_dim}.")


@dataclass
class MiniMaxH3Config(DiTConfig):
    """FastVideo component configuration for MiniMax H3 transformers."""

    arch_config: MiniMaxH3ArchConfig = field(default_factory=MiniMaxH3ArchConfig)
    prefix: str = "minimax_h3"
    # FastVideo's Fully Sharded Data Parallel (FSDP) loading path requires one
    # parameter dtype, while H3 inference keeps boundary projections in FP32.
    uniform_parameter_dtype: bool = False
