# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 proxy-to-video training plugin.

Turns H3 into a renderer: in goes a cheap proxy render of a scene plus a camera trajectory, out
comes a photoreal video of the same scene under the same motion. Two conditioning routes, chosen to
match what each signal is:

**The proxy rides the Ref2VA reference slots.** It is content — the layout, occlusion and motion the
output must agree with — and H3 already knows how to read content it is shown: the released Ref2VA
checkpoint packs ordered references as prefix rows of the video stream, at their own resolution and
their own rotary coordinates, held at a near-clean timestep while the target denoises. Feeding the
proxy there needs no architectural change at all. A single RGB anchor frame goes in the slot ahead
of it to fix appearance, which the proxy by construction cannot supply.

**The camera rides a ControlNet.** A trajectory is not content; it is a per-token constraint, and it
has to bind tightly enough that the same proxy under two trajectories yields two different videos. A
reference in the prefix is read once for the whole clip and cannot do that. So the camera becomes a
dense Plücker ray field on the target latent grid and enters through
:class:`~fastvideo.models.dits.minimax_h3_camera_controlnet.MiniMaxH3CameraControlNet`, which adds a
zero-initialised residual at the exact token each ray belongs to.

The backbone is frozen by default and only the control trunk trains, so step 0 reproduces the
released model exactly and every scene-specific signal has to earn its way through the trunk.

Audio is not supervised by default. Game and engine footage is silent, the packed layout still
requires audio rows, and returning a video-only prediction is what tells
:func:`~fastvideo.train.methods.fine_tuning.finetune._compute_finetune_loss_map` to leave the audio
branch alone rather than train it towards zero-latent silence.
"""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.camera import build_camera_latent
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3PackedLayout,
    audio_latent_num_frames,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.base import NoisePrediction
from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)

_VIDEO_LATENT_CHANNELS = 24
_AUDIO_LATENT_CHANNELS = 32


def _pad_to_patch(latents: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """Pad a latent grid's height and width up to the patch size.

    A reference is encoded at whatever resolution its render happened to have, and 16x VAE
    downsampling readily lands on an odd latent extent that the 2x2 patch cannot tile. Zero padding
    on the far edge adds at most one row and one column of tokens; cropping instead would silently
    drop a strip of the scene, and resizing would break pixel alignment with the target.
    """
    pad_h = (-latents.shape[-2]) % patch_h
    pad_w = (-latents.shape[-1]) % patch_w
    if not pad_h and not pad_w:
        return latents
    return torch.nn.functional.pad(latents, (0, pad_w, 0, pad_h), value=0.0)


class MiniMaxH3ProxyModel(MiniMaxH3Model):
    """H3 Ref2VA fine-tuning on cached proxy/target pairs, with a camera ControlNet."""

    _transformer_cls_name = "MiniMaxH3CameraTransformer3DModel"
    # Start from the Ref2VA partition, not the T2VA one. The whole reason the proxy needs no
    # architectural change is that this checkpoint already reads ordered references as prefix rows;
    # `transformer` has never seen one and would have to learn in-context conditioning from
    # scratch. A snapshot without `transformer_ref/` fails loudly here rather than quietly training
    # the wrong partition.
    _transformer_module_type = "transformer_ref"

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
        attention_backend: AttentionBackendEnum | str | None = AttentionBackendEnum.TORCH_SDPA,
        # --- camera control branch ---
        enable_camera_controlnet: bool = True,
        enable_control_depth: bool = False,
        controlnet_dim: int = 1024,
        controlnet_ffn_dim: int = 4096,
        controlnet_num_heads: int = 8,
        controlnet_layer_stride: int = 2,
        controlnet_temb_dim: int = 256,
        freeze_backbone: bool = True,
        camera_dropout: float = 0.1,
        # --- reference conditioning ---
        enable_anchor: bool = True,
        supervise_audio: bool = False,
        lora: Any = None,
    ) -> None:
        self._enable_camera_controlnet = bool(enable_camera_controlnet)
        self._enable_control_depth = bool(enable_control_depth)
        self._controlnet_dim = int(controlnet_dim)
        self._controlnet_ffn_dim = int(controlnet_ffn_dim)
        self._controlnet_num_heads = int(controlnet_num_heads)
        self._controlnet_layer_stride = int(controlnet_layer_stride)
        self._controlnet_temb_dim = int(controlnet_temb_dim)
        self._freeze_backbone = bool(freeze_backbone)
        self._camera_dropout = float(camera_dropout)
        self._enable_anchor = bool(enable_anchor)
        self._supervise_audio = bool(supervise_audio)

        if self._enable_control_depth and not self._enable_camera_controlnet:
            raise ValueError("enable_control_depth=true requires enable_camera_controlnet=true; depth is a second "
                             "modality on the camera trunk, not a branch of its own.")
        if not 0.0 <= self._camera_dropout < 1.0:
            raise ValueError(f"camera_dropout must be in [0, 1), got {camera_dropout!r}")
        if self._camera_dropout and self._enable_control_depth:
            raise ValueError("camera_dropout drops the whole control trunk, which would also drop depth. Set "
                             "camera_dropout=0.0 when enable_control_depth=true.")

        from fastvideo.train.utils.lora import LoraConfig
        lora_config = LoraConfig.coerce(lora)
        if lora_config is not None and lora_config.enable and self._freeze_backbone:
            raise ValueError("freeze_backbone=true and lora.enable=true ask for opposite things: LoRA *is* how this "
                             "plugin trains the backbone. Set freeze_backbone=false to train adapters (plus the "
                             "control trunk, if enabled), or drop the lora block to train only the trunk.")

        self._stamp_control_config(training_config)
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
            transformer_override_safetensor=transformer_override_safetensor,
            attention_backend=attention_backend,
            lora=lora,
        )

    # ------------------------------------------------------------------
    # Config plumbing
    # ------------------------------------------------------------------

    def _restore_trainable_after_lora(self, transformer: torch.nn.Module) -> None:
        """Keep the control trunk fully trainable alongside the backbone's adapters.

        The trunk has no pretrained weights to adapt — it is built on the meta device and
        initialized by the loader — so it trains as full parameters while the backbone trains as
        low-rank updates. ``camera_controlnet`` is in the arch config's ``exclude_lora_layers`` for
        the same reason, which is what leaves these parameters un-wrapped for this method to find.
        """
        if not self._enable_camera_controlnet:
            return
        restored = 0
        for name, param in transformer.named_parameters():
            if "camera_controlnet." in name:
                param.requires_grad_(True)
                restored += param.numel()
        if not restored:
            raise RuntimeError("enable_camera_controlnet=true but the loaded transformer exposes no "
                               "`camera_controlnet.*` parameters to train.")
        logger.info("Restored %.1fM control-trunk parameters as fully trainable alongside LoRA", restored / 1e6)

    def _validate_data_contract(self, training_config: TrainingConfig) -> None:
        """Keep the packed-layout requirements, drop the T2VA parquet one.

        This plugin reads a cached reference/control store rather than the joint parquet, so
        ``preprocessed_data_type`` describes nothing about the batches it will see.
        """
        if int(training_config.data.train_batch_size) != 1:
            raise ValueError(f"{type(self).__name__} requires training.data.train_batch_size=1")
        if float(training_config.data.training_cfg_rate) != 0.0:
            raise ValueError(f"{type(self).__name__} requires training.data.training_cfg_rate=0.0; drop the camera "
                             "branch with `camera_dropout` instead, which is the conditioning this model can "
                             "actually be asked to do without.")

    def _stamp_control_config(self, training_config: TrainingConfig) -> None:
        """Write ``camera_*`` fields onto the DiT config before the model is built.

        The released H3 config has no counterpart for these and the DiT reads them with ``getattr``
        defaults, so stamping is what turns the branch on without forking the upstream dataclass.
        """
        pipeline_config = getattr(training_config, "pipeline_config", None)
        dit_config = getattr(pipeline_config, "dit_config", None)
        arch_config = getattr(dit_config, "arch_config", None)
        if arch_config is None:
            raise ValueError("training_config.pipeline_config.dit_config.arch_config is required to configure the "
                             "camera ControlNet")
        arch_config.camera_enable_controlnet = self._enable_camera_controlnet
        arch_config.camera_enable_depth = self._enable_control_depth
        arch_config.camera_freeze_backbone = self._freeze_backbone
        arch_config.camera_control_dim = self._controlnet_dim
        arch_config.camera_control_ffn_dim = self._controlnet_ffn_dim
        arch_config.camera_control_num_heads = self._controlnet_num_heads
        arch_config.camera_control_layer_stride = self._controlnet_layer_stride
        arch_config.camera_control_temb_dim = self._controlnet_temb_dim

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        """Build the cached-sample dataloader; no VAE or text encoder is needed."""
        from fastvideo.distributed import get_sp_group, get_world_group
        from fastvideo.train.models.minimax_h3.proxy_dataset import build_proxy_train_dataloader

        self.sp_group = get_sp_group()
        world_group = get_world_group()
        sp_world_size = int(training_config.distributed.sp_size or 1)
        if world_group.world_size % sp_world_size:
            raise ValueError(f"world_size={world_group.world_size} must be divisible by sp_size={sp_world_size}")
        self.dataloader = build_proxy_train_dataloader(
            training_config.data,
            num_sp_groups=world_group.world_size // sp_world_size,
            sp_world_size=sp_world_size,
            global_rank=world_group.rank,
            include_anchor=self._enable_anchor,
            include_camera=self._enable_camera_controlnet,
            include_depth=self._enable_control_depth,
        )
        self.start_step = 0

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _reference_rows(
        self,
        raw_batch: dict[str, Any],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[list[MiniMaxH3PreparedReference], list[torch.Tensor]]:
        """Patchify the anchor and proxy latents into ordered Ref2VA condition rows.

        The returned descriptors carry geometry only. ``build_ref2va_packed_sequence`` reads nothing
        else from a prepared reference, so this avoids reconstructing pixel state the cache already
        encoded away.
        """
        patch_size = self.transformer.patch_size
        _, patch_h, patch_w = patch_size
        references: list[MiniMaxH3PreparedReference] = []
        rows: list[torch.Tensor] = []

        # Anchor first, proxy second. The order has to match the vision placeholders the text
        # embedding was built around: Qwen labels them `<Picture 1>` then `<Video 1>`, and the
        # layout walks references in list order.
        ordered: list[tuple[str, torch.Tensor]] = []
        if self._enable_anchor:
            ordered.append(("image", raw_batch["anchor_latent"]))
        ordered.append(("video", raw_batch["proxy_latent"]))

        for media_type, latents in ordered:
            latents = _pad_to_patch(latents.to(device=device, dtype=dtype), patch_h, patch_w)
            if latents.ndim != 5 or latents.shape[1] != _VIDEO_LATENT_CHANNELS:
                raise ValueError(f"A cached {media_type} reference must have shape "
                                 f"[1, {_VIDEO_LATENT_CHANNELS}, frames, height, width], got {tuple(latents.shape)}")
            if media_type == "image" and latents.shape[2] != 1:
                raise ValueError(f"The anchor reference must hold one latent frame, got {latents.shape[2]}")
            references.append(
                MiniMaxH3PreparedReference(
                    media_type=media_type,  # type: ignore[arg-type]
                    num_latent_frames=int(latents.shape[2]),
                    latent_height=int(latents.shape[3]),
                    latent_width=int(latents.shape[4]),
                ))
            rows.append(patchify_video_latents(latents, patch_size))
        return references, rows

    def _camera_rows(
        self,
        raw_batch: dict[str, Any],
        *,
        latent_shape: tuple[int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Build the ControlNet's patchified camera (and optional depth) rows."""
        if not self._enable_camera_controlnet:
            return {}
        # Every rank in a sequence-parallel group has to reach the same verdict: the control blocks
        # run collectives, so a split decision hangs the group rather than producing a wrong number.
        # `TrainingMethod` seeds one generator per SP group and this draw sits at a fixed point in
        # `prepare_batch`'s draw sequence, which is what keeps them in step. The `and` short-circuit
        # matters for the same reason: a disabled dropout must consume no draw at all.
        # Sample on the generator's own device -- a CUDA generator cannot drive a CPU draw.
        drop = self._camera_dropout and torch.rand(
            (), generator=generator, device=generator.device).item() < self._camera_dropout
        if drop:
            return {}

        num_latent_frames, latent_height, latent_width = latent_shape
        info = (raw_batch.get("info_list") or [{}])[0]
        pixel_size = info.get("pixel_size")
        if pixel_size is None:
            data_config = self.training_config.data
            pixel_size = (int(data_config.num_height), int(data_config.num_width))

        camera_latent = build_camera_latent(
            raw_batch["camera_extrinsics"][0].to(device=device, dtype=torch.float32),
            raw_batch["camera_intrinsics"][0].to(device=device, dtype=torch.float32),
            latent_height=latent_height,
            latent_width=latent_width,
            pixel_size=(int(pixel_size[0]), int(pixel_size[1])),
            num_latent_frames=num_latent_frames,
        )
        control: dict[str, torch.Tensor] = {
            "camera_latent": patchify_video_latents(camera_latent.to(dtype), self.transformer.patch_size)[None],
        }
        if self._enable_control_depth:
            depth = raw_batch["depth_latent"].to(device=device, dtype=dtype)
            control["depth_latent"] = patchify_video_latents(depth, self.transformer.patch_size)[None]
        return control

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        """Build a Ref2VA training document: reference rows, target noise, and control rows."""
        if latents_source != "data":
            raise NotImplementedError(f"{type(self).__name__} trains from cached proxy/target pairs and has no "
                                      f"latents_source={latents_source!r} path.")
        dtype = torch.bfloat16
        device = self.device
        patch_size = self.transformer.patch_size

        text_embedding = raw_batch["text_embedding"].to(device=device, dtype=dtype)
        text_token_tags = raw_batch["text_token_tags"][0].to(torch.long)
        if text_embedding.ndim != 3 or text_embedding.shape[0] != 1 or text_embedding.shape[-1] != 5120:
            raise ValueError(f"text_embedding must have shape [1, length, 5120], got {tuple(text_embedding.shape)}")
        if text_token_tags.shape[0] != text_embedding.shape[1]:
            raise ValueError("text_token_tags must have one tag per text embedding row")

        video_latents = raw_batch["vae_latent"].to(device=device, dtype=dtype)
        if video_latents.ndim != 5 or tuple(video_latents.shape[:2]) != (1, _VIDEO_LATENT_CHANNELS):
            raise ValueError(f"vae_latent must have shape [1, {_VIDEO_LATENT_CHANNELS}, frames, height, width], "
                             f"got {tuple(video_latents.shape)}")
        _, _, num_latent_frames, latent_height, latent_width = video_latents.shape

        data_config = self.training_config.data
        num_audio_latents = audio_latent_num_frames(int(data_config.num_frames))
        audio_latents = raw_batch.get("audio_latent")
        if audio_latents is None:
            # The packed layout requires audio rows even for silent footage. Zeros are the audio
            # VAE's own encoding of silence closely enough for rows that carry no loss.
            audio_latents = torch.zeros(
                (1, MINIMAX_H3_AUDIO_CHANNELS, _AUDIO_LATENT_CHANNELS, num_audio_latents),
                device=device,
                dtype=dtype,
            )
        else:
            audio_latents = audio_latents.to(device=device, dtype=dtype)[..., :num_audio_latents]

        references, reference_rows = self._reference_rows(raw_batch, dtype=dtype, device=device)
        layout = build_ref2va_packed_sequence(
            text_token_tags,
            references,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
        )

        video_noise = torch.randn(video_latents.shape, generator=generator, device=device, dtype=dtype)
        audio_noise = torch.randn(audio_latents.shape, generator=generator, device=device, dtype=dtype)
        video_noise_amount, audio_noise_amount = self._sample_noise_amounts(generator, device)
        video_sigmas = video_noise_amount.to(dtype).view(1, 1, 1, 1, 1)
        audio_sigmas = audio_noise_amount.to(dtype).view(1, 1, 1, 1)

        # References are conditions, not targets: they are held at the same near-clean amount
        # inference uses so the model sees the identical signal it will be given at sampling time.
        condition_rows = torch.cat(reference_rows)
        condition_noise = torch.randn(condition_rows.shape, generator=generator, device=device, dtype=dtype)
        condition_rows = (MINIMAX_H3_KEYFRAME_NOISE_AUG * condition_rows +
                          (1.0 - MINIMAX_H3_KEYFRAME_NOISE_AUG) * condition_noise)

        control = self._camera_rows(
            raw_batch,
            latent_shape=(num_latent_frames, latent_height, latent_width),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        control["condition_video_rows"] = condition_rows

        training_batch = TrainingBatch()
        training_batch.latents = video_latents.permute(0, 2, 1, 3, 4)
        training_batch.audio_latents = audio_latents
        training_batch.encoder_hidden_states = text_embedding
        training_batch.encoder_attention_mask = torch.ones(text_embedding.shape[:2], device=device, dtype=dtype)
        training_batch.infos = raw_batch.get("info_list")
        training_batch.raw_latent_shape = tuple(video_latents.shape)
        training_batch.noisy_model_input = (1.0 - video_sigmas) * video_latents + video_sigmas * video_noise
        training_batch.audio_noisy_model_input = (1.0 - audio_sigmas) * audio_latents + audio_sigmas * audio_noise
        training_batch.noise = video_noise
        training_batch.audio_noise = audio_noise
        training_batch.sigmas = video_sigmas
        training_batch.audio_sigmas = audio_sigmas
        training_batch.timesteps = 1.0 - video_noise_amount
        training_batch.audio_timesteps = 1.0 - audio_noise_amount
        training_batch.minimax_h3_layout = layout
        training_batch.minimax_h3_control = control
        training_batch.attn_metadata = None
        training_batch.attn_metadata_vsa = None
        return training_batch

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> NoisePrediction:
        """Run one Ref2VA document and return the target rows' flow prediction."""
        del timestep
        if not conditional or cfg_uncond is not None:
            raise ValueError(f"{type(self).__name__} predicts one conditional sample")
        if attn_kind != "dense":
            raise ValueError(f"{type(self).__name__} supports dense attention for training")
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise RuntimeError("prepare_batch() must set TrainingBatch.minimax_h3_layout")
        control = batch.minimax_h3_control
        if not control or "condition_video_rows" not in control:
            raise RuntimeError("prepare_batch() must set TrainingBatch.minimax_h3_control")
        if batch.audio_noisy_model_input is None or batch.encoder_hidden_states is None:
            raise RuntimeError("prepare_batch() must set audio and text transformer inputs")
        if batch.timesteps is None or batch.audio_timesteps is None:
            raise RuntimeError("prepare_batch() must set video and audio timesteps")

        dtype = torch.bfloat16
        device = self.device
        video_bcthw = noisy_latents.permute(0, 2, 1, 3, 4).to(dtype)
        target_rows = patchify_video_latents(video_bcthw, self.transformer.patch_size)
        # Reference rows come first, in reference order: the layout's `video_indices` scatters this
        # tensor row-for-row into the packed sequence, and its condition prefix is what
        # `build_row_timesteps` holds at the near-clean amount.
        video_rows = torch.cat((control["condition_video_rows"], target_rows))

        audio_latents = batch.audio_noisy_model_input.to(dtype)
        num_audio_latents = audio_latents.shape[-1]
        audio_rows = audio_latents.permute(0, 1, 3, 2).reshape(-1, _AUDIO_LATENT_CHANNELS)

        video_timestep = float(batch.timesteps[0])
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_timestep=video_timestep,
            audio_timestep=float(batch.audio_timesteps[0]),
            condition_video_timestep=max(video_timestep, MINIMAX_H3_KEYFRAME_NOISE_AUG),
            condition_audio_timestep=1.0,
        )
        unique_timesteps = unique_timesteps.to(device)
        timestep_indices = timestep_indices.to(device)
        video_indices = layout.video_indices.to(device)

        control_kwargs: dict[str, torch.Tensor] = {}
        if "camera_latent" in control:
            control_kwargs["camera_latent"] = control["camera_latent"]
            control_kwargs["camera_row_indices"] = video_indices[layout.num_condition_video_rows:]
            if "depth_latent" in control:
                control_kwargs["depth_latent"] = control["depth_latent"]

        with torch.autocast(device.type, dtype=dtype), set_forward_context(
                current_timestep=unique_timesteps,
                attn_metadata=None,
        ):
            video_velocity, audio_velocity = self.transformer(
                hidden_states=video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=batch.encoder_hidden_states,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=layout.token_tags.to(device),
                position_ids=layout.position_ids.to(device),
                video_indices=video_indices,
                audio_indices=layout.audio_indices.to(device),
                text_indices=layout.text_indices.to(device),
                **control_kwargs,
            )

        _, _, num_video_latents, latent_height, latent_width = video_bcthw.shape
        video_prediction = unpatchify_video_tokens(
            video_velocity[:, layout.num_condition_video_rows:],
            num_video_latents,
            latent_height,
            latent_width,
            _VIDEO_LATENT_CHANNELS,
            self.transformer.patch_size,
        ).permute(0, 2, 1, 3, 4)
        if not self._supervise_audio:
            # A bare tensor selects the video-only branch of the finetune loss, which leaves the
            # audio head untouched rather than training it against placeholder silence.
            return -video_prediction
        audio_prediction = unpack_audio_tokens(audio_velocity[0], num_audio_latents)[None]
        return -video_prediction, -audio_prediction


__all__ = ["MiniMaxH3ProxyModel"]
