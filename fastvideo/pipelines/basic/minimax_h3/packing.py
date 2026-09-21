# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MiniMax and HuggingFace Teams.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from fastvideo.logger import init_logger

if TYPE_CHECKING:
    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference

logger = init_logger(__name__)

MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

MINIMAX_H3_FPS = 24
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4
MINIMAX_H3_MIN_DURATION = 5.0
MINIMAX_H3_MAX_DURATION = 15.0
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5
MINIMAX_H3_TEXT_ENCODER_LAYER = 50
MINIMAX_H3_VISION_START_TOKEN = "<|vision_start|>"
MINIMAX_H3_IMAGE_PAD_TOKEN = "<|image_pad|>"
MINIMAX_H3_VIDEO_PAD_TOKEN = "<|video_pad|>"
MINIMAX_H3_VISION_END_TOKEN = "<|vision_end|>"
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999
MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42

MINIMAX_H3_ROPE_FRAME_RESCALE = 5.0 / 3.0
MINIMAX_H3_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass(frozen=True)
class MiniMaxH3PackedLayout:
    """One packed joint sequence and the geometry needed to interpret it."""

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_video_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}.")

    if ratio >= 1:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)
    else:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio
    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area)**0.5
        width, height = width * scale, height * scale
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def align_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def prepare_keyframe_image(image: Image.Image, height: int, width: int, stretch: bool) -> Image.Image:
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (max(width, round(image.size[0] * scale)), max(height, round(image.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def pad_video_latents_to_patch(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Pad a latent grid's height and width up to the spatial patch.

    A reference is encoded at whatever resolution its render happened to have, and 16x VAE
    downsampling readily lands on an odd latent extent that the 2x2 patch cannot tile. A 336x192
    proxy is the case that showed up first: 21 x 12 latents. Zero padding on the far edge adds at
    most one row and one column of tokens; cropping would silently drop a strip of the scene, and
    resizing would break pixel alignment with the target. Training and validation have to do this
    the same way or the token counts diverge.
    """
    if latents.ndim != 5:
        raise ValueError(f"Video latents must be [B, C, T, H, W], got shape {tuple(latents.shape)}.")
    _, patch_h, patch_w = patch_size
    pad_h = (-latents.shape[-2]) % patch_h
    pad_w = (-latents.shape[-1]) % patch_w
    if not pad_h and not pad_w:
        return latents
    return torch.nn.functional.pad(latents, (0, pad_w, 0, pad_h), value=0.0)


def replicate_latents_to_grid(latents: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
    """Replicate a latent grid up onto a larger one, for a control signal that acts on target rows.

    The control trunk adds its residual to a target video row, so it needs one row for every target
    row -- and a proxy encoded at its own reference resolution has a small fraction of them. A
    336x192 proxy is 12 x 21 latents against a 768x1344 target's 48 x 84.

    Replication closes that gap exactly rather than approximately. The VAE's stride of 16 means
    proxy latent cell ``(i, j)`` covers the same pixels as target latent cells
    ``(4i..4i+3, 4j..4j+3)``, so copying each cell over that block puts it precisely where the pixels
    it describes ended up. A grid that does not divide is an error: interpolating would spread one
    proxy cell over a non-integer number of target cells, and the registration would then vary
    across the frame.

    Replicating latents rather than re-encoding replicated pixels is deliberate. The two are not the
    same tensor, because the VAE is not linear -- but they differ only in that encoding blown-up
    pixels also encodes the block edges the blow-up introduced, which is an artifact rather than
    scene detail. Neither route adds detail the reference did not have; what either adds is a
    position at which the signal is allowed to act. And this tensor is read by a freshly initialised
    linear and never decoded, so there is no reason for it to look like something the VAE would have
    produced -- which is what makes the cheaper route also the cleaner one.
    """
    if latents.ndim != 5:
        raise ValueError(f"Video latents must be [B, C, T, H, W], got shape {tuple(latents.shape)}.")
    source_height, source_width = int(latents.shape[-2]), int(latents.shape[-1])
    if latent_height % source_height or latent_width % source_width:
        raise ValueError(f"the target latent grid {latent_height}x{latent_width} is not an integer multiple of the "
                         f"proxy's {source_height}x{source_width}, so the proxy cannot be replicated onto it with "
                         "uniform registration. Encode both at geometries whose latent grids divide -- 768x1344 over "
                         "192x336 is 4x, while 704x1280 over the same proxy is 3.67x by 3.81x.")
    return latents.repeat_interleave(latent_height // source_height,
                                     dim=-2).repeat_interleave(latent_width // source_width, dim=-1)


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    latents = pad_video_latents_to_patch(latents, patch_size)
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    # Channel-major patch features are a checkpoint contract; do not use einops's common patch-major order.
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    rows = rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return rows.permute(0, 2, 1).contiguous()


def spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # NumPy endpoint=False and float64 are part of the reference RoPE coordinate arithmetic.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            MINIMAX_H3_ROPE_FRAME_RESCALE *
            MINIMAX_H3_ROPE_FRAMES_PER_LATENT[index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span(num_latent_frames: int) -> float:
    # Preserve NumPy's pairwise summation: sequential torch summation diverges in the final ulp for longer clips.
    spans = np.ones(num_latent_frames, dtype=np.float64) * MINIMAX_H3_ROPE_FRAME_RESCALE
    for index, frames in enumerate(MINIMAX_H3_ROPE_FRAMES_PER_LATENT):
        spans[index::len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)] *= frames
    return float(spans.sum())


def build_packed_sequence(
        text_token_tags: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple[int, int, int],
        keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedLayout:
    if text_token_tags.ndim != 1:
        raise ValueError(f"text_token_tags must be one-dimensional, got {tuple(text_token_tags.shape)}.")
    valid_text_tags = (text_token_tags == MINIMAX_H3_TEXT_TAG) | (text_token_tags == MINIMAX_H3_VIDEO_TAG)
    if not bool(valid_text_tags.all()):
        raise ValueError("text_token_tags may contain only text and vision tags; semantic padding is not allowed.")

    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = (float(num_text_tokens) + _temporal_position_span(num_latent_frames) -
                           MINIMAX_H3_ROPE_FRAME_RESCALE)
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, float(num_text_tokens),
                          width_grid)

    video_positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    return MiniMaxH3PackedLayout(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
        num_video_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
    )


def _reference_temporal_span(num_latent_frames: int) -> float:
    """Advance the Ref2VA clock using its required sequential summation order."""
    return sum(MINIMAX_H3_ROPE_FRAME_RESCALE *
               MINIMAX_H3_ROPE_FRAMES_PER_LATENT[index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)]
               for index in range(num_latent_frames))


def _reference_clock_advance(reference: MiniMaxH3PreparedReference) -> float:
    """How far this reference moves the Ref2VA clock for whatever follows it.

    Separate from the fill loop because the target's rotary origin has to be known *before* the
    references are written when any of them is time-aligned, and computing it a second way would be
    a second definition of the layout.
    """
    if reference.media_type == "image":
        return 1.0
    if reference.media_type == "audio":
        return float(reference.num_audio_latents)
    return max(
        float(reference.num_audio_latents if reference.has_audio else 0),
        _reference_temporal_span(reference.num_latent_frames),
    )


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def strided_sample_indices(n_target: int, n_reference: int) -> torch.Tensor:
    """Token indices of a sparse sampling of ``n_target`` by ``n_reference`` points.

    A 4x latent downsample is 24 vs 6 height tokens and, after the 21-to-22 pad, 42 vs 11 width
    tokens. ``round(n_target / n_reference)`` is 4 in both axes, so the proxy lands on every 4th
    target token -- the VAE's own 4x cell correspondence -- rather than a 3.818 non-integer stride
    that attention has to resolve from content and noise. Clamped so a pad column cannot walk off
    the target's last token.
    """
    if n_target <= 0 or n_reference <= 0:
        raise ValueError(f"strided sample needs positive token counts, got {n_target} / {n_reference}.")
    if n_reference == 1:
        return torch.zeros(1, dtype=torch.int64)
    stride = max(1, int(round(n_target / n_reference)))
    indices = torch.arange(n_reference, dtype=torch.int64) * stride
    last = int(indices[-1])
    if last > n_target - 1:
        logger.warning(
            "time-aligned spatial stride %d over %d reference tokens overshoots a %d-token target axis; "
            "clamping the last %d tokens onto the far edge.",
            stride,
            n_reference,
            n_target,
            int((indices > n_target - 1).sum()),
        )
    return torch.clamp(indices, max=n_target - 1)


def time_aligned_frame_grid(
    reference_latent_height: int,
    reference_latent_width: int,
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A time-aligned reference occupies a strided sample of the *target's* spatial grid.

    ``spatial_position_grid`` normalises each tensor by its own ``sqrt(area)``, so a padded proxy
    (12 x 21 latents -> 12 x 22) does not span the same interval as a 48 x 84 target even though the
    *unpadded* aspect ratios match. Copying the target's coordinates at an integer stride puts
    proxy token (i, j) on the same rotary (h, w) as target token (stride*i, stride*j), which is the
    sparse sampling a 4x-smaller render actually is. The unaligned path still uses the reference's
    own grid: Ref2VA references are exemplars, not a downsample of the target.
    """
    target_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    ref_h = reference_latent_height // patch_h
    ref_w = reference_latent_width // patch_w
    tgt_h = latent_height // patch_h
    tgt_w = latent_width // patch_w
    if ref_h <= 0 or ref_w <= 0:
        raise ValueError(f"time-aligned reference token grid is empty: {reference_latent_height}x"
                         f"{reference_latent_width} with patch {(patch_h, patch_w)}.")
    h_idx = strided_sample_indices(tgt_h, ref_h)
    w_idx = strided_sample_indices(tgt_w, ref_w)
    sampled = target_grid.view(tgt_h, tgt_w, 2).index_select(0, h_idx).index_select(1, w_idx)
    return sampled.reshape(-1, 2).contiguous(), target_width_grid.index_select(0, w_idx)


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
) -> None:
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat([
        torch.full((num_audio_latents, ), float(width_grid[0]), dtype=torch.float64),
        torch.full((num_audio_latents, ), float(width_grid[-1]), dtype=torch.float64),
    ])


def _num_video_rows(reference: MiniMaxH3PreparedReference, patch_size: tuple[int, int, int]) -> int:
    patch_t, patch_h, patch_w = patch_size
    geometry = (reference.num_latent_frames, reference.latent_height, reference.latent_width)
    if any(value <= 0 for value in geometry):
        raise ValueError(f"Incomplete visual reference geometry: {geometry}.")
    if any(value % patch for value, patch in zip(geometry, patch_size, strict=True)):
        raise ValueError(f"Visual reference geometry {geometry} is not divisible by patch {patch_size}.")
    return (reference.num_latent_frames // patch_t) * (reference.latent_height // patch_h) * (reference.latent_width //
                                                                                              patch_w)


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> MiniMaxH3PackedLayout:
    """Build `[text | ordered references | target audio | target video]`."""
    if patch_size != (1, 2, 2):
        raise ValueError(f"MiniMax-H3 Ref2VA requires patch_size=(1, 2, 2), got {patch_size}.")
    if text_token_tags.ndim != 1:
        raise ValueError(f"text_token_tags must be one-dimensional, got {tuple(text_token_tags.shape)}.")
    valid_text_tags = (text_token_tags == MINIMAX_H3_TEXT_TAG) | (text_token_tags == MINIMAX_H3_VIDEO_TAG)
    if not bool(valid_text_tags.all()):
        raise ValueError("text_token_tags may contain only text and vision tags.")
    if not references:
        raise ValueError("Ref2VA requires at least one prepared reference.")

    patch_t, patch_h, patch_w = patch_size
    target_geometry = (num_latent_frames, latent_height, latent_width)
    if any(value <= 0 for value in target_geometry) or num_audio_latents <= 0:
        raise ValueError("Ref2VA target latent geometry must be positive.")
    if any(value % patch for value, patch in zip(target_geometry, patch_size, strict=True)):
        raise ValueError(f"Target geometry {target_geometry} is not divisible by patch {patch_size}.")

    visual_row_counts = [
        0 if reference.media_type == "audio" else _num_video_rows(reference, patch_size) for reference in references
    ]
    for reference in references:
        if reference.media_type == "audio" and not reference.has_audio:
            raise ValueError("An audio reference must carry decoded waveform latents.")
        if reference.has_audio and reference.num_audio_latents <= 0:
            raise ValueError("An audio-bearing reference has no resolved audio latents.")

    for reference in references:
        if not reference.time_aligned:
            continue
        # The switch means one thing: this reference's frames sit at the target's own rotary times.
        # An image has no frames to align, and an audio reference's clock is its latent count, so
        # neither can express it -- and silently ignoring the flag would report an experiment that
        # did not run.
        if reference.media_type != "video":
            raise ValueError(f"time_aligned is only meaningful for a video reference, not "
                             f"{reference.media_type!r}: an image occupies a single rotary instant and an "
                             "audio reference is clocked by its latent count.")
        if reference.num_latent_frames != num_latent_frames:
            raise ValueError(
                f"a time-aligned reference must have the target's latent frame count: reference has "
                f"{reference.num_latent_frames}, target has {num_latent_frames}. A shorter one would land on "
                "a prefix of the target's timeline and leave the rest unconstrained, which is not the "
                "alignment this flag claims.")

    num_text_tokens = int(text_token_tags.shape[0])
    num_target_video_rows = (num_latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_reference_video_rows = sum(visual_row_counts)
    num_reference_audio_rows = sum(reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS for reference in references
                                   if reference.has_audio)
    sequence_length = (num_text_tokens + num_reference_video_rows + num_reference_audio_rows + num_target_audio_rows +
                       num_target_video_rows)

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    video_indices: list[torch.Tensor] = []
    audio_indices: list[torch.Tensor] = []
    cursor = num_text_tokens
    # Where the target's own clock starts, computed by running the same sequential accumulation the
    # fill loop below runs. Sequential rather than `sum(...)` because the clock's float error is part
    # of the reference coordinate arithmetic, and a time-aligned reference has to land on the
    # target's times exactly rather than to within an ulp.
    rotary_time = float(num_text_tokens)
    for reference in references:
        rotary_time += _reference_clock_advance(reference)
    target_rotary_time = rotary_time

    rotary_time = float(num_text_tokens)
    for reference, visual_row_count in zip(references, visual_row_counts, strict=True):
        if reference.media_type == "image":
            rows = slice(cursor, cursor + visual_row_count)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.media_type == "audio":
            count = reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
            rows = slice(cursor, cursor + count)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, reference.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference.num_audio_latents)
        elif reference.media_type == "video":
            audio_count = reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS if reference.has_audio else 0
            audio_rows = slice(cursor, cursor + audio_count)
            video_rows = slice(audio_rows.stop, audio_rows.stop + visual_row_count)
            cursor = video_rows.stop
            if audio_count:
                audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))

            # A time-aligned reference is the target at lower resolution, so it takes both the
            # target's rotary times *and* a strided sample of the target's spatial grid. The
            # unaligned path keeps the reference's own sqrt(area) grid: Ref2VA exemplars are not a
            # downsample of the target, and their sequential clock is what keeps them from colliding
            # with it. The clock still advances by the full span either way, so the target's own
            # positions are unchanged -- the only thing this moves is the proxy.
            if reference.time_aligned:
                frame_grid, width_grid = time_aligned_frame_grid(
                    reference.latent_height,
                    reference.latent_width,
                    latent_height,
                    latent_width,
                    patch_h,
                    patch_w,
                )
            else:
                frame_grid, width_grid = _frame_position_grid(
                    reference.latent_height,
                    reference.latent_width,
                    patch_h,
                    patch_w,
                )
            if audio_count:
                _fill_audio_positions(
                    position_ids,
                    audio_rows,
                    reference.num_audio_latents,
                    rotary_time,
                    width_grid,
                )
            frame_time = temporal_position_grid(reference.num_latent_frames,
                                                target_rotary_time if reference.time_aligned else rotary_time)
            rows_per_frame = frame_grid.shape[0]
            position_ids[video_rows, 0] = frame_time.repeat_interleave(rows_per_frame)
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames // patch_t, 1)
            rotary_time += _reference_clock_advance(reference)
        else:
            raise ValueError(f"Unsupported prepared reference type: {reference.media_type!r}.")

    # The prepass and the loop above are two statements of the same clock, and a time-aligned
    # reference is written from the first while the target is written from the second. Exact
    # equality rather than a tolerance: both run the identical sequence of additions, so any
    # difference means someone changed one and not the other, which would silently place the proxy
    # off the target's timeline -- the one thing this is all for.
    if rotary_time != target_rotary_time:
        raise AssertionError(f"Ref2VA clock disagrees with its prepass ({rotary_time} vs {target_rotary_time}): "
                             "_reference_clock_advance has drifted from the reference fill loop.")

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time,
                          target_width_grid)
    frame_time = temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames // patch_t, 1)

    video_indices.append(torch.arange(video_start, sequence_length))
    audio_indices.append(torch.arange(audio_start, video_start))
    packed_video_indices = torch.cat(video_indices)
    packed_audio_indices = torch.cat(audio_indices)
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[packed_audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[packed_video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedLayout(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=packed_video_indices,
        audio_indices=packed_audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
        num_video_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
    )


def target_rows_per_latent_frame(layout: MiniMaxH3PackedLayout, patch_size: tuple[int, int, int]) -> int:
    """How many packed rows one latent frame of the target video occupies."""
    _, patch_h, patch_w = patch_size
    if layout.latent_height % patch_h or layout.latent_width % patch_w:
        raise ValueError(f"Target latent grid {layout.latent_height}x{layout.latent_width} is not divisible by the "
                         f"spatial patch {(patch_h, patch_w)}.")
    return (layout.latent_height // patch_h) * (layout.latent_width // patch_w)


def build_row_timesteps(
    layout: MiniMaxH3PackedLayout,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
    num_fixed_video_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group the packed sequence's rows by the noise amount each one is held at.

    ``num_fixed_video_rows`` covers leading rows of the *target* video that are a given rather than
    something to predict -- the first latent frame locked to the appearance anchor. They are held at
    ``condition_video_timestep``, the same amount the reference prefix uses, because that is the
    modulation the backbone already associates with "this row is shown to you, not asked of you".
    Reusing it also means a locked frame adds no new unique timestep to the group.
    """
    if num_fixed_video_rows < 0:
        raise ValueError(f"num_fixed_video_rows must be non-negative, got {num_fixed_video_rows}.")
    num_target_video_rows = layout.video_indices.numel() - layout.num_condition_video_rows
    if num_fixed_video_rows > num_target_video_rows:
        raise ValueError(f"Cannot fix {num_fixed_video_rows} of {num_target_video_rows} target video rows.")
    row_timesteps = torch.full((layout.sequence_length, ), video_timestep, dtype=torch.float32)
    fixed_stop = layout.num_condition_video_rows + num_fixed_video_rows
    row_timesteps[layout.video_indices[:fixed_stop]] = condition_video_timestep
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows:]] = audio_timestep
    row_timesteps[layout.audio_indices[:layout.num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def keyframe_condition_noise(
    condition_latent_shapes: tuple[tuple[int, int, int], ...],
    patch_size: tuple[int, int, int],
    latent_channels: int,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rows = []
    for num_latent_frames, latent_height, latent_width in condition_latent_shapes:
        noise = randn_tensor(
            (1, latent_channels, num_latent_frames, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        rows.append(patchify_video_latents(noise, patch_size))
    return torch.cat(rows)
