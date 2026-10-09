# SPDX-License-Identifier: Apache-2.0
"""Encode proxy/target clip pairs into the cached ``.pt`` format for MiniMax-H3 training.

Reads a JSONL manifest and writes one ``.pt`` per clip holding the high-quality target's VAE latent,
the proxy render's VAE latent, an RGB anchor frame latent, the Qwen3-VL text embedding with its
per-token modality tags, and the camera trajectory. Training then loads neither a VAE nor a text
encoder.

Manifest, one JSON object per line::

    {"name": "seg_0001",
     "target": "hq/0001.mp4",
     "proxy": "proxy/0001.mp4",
     "proxy_duv": "duv/0001",
     "proxy_duv_video": "duv/0001.mp4",
     "anchor": "anchor/0001.png",
     "camera": "poses/0001.npz",
     "prompt": "a knight walks through a ruined cathedral"}

Exactly one of ``proxy``, ``proxy_duv`` and ``proxy_duv_video`` is required. Prefer either DUV form
when the renderer can emit depth, because a geometry channel constrains the output far more tightly
than a shaded render does.

``proxy``
    An ordinary RGB render, used as-is and resized to the proxy grid.
``proxy_duv``
    A directory of per-frame ``NNNNNN.depth.f32`` and ``NNNNNN.semantic_id.png`` pairs, packed into
    three channels by :mod:`fastvideo.pipelines.basic.minimax_h3.proxy` -- log depth over 0.3 m to
    256 m plus a two-channel class code.
``proxy_duv_video``
    A DUV that some upstream pipeline already composed into a lossless video. The frames reach the
    VAE unchanged, so the channel convention is whatever the producer used rather than this repo's;
    it only has to be the same one at sampling time. Nothing is resized -- see
    :func:`read_duv_video_clip`.

``--proxy-references`` decides what a ``proxy_duv`` directory becomes. The default ``duv`` is the
one packed reference above, stored as ``proxy_latent`` ``[24, T, h, w]``. ``depth semantic`` makes
two separate video references out of the same planes -- a grey depth video and a flat-colour class
video, each with all three channels to itself -- stored in that order as ``proxy_latents``
``[R, 24, T, h, w]`` with ``info["proxy_references"]`` naming them. Qwen sees them as ``<Video 1>``
and ``<Video 2>``, so the chat role has to describe two videos: ``--cwm-system w0_depth_semantic``.

``--fit center-crop`` reaches the grid without changing aspect: each stream is scaled to cover the
grid and the overflow is cut equally from both sides, so a 1280x720 target becomes 1280x704 by
dropping 8 rows top and bottom. Depth and class planes are cropped but not scaled by default.
``--code-resize nearest`` permits a smaller aligned reference: crop the planes to the target field
of view, nearest-neighbour resize them, then construct depth, semantic or DUV pixels. A pre-packed
DUV video is never resized. The anchor is cropped to the target's aspect before it is scaled to its
short edge. The default ``resize`` is the legacy behaviour.

``anchor`` defaults to the target's first frame. Supplying a separate one is what lets the anchor
carry an appearance the target clip never shows — a different art style, a reference photograph.

``camera`` is an ``.npz`` with ``extrinsics`` ``[F, 4, 4]`` world-to-camera and ``intrinsics``
``[F, 3, 3]`` in pixels of the *source* render, plus optional ``pixel_size`` ``[2]``. It is required
only when training the camera ControlNet.

Usage::

    python scripts/h3_proxy/prepare_data/encode_proxy_samples.py \\
        --manifest data/h3_proxy/manifest.jsonl \\
        --root data/h3_proxy/raw \\
        --output /data/raw/h3_proxy/train \\
        --model-path data/models/MiniMax-H3 \\
        --num-frames 124 --height 768 --width 1344 \\
        --cwm-system w0

    Omni cache from a 1280x720 native corpus -- target, depth and semantic all 1280x704::

    python scripts/h3_proxy/prepare_data/encode_proxy_samples.py \\
        --manifest ... --root ... --output ... --model-path ... \\
        --num-frames 124 --height 704 --width 1280 --proxy-height 704 --proxy-width 1280 \\
        --fit center-crop --proxy-references depth semantic --cwm-system w0_depth_semantic

    Mixed-single-reference omni cache. Unlabelled rows are deterministically balanced across the
    available modalities; an explicit ``proxy_modality`` field overrides the assignment::

        python scripts/h3_proxy/prepare_data/encode_proxy_samples.py \\
        --manifest ... --root ... --output ... --model-path ... \\
        --num-frames 124 --height 704 --width 1280 --proxy-height 176 --proxy-width 320 \\
        --fit center-crop --code-resize nearest \\
        --proxy-variants duv depth semantic style --cwm-system w0_omni

    Refresh only the Qwen rows (VAE latents stay) after changing the chat wrap::

    python scripts/h3_proxy/prepare_data/encode_proxy_samples.py \\
        --manifest ... --root ... --output ... --model-path ... --text-only --cwm-system w0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

REQUIRED_MEDIA_KEYS = ("target", "prompt")
# Exactly one of these names the proxy. Order is only for error messages.
PROXY_KEYS = ("proxy", "proxy_duv", "proxy_duv_video")
FIT_MODES = ("resize", "center-crop")
# Mirrors `proxy.PROXY_REFERENCE_KINDS` and `cwm_presentation.CWM_SYSTEM_ROLES`; spelled out so
# --help does not import fastvideo.
PROXY_REFERENCES = ("duv", "depth", "semantic")
PROXY_VARIANTS = (*PROXY_REFERENCES, "style")
CWM_SYSTEM_CHOICES = ("w0", "wn", "w0_depth_semantic", "w0_omni", "none")
# The video references each chat role names, in <Video N> order. Roles not listed describe the one
# proxy video of the CWM release.
ROLE_VIDEO_REFERENCES = {"w0_depth_semantic": ("depth", "semantic")}
LEGACY_REFERENCES = ("duv", )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".", help="Base directory for relative manifest paths.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True, help="MiniMax-H3 snapshot providing vae/ and text_encoder/.")
    parser.add_argument("--num-frames", type=int, default=124, help="Must satisfy num_frames %% 17 == 5.")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    # A proxy carries layout and motion, which survive downsampling; a quarter-resolution reference
    # costs ~1/16 the tokens of a full-resolution one. 336x192 is the released CWM geometry.
    parser.add_argument("--proxy-height", type=int, default=192)
    parser.add_argument("--proxy-width", type=int, default=336)
    parser.add_argument(
        "--qwen-video-fps",
        type=float,
        default=24.0,
        help="Frame rate presented to Qwen for <Video 1>. The proxy VAE always encodes all 24-fps frames. "
        "Use 2 only to reproduce a legacy text cache.",
    )
    # The released short edge, which CWM also uses. It costs ~7x the anchor tokens of a 768 canvas
    # -- both as Qwen vision tokens and as Ref2VA reference rows -- and buys detail the target
    # canvas cannot show. That trade only looks bad if the anchor is treated as a picture of the
    # first frame; it is the appearance dictionary for the whole take, and the run that cut it to
    # 768 could not make the proxy steer the camera.
    parser.add_argument("--anchor-short-edge", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0, help="This worker's index, for splitting a manifest.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cwm-system",
        default="w0",
        choices=CWM_SYSTEM_CHOICES,
        help="Wrap Qwen in CWM's AWM_PROXY_CONTROL chat. ABot single-window clips are w0; separate depth "
        "and semantic references are w0_depth_semantic. Per-row 'cwm_system' in the manifest overrides this. "
        "'none' keeps the flat user body.",
    )
    parser.add_argument(
        "--fit",
        default="resize",
        choices=FIT_MODES,
        help="How every stream reaches its grid. 'center-crop' keeps aspect and cuts the overflow equally from "
        "both sides (1280x720 -> 1280x704); 'resize' is the legacy stretch.",
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--proxy-references",
        nargs="+",
        default=None,
        choices=PROXY_REFERENCES,
        help="Video references made from a proxy_duv directory, in <Video N> order. 'duv' is the packed legacy "
        "reference; 'depth semantic' is two separate ones.",
    )
    reference_group.add_argument(
        "--proxy-variants",
        nargs="+",
        default=None,
        choices=PROXY_VARIANTS,
        help="Mixed-single-reference mode: each manifest row chooses one allowed modality with "
        "'proxy_modality'. Requires --cwm-system w0_omni.",
    )
    parser.add_argument(
        "--code-resize",
        choices=("reject", "nearest"),
        default="reject",
        help="How depth and semantic code planes reach a smaller proxy grid after the common aspect crop. "
        "'nearest' preserves labels and boundaries; the legacy default rejects every scale.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Rewrite text_embedding/text_token_tags on existing .pt files. Does not load the VAE "
        "or touch latents. Missing .pt files are skipped (run a full encode for those first).",
    )
    args = parser.parse_args()
    if args.num_frames % 17 != 5:
        raise SystemExit(f"--num-frames must satisfy n %% 17 == 5 for the H3 causal VAE, got {args.num_frames}")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if not 0 < args.qwen_video_fps <= 24:
        raise SystemExit(f"--qwen-video-fps must be in (0, 24], got {args.qwen_video_fps}")
    args.proxy_references = tuple(args.proxy_references or LEGACY_REFERENCES)
    args.proxy_variants = tuple(args.proxy_variants or ())
    if len(set(args.proxy_references)) != len(args.proxy_references):
        raise SystemExit(f"--proxy-references names a reference twice: {list(args.proxy_references)}")
    if len(set(args.proxy_variants)) != len(args.proxy_variants):
        raise SystemExit(f"--proxy-variants names a modality twice: {list(args.proxy_variants)}")
    if args.proxy_variants:
        if args.cwm_system != "w0_omni":
            raise SystemExit("--proxy-variants requires --cwm-system w0_omni")
        for variant in args.proxy_variants:
            check_role_references(args.cwm_system, (variant,))
    else:
        try:
            check_role_references(args.cwm_system, args.proxy_references)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    return args


def check_role_references(role: str, references: tuple[str, ...]) -> None:
    """Refuse a chat role that describes different videos than the sample carries.

    The system prompt is the only place Qwen is told what <Video 1> and <Video 2> are, so a w0
    prompt over a depth+semantic pair would call the depth video "the conditioning proxy" and say
    nothing of the second one -- a cache that encodes and trains without complaint.
    """
    if role == "none":
        return
    if role == "w0_omni":
        if len(references) != 1 or references[0] not in PROXY_VARIANTS:
            raise ValueError(
                f"CWM role {role!r} requires exactly one typed proxy video, got {list(references)}"
            )
        return
    expected = ROLE_VIDEO_REFERENCES.get(role)
    if expected is None:
        if len(references) != 1:
            raise ValueError(f"CWM role {role!r} describes one proxy video, but the proxy references are "
                             f"{list(references)}. Use --cwm-system w0_depth_semantic for depth+semantic, or none.")
    elif tuple(references) != expected:
        raise ValueError(f"CWM role {role!r} describes the video references {list(expected)} in that order, but "
                         f"the proxy references are {list(references)}.")


# ----------------------------------------------------------------------
# Media loading
# ----------------------------------------------------------------------


def cover_size(source: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    """The ``(H, W)`` an aspect-preserving scale of ``source`` reaches when it just covers ``target``."""
    (source_h, source_w), (target_h, target_w) = source, target
    scale = max(target_h / source_h, target_w / source_w)
    return max(target_h, round(source_h * scale)), max(target_w, round(source_w * scale))


def fit_frames(
    frames: np.ndarray,
    height: int,
    width: int,
    fit: str,
    *,
    codes: bool,
    what: str,
    code_resize: str = "reject",
) -> np.ndarray:
    """Bring ``[T, H, W]`` or ``[T, H, W, C]`` frames onto the ``height`` x ``width`` grid.

    ``codes`` marks planes whose values are not intensities -- metric depth, class ids, a DUV
    video's packed codes -- which may be cropped but never interpolated.
    """
    from fastvideo.pipelines.basic.minimax_h3.proxy import center_crop_box

    source = tuple(int(size) for size in frames.shape[1:3])
    if source == (height, width):
        return frames
    if codes and code_resize == "nearest":
        if fit != "center-crop":
            raise ValueError("--code-resize nearest requires --fit center-crop")
        source_h, source_w = source
        if source_w * height > width * source_h:
            crop_h, crop_w = source_h, round(source_h * width / height)
        else:
            crop_h, crop_w = round(source_w * height / width), source_w
        top, left = center_crop_box(source, (crop_h, crop_w))
        cropped = frames[:, top:top + crop_h, left:left + crop_w]
        return np.stack([
            np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.NEAREST))
            for frame in cropped
        ]).astype(frames.dtype, copy=False)
    if fit == "resize":
        if codes:
            raise ValueError(f"{what} is {source[1]}x{source[0]} but the grid is {width}x{height}, and its values "
                             "are codes that resizing would blend. Write it at the grid, or use --fit center-crop "
                             "if it is the target's resolution.")
        return np.stack(
            [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames])
    cover = cover_size(source, (height, width))
    if cover != source:
        if codes:
            raise ValueError(f"{what} is {source[1]}x{source[0]}; center-cropping it onto {width}x{height} would "
                             f"first need a scale to {cover[1]}x{cover[0]}, and its values are codes that cannot be "
                             "scaled. Cut the corpus at the target's resolution.")
        frames = np.stack([
            np.asarray(Image.fromarray(frame).resize((cover[1], cover[0]), Image.Resampling.LANCZOS))
            for frame in frames
        ])
    top, left = center_crop_box(cover, (height, width))
    return np.ascontiguousarray(frames[:, top:top + height, left:left + width])


def read_video_frames(path: Path, num_frames: int, height: int, width: int, fit: str = "resize") -> np.ndarray:
    """Decode, resample to 24 fps, trim, and fit a clip to ``[T, H, W, 3]`` uint8."""
    from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_video, resample_reference_frames

    frames, source_fps, _ = decode_reference_video(path)
    frames = resample_reference_frames(frames, source_fps)
    if frames.shape[0] < num_frames:
        raise ValueError(f"{path} yields {frames.shape[0]} frames at 24 fps; {num_frames} are required.")
    return fit_frames(frames[:num_frames], height, width, fit, codes=False, what=str(path))


def read_proxy_planes(directory: Path, num_frames: int, height: int, width: int,
                      fit: str = "resize", code_resize: str = "reject") -> tuple[np.ndarray, np.ndarray]:
    """Read ``[T, H, W]`` metric depth and class ids from a per-frame directory, on the grid.

    Under ``resize`` the planes must already be on the grid. Under ``center-crop`` they are read at
    whatever resolution the first semantic PNG says and cropped.
    """
    from fastvideo.pipelines.basic.minimax_h3.proxy import read_raw_depth, read_semantic_png

    plane_h, plane_w = height, width
    if fit == "center-crop":
        with Image.open(directory / f"{0:06d}.semantic_id.png") as first:
            plane_w, plane_h = first.size
    depth_frames = []
    semantic_frames = []
    for ordinal in range(num_frames):
        depth_frames.append(read_raw_depth(directory / f"{ordinal:06d}.depth.f32", height=plane_h, width=plane_w))
        semantic_frames.append(
            read_semantic_png(directory / f"{ordinal:06d}.semantic_id.png", height=plane_h, width=plane_w))
    what = f"The proxy planes in {directory}"
    depth = fit_frames(
        np.stack(depth_frames), height, width, fit, codes=True, what=what, code_resize=code_resize
    )
    semantic = fit_frames(
        np.stack(semantic_frames), height, width, fit, codes=True, what=what, code_resize=code_resize
    )
    return depth, semantic


def pixels_to_preview(pixels: torch.Tensor) -> np.ndarray:
    """``[1, 3, T, H, W]`` float pixels as the ``[T, H, W, 3]`` uint8 frames Qwen is shown.

    The VAE reads the full float, while Qwen only ever sees an 8-bit rendering of it, so quantizing
    once here keeps the preview honest about what the text encoder was shown.
    """
    return (pixels[0].permute(1, 2, 3, 0) * 255.0).round().clamp_(0, 255).to(torch.uint8).numpy()


def read_proxy_reference_clips(directory: Path, num_frames: int, height: int, width: int, fit: str,
                               references: tuple[str, ...],
                               code_resize: str = "reject") -> list[tuple[torch.Tensor, np.ndarray]]:
    """Build each named reference from one depth + class directory, as ``(pixels, preview)`` pairs."""
    from fastvideo.pipelines.basic.minimax_h3.proxy import proxy_reference_clip

    depth, semantic = read_proxy_planes(
        directory, num_frames, height, width, fit, code_resize
    )
    clips = []
    for kind in references:
        pixels = proxy_reference_clip(kind, depth, semantic)
        clips.append((pixels, pixels_to_preview(pixels)))
    return clips


def read_duv_clip(directory: Path,
                  num_frames: int,
                  height: int,
                  width: int,
                  fit: str = "resize") -> tuple[torch.Tensor, np.ndarray]:
    """Pack a depth + semantic-id frame directory into VAE pixels and a Qwen preview.

    Returns ``([1, 3, T, H, W]`` float32 in ``[0, 1]``, ``[T, H, W, 3]`` uint8``)``.
    """
    return read_proxy_reference_clips(directory, num_frames, height, width, fit, LEGACY_REFERENCES)[0]


def read_duv_video_clip(path: Path,
                        num_frames: int,
                        height: int,
                        width: int,
                        fit: str = "resize") -> tuple[torch.Tensor, np.ndarray]:
    """Read a pre-composed DUV video, refusing to resample it.

    Returns ``([1, 3, T, H, W]`` float32 in ``[0, 1]``, ``[T, H, W, 3]`` uint8``). Unlike
    :func:`read_duv_clip` this does no packing: the producer already encoded depth and class into
    the three channels, so the frames go to the VAE as they were written and the same bytes are
    what Qwen previews.

    There is deliberately no resize branch, which is the one way this differs from
    :func:`read_video_frames`. A DUV frame is three integer codes wearing an RGB costume, so any
    interpolation averages unrelated depths and paints class boundaries a code no segmenter ever
    predicted -- and produces a perfectly plausible-looking image while doing it. A grid mismatch is
    therefore an error to report, not a difference to smooth over. ``center-crop`` may still cut a
    target-resolution DUV down to the grid, since cropping keeps every code.
    """
    from fastvideo.pipelines.basic.minimax_h3.proxy import rgb_clip_to_pixels
    from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_video, resample_reference_frames

    frames, source_fps, _ = decode_reference_video(path)
    # Resampling to 24 fps only ever selects, repeats or drops whole frames, so it is safe on codes.
    frames = resample_reference_frames(frames, source_fps)
    if frames.shape[0] < num_frames:
        raise ValueError(f"{path} yields {frames.shape[0]} frames at 24 fps; {num_frames} are required.")
    frames = frames[:num_frames]
    if fit == "center-crop":
        frames = fit_frames(frames, height, width, fit, codes=True, what=str(path))
    if frames.shape[1:3] != (height, width):
        raise ValueError(f"{path} is {frames.shape[2]}x{frames.shape[1]} but the proxy grid is {width}x{height}. A "
                         "DUV video carries integer codes, so it has to be encoded at the grid it is consumed on; "
                         "resampling it would average unrelated depth codes and blend class colours. Re-encode the "
                         "clip, or set --proxy-width/--proxy-height to the grid it was written at.")
    return rgb_clip_to_pixels(frames), frames


def crop_to_aspect(image: Image.Image, height: int, width: int) -> Image.Image:
    """Center-crop ``image`` to the ``width / height`` aspect, keeping as many pixels as possible."""
    source_w, source_h = image.size
    if source_w * height > width * source_h:
        crop_w, crop_h = round(source_h * width / height), source_h
    else:
        crop_w, crop_h = source_w, round(source_w * height / width)
    if (crop_w, crop_h) == (source_w, source_h):
        return image
    left, top = (source_w - crop_w) // 2, (source_h - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def read_anchor_image(path: Path | None,
                      target_frames: np.ndarray,
                      short_edge: int,
                      *,
                      aspect: tuple[int, int] | None = None) -> Image.Image:
    """Resolve the anchor frame and scale it to a canvas the patch grid can tile.

    ``aspect`` ``(H, W)`` center-crops the anchor to the target's framing first, so a 16:9 anchor
    of a 1280x704 target shows what the target shows rather than 8 extra rows it never will.
    """
    image = Image.open(path).convert("RGB") if path is not None else Image.fromarray(target_frames[0])
    if aspect is not None:
        image = crop_to_aspect(image, *aspect)
    scale = short_edge / min(image.size)
    multiple = 32
    width = max(multiple, round(image.size[0] * scale / multiple) * multiple)
    height = max(multiple, round(image.size[1] * scale / multiple) * multiple)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def read_camera(path: Path, num_frames: int) -> dict[str, torch.Tensor | tuple[int, int]]:
    """Load a world-to-camera trajectory and its pixel-unit intrinsics."""
    with np.load(path) as payload:
        extrinsics = torch.from_numpy(np.asarray(payload["extrinsics"], dtype=np.float32))
        intrinsics = torch.from_numpy(np.asarray(payload["intrinsics"], dtype=np.float32))
        # `.files` rather than `in payload`: NpzFile only became a Mapping in recent NumPy.
        pixel_size = payload["pixel_size"] if "pixel_size" in payload.files else None
    if extrinsics.shape[0] < num_frames or intrinsics.shape[0] < num_frames:
        raise ValueError(f"{path} covers {extrinsics.shape[0]} frames; {num_frames} are required.")
    result: dict[str, Any] = {
        "camera_extrinsics": extrinsics[:num_frames].contiguous(),
        "camera_intrinsics": intrinsics[:num_frames].contiguous(),
    }
    if pixel_size is not None:
        result["pixel_size"] = (int(pixel_size[0]), int(pixel_size[1]))
    return result


# ----------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------


class Encoders:
    """Hold the H3 video VAE and Qwen3-VL conditioner for the length of one shard.

    Both are loaded through the inference component registry so that the cache is produced by the
    same classes and precision policy that will consume it. Unlike the single-sample overfit
    preprocessor these stay resident: a shard encodes hundreds of clips, and reloading a 32B text
    encoder per clip would dominate the run.
    """

    def __init__(
        self,
        model_path: Path,
        device: str,
        *,
        text_only: bool = False,
        cwm_system: str = "w0",
        qwen_video_fps: float = 24.0,
    ) -> None:
        from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
        from fastvideo.fastvideo_args import FastVideoArgs
        from fastvideo.models.loader.component_loader import PipelineComponentLoader
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
        from fastvideo.utils import verify_model_config_and_directory

        self.device = torch.device(device)
        self.model_path = model_path
        self.cwm_system = "" if cwm_system == "none" else cwm_system
        self.qwen_video_fps = float(qwen_video_fps)
        self.model_index = verify_model_config_and_directory(str(model_path))
        self.fastvideo_args = FastVideoArgs(
            model_path=str(model_path),
            pipeline_config=MiniMaxH3PipelineConfig(),
            num_gpus=1,
            tp_size=1,
            sp_size=1,
            hsdp_shard_dim=1,
            use_fsdp_inference=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
        )

        def load(name: str) -> Any:
            transformers_or_diffusers, _ = self.model_index[name][:2]
            return PipelineComponentLoader.load_module(
                module_name=name,
                component_model_path=str(model_path / name),
                transformers_or_diffusers=transformers_or_diffusers,
                fastvideo_args=self.fastvideo_args,
            )

        self.vae = None if text_only else load("vae")
        self.conditioning = MiniMaxH3ConditioningStage(
            conditioner=load("text_encoder"),
            tokenizer=load("tokenizer"),
            processor=load("processor"),
            ref2va=True,
        )

    @torch.no_grad()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode ``[1, 3, T, H, W]`` pixels in ``[0, 1]`` to ``[24, T', H', W']`` latents.

        The seed matches ``MINIMAX_H3_KEYFRAME_ENCODE_SEED`` so a cached reference is bit-identical
        to the one inference would build from the same pixels.
        """
        from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_ENCODE_SEED

        if self.vae is None:
            raise RuntimeError("Encoders were constructed with text_only=True; VAE is not loaded.")
        pixels = pixels.to(device=self.device, dtype=torch.float32)
        posterior = self.vae.encode(self.vae.normalize_pixels(pixels)).latent_dist
        generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        latents = self.vae.normalize_latents(posterior.sample(generator=generator).to(torch.float16).float())
        return latents.squeeze(0).float().cpu().contiguous()

    @torch.no_grad()
    def encode_keyframe(self, image: Image.Image) -> torch.Tensor:
        """Encode one still through the VAE's keyframe path to ``[24, 1, H', W']``."""
        from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_ENCODE_SEED

        if self.vae is None:
            raise RuntimeError("Encoders were constructed with text_only=True; VAE is not loaded.")
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None]
        pixels = pixels.to(device=self.device, dtype=torch.float32).div_(255.0)
        posterior = self.vae.encode_keyframe(self.vae.normalize_pixels(pixels)).latent_dist
        generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        latents = self.vae.normalize_latents(posterior.sample(generator=generator).to(torch.float16).float())
        return latents.squeeze(0).float().cpu().contiguous()

    @torch.no_grad()
    def encode_text(self,
                    prompt: str,
                    anchor: Image.Image,
                    proxy_previews: np.ndarray | list[np.ndarray],
                    *,
                    cwm_system: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Ref2VA conditioning stage over the same reference order training will pack.

        The stage tokenizes ``<Picture 1>`` then ``<Video 1>``, ``<Video 2>``... labels around
        Qwen's vision placeholders, so the per-token tags it returns are only valid for that exact
        reference order. The training plugin packs anchor-then-proxies for the same reason.
        """
        from fastvideo.pipelines import ForwardBatch
        from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import (
            MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS_KEY,
            MINIMAX_H3_TEXT_TOKEN_TAGS_KEY, )

        from fastvideo.pipelines.basic.minimax_h3.cwm_presentation import CWM_SYSTEM_PROMPT_KEY

        batch = ForwardBatch(data_type="video", prompt=prompt)
        role = self.cwm_system if cwm_system is None else ("" if cwm_system == "none" else cwm_system)
        if role:
            batch.extra[CWM_SYSTEM_PROMPT_KEY] = role
        batch.extra[MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS_KEY] = self.qwen_video_fps
        previews = [proxy_previews] if isinstance(proxy_previews, np.ndarray) else list(proxy_previews)
        batch.references = [
            MiniMaxH3PreparedReference(media_type="image", image=anchor),
            *(MiniMaxH3PreparedReference(media_type="video", frames=preview) for preview in previews),
        ]
        batch = self.conditioning.forward(batch, self.fastvideo_args)
        if not batch.prompt_embeds:
            raise RuntimeError("MiniMax-H3 conditioning returned no prompt embedding")
        tags = batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY]
        return batch.prompt_embeds[0].squeeze(0).float().cpu().contiguous(), tags.to(torch.long).contiguous()


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def free_localhost_port() -> str:
    """Reserve an ephemeral port by binding it, then release it for the store to rebind."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


def init_single_process_distributed() -> None:
    """Initialize the one-rank process groups the component loaders require.

    The port has to be per-process, not a fixed constant: sharding a manifest across eight GPUs
    means eight of these running at once, and each rank-0 group stands up its own TCP store. A
    shared port lets exactly one shard start and the other seven die on ``EADDRINUSE`` seconds in,
    which looks like a data problem and is not one. An explicit ``MASTER_PORT`` still wins, so a
    caller that needs a fixed port can set one.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", free_localhost_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)


def encode_entry(entry: dict[str, Any], encoders: Encoders, args: argparse.Namespace, root: Path) -> dict[str, Any]:
    missing = [key for key in REQUIRED_MEDIA_KEYS if not entry.get(key)]
    if missing:
        raise KeyError(f"Manifest entry is missing {missing}")
    supplied = [key for key in PROXY_KEYS if entry.get(key)]
    if not mixed_variants(args) and len(supplied) != 1:
        raise KeyError(f"A manifest entry needs exactly one of {list(PROXY_KEYS)}, got {supplied}")

    def resolve(key: str) -> Path | None:
        value = entry.get(key)
        return None if not value else (root / str(value))

    role = entry_cwm_system(entry, args.cwm_system)
    references = entry_proxy_references(entry, args)
    check_role_references(role, references)
    target_frames = read_video_frames(root / str(entry["target"]), args.num_frames, args.height, args.width, args.fit)
    target_pixels = torch.from_numpy(target_frames.copy()).permute(3, 0, 1, 2)[None].float().div_(255.0)

    proxies = load_proxy_clips(entry, args, root)
    anchor = read_anchor_image(None if mixed_variants(args) else resolve("anchor"),
                               target_frames,
                               args.anchor_short_edge,
                               aspect=anchor_aspect(args))
    original_prompt = str(entry["prompt"])
    conditioning_prompt = (
        f"Reference modality: {references[0]} video.\n{original_prompt}"
        if mixed_variants(args)
        else original_prompt
    )
    text_embedding, text_token_tags = encoders.encode_text(
        conditioning_prompt, anchor, [preview for _, preview in proxies], cwm_system=role)

    sample: dict[str, Any] = {"vae_latent": encoders.encode_pixels(target_pixels)}
    if is_legacy(args):
        sample["proxy_latent"] = encoders.encode_pixels(proxies[0][0])
    else:
        sample["proxy_latents"] = torch.stack([encoders.encode_pixels(pixels) for pixels, _ in proxies])
    sample.update({
        "anchor_latent": encoders.encode_keyframe(anchor),
        "text_embedding": text_embedding,
        "text_token_tags": text_token_tags,
        "info": {
            "num_frames": int(args.num_frames),
            "pixel_size": (int(args.height), int(args.width)),
            "qwen_video_fps": float(args.qwen_video_fps),
            "prompt": original_prompt,
            **({"conditioning_prompt": conditioning_prompt} if mixed_variants(args) else {}),
            **({"anchor_source": "target_first_frame"} if mixed_variants(args) else {}),
            "cwm_system": role,
            **reference_info(args, entry),
        },
    })
    camera_path = resolve("camera")
    if camera_path is not None:
        camera = read_camera(camera_path, args.num_frames)
        pixel_size = camera.pop("pixel_size", None)
        sample.update(camera)
        if pixel_size is not None:
            sample["info"]["pixel_size"] = pixel_size
    return sample


def entry_cwm_system(entry: dict[str, Any], default: str) -> str:
    """Per-row override, then the process-wide flag. Empty means no chat wrap."""
    raw = entry.get("cwm_system", default)
    role = str(raw or "none").strip().lower()
    return "none" if role in {"", "none"} else role


def is_legacy(args: argparse.Namespace) -> bool:
    """The single packed reference, cached under ``proxy_latent`` exactly as before."""
    return not mixed_variants(args) and tuple(args.proxy_references) == LEGACY_REFERENCES


def mixed_variants(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(getattr(args, "proxy_variants", ()) or ())


def entry_proxy_references(entry: dict[str, Any], args: argparse.Namespace) -> tuple[str, ...]:
    variants = mixed_variants(args)
    if not variants:
        return tuple(args.proxy_references)
    modality = str(entry.get("proxy_modality") or "").strip().lower()
    if modality not in variants:
        raise ValueError(
            f"manifest proxy_modality must be one of {list(variants)}, got {modality!r}"
        )
    return (modality,)


def anchor_aspect(args: argparse.Namespace) -> tuple[int, int] | None:
    return (int(args.height), int(args.width)) if args.fit == "center-crop" else None


def reference_info(
    args: argparse.Namespace,
    entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cache metadata for the non-legacy options, absent otherwise so a legacy sample is unchanged."""
    info: dict[str, Any] = {}
    if args.fit != "resize":
        info["fit"] = args.fit
    variants = mixed_variants(args)
    if variants:
        if entry is None:
            info["proxy_variants"] = list(variants)
        else:
            modality = entry_proxy_references(entry, args)[0]
            info.update({
                "proxy_references": [modality],
                "proxy_modality": modality,
                "proxy_variants": list(variants),
                "code_resize": str(getattr(args, "code_resize", "reject")),
            })
    elif not is_legacy(args):
        info["proxy_references"] = list(args.proxy_references)
    return info


def load_proxy_clips(entry: dict[str, Any], args: argparse.Namespace,
                     root: Path) -> list[tuple[torch.Tensor, np.ndarray]]:
    """Every proxy video reference of an entry, in <Video N> order, as ``(pixels, preview)``."""
    grid = (args.num_frames, args.proxy_height, args.proxy_width)
    variants = mixed_variants(args)
    if variants:
        modality = entry_proxy_references(entry, args)[0]
        if modality == "style":
            if not entry.get("proxy"):
                raise KeyError("proxy_modality='style' requires the manifest 'proxy' RGB video")
            preview = read_video_frames(root / str(entry["proxy"]), *grid, args.fit)
            pixels = torch.from_numpy(preview.copy()).permute(3, 0, 1, 2)[None].float().div_(255.0)
            return [(pixels, preview)]
        if not entry.get("proxy_duv"):
            raise KeyError(
                f"proxy_modality={modality!r} requires 'proxy_duv' depth/semantic planes"
            )
        return read_proxy_reference_clips(
            root / str(entry["proxy_duv"]),
            *grid,
            args.fit,
            (modality,),
            str(getattr(args, "code_resize", "reject")),
        )
    if not is_legacy(args):
        if not entry.get("proxy_duv"):
            raise KeyError(f"--proxy-references {' '.join(args.proxy_references)} builds the references from depth and "
                           "class planes, so the entry needs 'proxy_duv' (a directory of .depth.f32 and "
                           ".semantic_id.png frames); a composed video or RGB render cannot be split.")
        return read_proxy_reference_clips(
            root / str(entry["proxy_duv"]),
            *grid,
            args.fit,
            args.proxy_references,
            str(getattr(args, "code_resize", "reject")),
        )
    if entry.get("proxy_duv"):
        return [read_duv_clip(root / str(entry["proxy_duv"]), *grid, args.fit)]
    if entry.get("proxy_duv_video"):
        return [read_duv_video_clip(root / str(entry["proxy_duv_video"]), *grid, args.fit)]
    preview = read_video_frames(root / str(entry["proxy"]), *grid, args.fit)
    return [(torch.from_numpy(preview.copy()).permute(3, 0, 1, 2)[None].float().div_(255.0), preview)]


def load_proxy_previews(entry: dict[str, Any], args: argparse.Namespace, root: Path) -> list[np.ndarray]:
    return [preview for _, preview in load_proxy_clips(entry, args, root)]


def encode_entry_text_only(entry: dict[str, Any], encoders: Encoders, args: argparse.Namespace, root: Path,
                           out_path: Path) -> dict[str, Any]:
    """Refresh Qwen rows on an existing sample. Latents stay as they were."""
    sample = torch.load(out_path, map_location="cpu", weights_only=False)
    if not isinstance(sample, dict) or "vae_latent" not in sample:
        raise ValueError(f"{out_path} is not an H3 proxy cache sample")
    recorded = tuple((sample.get("info") or {}).get("proxy_references") or LEGACY_REFERENCES)
    references = entry_proxy_references(entry, args)
    if recorded != references:
        raise ValueError(
            f"{out_path} carries the proxy references {list(recorded)}, but this row requires "
            f"{list(references)}; the Qwen rows would describe videos the latents are not."
        )
    role = entry_cwm_system(entry, args.cwm_system)
    check_role_references(role, references)
    proxy_previews = load_proxy_previews(entry, args, root)
    if entry.get("anchor") and not mixed_variants(args):
        # Target pixels are not loaded; the stored anchor size is unused for Qwen's PIL path.
        dummy = np.zeros((1, args.height, args.width, 3), dtype=np.uint8)
        anchor = read_anchor_image(root / str(entry["anchor"]), dummy, args.anchor_short_edge, aspect=anchor_aspect(args))
    else:
        target_frames = read_video_frames(root / str(entry["target"]), args.num_frames, args.height, args.width,
                                          args.fit)
        anchor = read_anchor_image(None, target_frames, args.anchor_short_edge, aspect=anchor_aspect(args))
    original_prompt = str(entry["prompt"])
    conditioning_prompt = (
        f"Reference modality: {references[0]} video.\n{original_prompt}"
        if mixed_variants(args)
        else original_prompt
    )
    text_embedding, text_token_tags = encoders.encode_text(
        conditioning_prompt, anchor, proxy_previews, cwm_system=role)
    sample["text_embedding"] = text_embedding
    sample["text_token_tags"] = text_token_tags
    info = dict(sample.get("info") or {})
    info["prompt"] = original_prompt
    if mixed_variants(args):
        info["conditioning_prompt"] = conditioning_prompt
        info.update(reference_info(args, entry))
    info["cwm_system"] = role
    info["qwen_video_fps"] = float(args.qwen_video_fps)
    sample["info"] = info
    return sample


MEDIA_KEYS = ("target", *PROXY_KEYS, "anchor")


def assign_mixed_modalities(
    entries: list[dict[str, Any]],
    variants: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Assign one available modality per unlabelled row, balanced and deterministically.

    A row may carry both ``proxy_duv`` planes and a ``proxy`` style video. It is still encoded
    exactly once: explicit ``proxy_modality`` wins; otherwise consecutive rows rotate through the
    requested variants. This avoids storing the large target latent four times.
    """

    assigned: list[dict[str, Any]] = []
    counts = dict.fromkeys(variants, 0)
    for index, original in enumerate(entries):
        entry = dict(original)
        explicit = str(entry.get("proxy_modality") or "").strip().lower()
        if explicit:
            if explicit not in variants:
                raise SystemExit(
                    f"Manifest row {index + 1} has proxy_modality={explicit!r}, not one of "
                    f"{list(variants)}"
                )
            choices = (explicit,)
        else:
            rotated = variants[index % len(variants):] + variants[:index % len(variants)]
            choices = tuple(sorted(rotated, key=lambda kind: counts[kind]))
        available = [
            kind
            for kind in choices
            if (kind == "style" and entry.get("proxy"))
            or (kind != "style" and entry.get("proxy_duv"))
        ]
        if not available:
            raise SystemExit(
                f"Manifest row {index + 1} has no source for any selected proxy modality; "
                "DUV/depth/semantic need 'proxy_duv', and style needs 'proxy'"
            )
        entry["proxy_modality"] = available[0]
        counts[available[0]] += 1
        assigned.append(entry)
    missing = [kind for kind, count in counts.items() if count == 0]
    if missing:
        raise SystemExit(
            f"No manifest row was assigned requested proxy modalities {missing}; "
            "remove unavailable names from --proxy-variants or add their media"
        )
    return assigned


def preflight_media(
    entries: list[dict[str, Any]],
    root: Path,
    args: argparse.Namespace | None = None,
) -> None:
    """Stop on a wrong ``--root`` before the VAE and the text encoder load.

    Every path a seg manifest writes is relative and ``--root`` defaults to the launch directory, so
    pointing it at the checkout instead of the dataset is the easy mistake -- and it currently costs
    minutes of model loading followed by one identical ENOENT per clip, which reads like a corrupt
    corpus rather than a mistyped flag.

    A few entries are sampled rather than one. A single clip that lost a file is a real failure that
    belongs in the per-clip log; every sampled clip missing everything is the flag.
    """
    sample = entries[:3]
    reports = []
    for entry in sample:
        if args is not None and mixed_variants(args):
            modality = entry_proxy_references(entry, args)[0]
            source_key = "proxy" if modality == "style" else "proxy_duv"
            keys = ("target", source_key)
        else:
            keys = MEDIA_KEYS
        present = [str(entry[key]) for key in keys if entry.get(key)]
        missing = [value for value in present if not (root / value).exists()]
        if len(missing) != len(present) or not present:
            return
        reports.append(missing[0])
    resolved = root.resolve()
    raise SystemExit(
        f"None of the media in the first {len(sample)} manifest entries exists under --root.\n"
        f"  --root resolves to {resolved}\n"
        f"  tried e.g. {resolved / reports[0]}\n"
        "  Manifest paths are relative, so --root has to be the dataset directory that holds the "
        "seg_*/ subdirectories, not the checkout.")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip() and not line.startswith("#")]
    if mixed_variants(args):
        entries = assign_mixed_modalities(entries, mixed_variants(args))
    entries = entries[args.shard_index::args.num_shards]
    if not entries:
        raise SystemExit(f"Manifest shard {args.shard_index}/{args.num_shards} is empty")
    preflight_media(entries, root, args)

    init_single_process_distributed()
    encoders = Encoders(
        Path(args.model_path).expanduser().resolve(),
        device=args.device,
        text_only=args.text_only,
        cwm_system=args.cwm_system,
        qwen_video_fps=args.qwen_video_fps,
    )

    written = skipped = failed = 0
    for index, entry in enumerate(entries):
        # Flatten ids like ``kof-video-0809/seg_0001``: the train loader scans one directory level.
        name = str(entry.get("name") or Path(str(entry["target"])).stem).replace("/", "__")
        out_path = output_dir / f"{name}.pt"
        if args.text_only:
            if not out_path.exists():
                skipped += 1
                print(f"[{index + 1}/{len(entries)}] SKIP {name}: no existing .pt for --text-only")
                continue
        elif out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            if args.text_only:
                sample = encode_entry_text_only(entry, encoders, args, root, out_path)
            else:
                sample = encode_entry(entry, encoders, args, root)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            failed += 1
            print(f"[{index + 1}/{len(entries)}] FAILED {name}: {error}")
            continue
        # Write beside the target and rename, so an interrupted shard never leaves a truncated
        # sample that the training loader would have to skip.
        temporary = out_path.with_suffix(".pt.tmp")
        torch.save(sample, temporary)
        os.replace(temporary, out_path)
        written += 1
        if written % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        if args.text_only:
            print(f"[{index + 1}/{len(entries)}] {name}: text {tuple(sample['text_embedding'].shape)} "
                  f"cwm_system={sample['info'].get('cwm_system')}")
        else:
            proxy_latent = sample.get("proxy_latent", sample.get("proxy_latents"))
            print(f"[{index + 1}/{len(entries)}] {name}: target {tuple(sample['vae_latent'].shape)}, "
                  f"proxy {tuple(proxy_latent.shape)}")

    print(f"Done: {written} written, {skipped} skipped, {failed} failed -> {output_dir}")


if __name__ == "__main__":
    main()
