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
     "anchor": "anchor/0001.png",
     "camera": "poses/0001.npz",
     "prompt": "a knight walks through a ruined cathedral"}

Exactly one of ``proxy`` and ``proxy_duv`` is required. ``proxy`` is an ordinary RGB render, used
as-is. ``proxy_duv`` is a directory of per-frame ``NNNNNN.depth.f32`` and ``NNNNNN.semantic_id.png``
pairs, packed into three channels by :mod:`fastvideo.pipelines.basic.minimax_h3.proxy`; prefer it
when the renderer can emit depth, because a geometry channel constrains the output far more tightly
than a shaded render does.

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
        --num-frames 124 --height 768 --width 1344
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
    # CWM anchors at 2048 short edge. That buys detail the target canvas cannot show and costs ~7x
    # the anchor tokens, so the default here matches the target canvas instead.
    parser.add_argument("--anchor-short-edge", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0, help="This worker's index, for splitting a manifest.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_frames % 17 != 5:
        raise SystemExit(f"--num-frames must satisfy n %% 17 == 5 for the H3 causal VAE, got {args.num_frames}")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    return args


# ----------------------------------------------------------------------
# Media loading
# ----------------------------------------------------------------------


def read_video_frames(path: Path, num_frames: int, height: int, width: int) -> np.ndarray:
    """Decode, resample to 24 fps, trim, and resize a clip to ``[T, H, W, 3]`` uint8."""
    from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_video, resample_reference_frames

    frames, source_fps, _ = decode_reference_video(path)
    frames = resample_reference_frames(frames, source_fps)
    if frames.shape[0] < num_frames:
        raise ValueError(f"{path} yields {frames.shape[0]} frames at 24 fps; {num_frames} are required.")
    frames = frames[:num_frames]
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames])


def read_duv_clip(directory: Path, num_frames: int, height: int, width: int) -> tuple[torch.Tensor, np.ndarray]:
    """Pack a depth + semantic-id frame directory into VAE pixels and a Qwen preview.

    Returns ``([1, 3, T, H, W]`` float32 in ``[0, 1]``, ``[T, H, W, 3]`` uint8``)``. The two differ
    in depth precision: the VAE reads the full log-normalized float, while Qwen only ever sees an
    8-bit rendering of it, so quantizing once here keeps the preview honest about what the text
    encoder was shown.
    """
    from fastvideo.pipelines.basic.minimax_h3.proxy import pack_duv_clip, read_raw_depth, read_semantic_png

    depth_frames = []
    semantic_frames = []
    for ordinal in range(num_frames):
        depth_frames.append(read_raw_depth(directory / f"{ordinal:06d}.depth.f32", height=height, width=width))
        semantic_frames.append(
            read_semantic_png(directory / f"{ordinal:06d}.semantic_id.png", height=height, width=width))
    pixels = pack_duv_clip(np.stack(depth_frames), np.stack(semantic_frames))
    preview = (pixels[0].permute(1, 2, 3, 0) * 255.0).round().clamp_(0, 255).to(torch.uint8).numpy()
    return pixels, preview


def read_anchor_image(path: Path | None, target_frames: np.ndarray, short_edge: int) -> Image.Image:
    """Resolve the anchor frame and scale it to a canvas the patch grid can tile."""
    image = Image.open(path).convert("RGB") if path is not None else Image.fromarray(target_frames[0])
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

    def __init__(self, model_path: Path, device: str) -> None:
        from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
        from fastvideo.fastvideo_args import FastVideoArgs
        from fastvideo.models.loader.component_loader import PipelineComponentLoader
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
        from fastvideo.utils import verify_model_config_and_directory

        self.device = torch.device(device)
        self.model_path = model_path
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

        self.vae = load("vae")
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

        pixels = pixels.to(device=self.device, dtype=torch.float32)
        posterior = self.vae.encode(self.vae.normalize_pixels(pixels)).latent_dist
        generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        latents = self.vae.normalize_latents(posterior.sample(generator=generator).to(torch.float16).float())
        return latents.squeeze(0).float().cpu().contiguous()

    @torch.no_grad()
    def encode_keyframe(self, image: Image.Image) -> torch.Tensor:
        """Encode one still through the VAE's keyframe path to ``[24, 1, H', W']``."""
        from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_ENCODE_SEED

        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None]
        pixels = pixels.to(device=self.device, dtype=torch.float32).div_(255.0)
        posterior = self.vae.encode_keyframe(self.vae.normalize_pixels(pixels)).latent_dist
        generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        latents = self.vae.normalize_latents(posterior.sample(generator=generator).to(torch.float16).float())
        return latents.squeeze(0).float().cpu().contiguous()

    @torch.no_grad()
    def encode_text(self, prompt: str, anchor: Image.Image,
                    proxy_preview: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Ref2VA conditioning stage over the same reference order training will pack.

        The stage tokenizes ``<Picture 1>`` then ``<Video 1>`` labels around Qwen's vision
        placeholders, so the per-token tags it returns are only valid for that exact reference
        order. The training plugin packs anchor-then-proxy for the same reason.
        """
        from fastvideo.pipelines import ForwardBatch
        from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import (
            MINIMAX_H3_TEXT_TOKEN_TAGS_KEY, )

        batch = ForwardBatch(data_type="video", prompt=prompt)
        batch.references = [
            MiniMaxH3PreparedReference(media_type="image", image=anchor),
            MiniMaxH3PreparedReference(media_type="video", frames=proxy_preview),
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
    if bool(entry.get("proxy")) == bool(entry.get("proxy_duv")):
        raise KeyError("A manifest entry needs exactly one of 'proxy' and 'proxy_duv'")

    def resolve(key: str) -> Path | None:
        value = entry.get(key)
        return None if not value else (root / str(value))

    target_frames = read_video_frames(root / str(entry["target"]), args.num_frames, args.height, args.width)
    target_pixels = torch.from_numpy(target_frames.copy()).permute(3, 0, 1, 2)[None].float().div_(255.0)

    if entry.get("proxy_duv"):
        proxy_pixels, proxy_preview = read_duv_clip(root / str(entry["proxy_duv"]), args.num_frames, args.proxy_height,
                                                    args.proxy_width)
    else:
        proxy_preview = read_video_frames(root / str(entry["proxy"]), args.num_frames, args.proxy_height,
                                          args.proxy_width)
        proxy_pixels = torch.from_numpy(proxy_preview.copy()).permute(3, 0, 1, 2)[None].float().div_(255.0)

    anchor = read_anchor_image(resolve("anchor"), target_frames, args.anchor_short_edge)
    text_embedding, text_token_tags = encoders.encode_text(str(entry["prompt"]), anchor, proxy_preview)

    sample: dict[str, Any] = {
        "vae_latent": encoders.encode_pixels(target_pixels),
        "proxy_latent": encoders.encode_pixels(proxy_pixels),
        "anchor_latent": encoders.encode_keyframe(anchor),
        "text_embedding": text_embedding,
        "text_token_tags": text_token_tags,
        "info": {
            "num_frames": int(args.num_frames),
            "pixel_size": (int(args.height), int(args.width)),
            "prompt": str(entry["prompt"]),
        },
    }
    camera_path = resolve("camera")
    if camera_path is not None:
        camera = read_camera(camera_path, args.num_frames)
        pixel_size = camera.pop("pixel_size", None)
        sample.update(camera)
        if pixel_size is not None:
            sample["info"]["pixel_size"] = pixel_size
    return sample


def main() -> None:
    args = parse_args()
    init_single_process_distributed()
    root = Path(args.root).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip() and not line.startswith("#")]
    entries = entries[args.shard_index::args.num_shards]
    if not entries:
        raise SystemExit(f"Manifest shard {args.shard_index}/{args.num_shards} is empty")

    encoders = Encoders(Path(args.model_path).expanduser().resolve(), device=args.device)

    written = skipped = failed = 0
    for index, entry in enumerate(entries):
        # Flatten ids like ``kof-video-0809/seg_0001``: the train loader scans one directory level.
        name = str(entry.get("name") or Path(str(entry["target"])).stem).replace("/", "__")
        out_path = output_dir / f"{name}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
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
        print(f"[{index + 1}/{len(entries)}] {name}: target {tuple(sample['vae_latent'].shape)}, "
              f"proxy {tuple(sample['proxy_latent'].shape)}")

    print(f"Done: {written} written, {skipped} skipped, {failed} failed -> {output_dir}")


if __name__ == "__main__":
    main()
