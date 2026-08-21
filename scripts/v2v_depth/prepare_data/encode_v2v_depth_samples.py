# SPDX-License-Identifier: Apache-2.0
"""Encode game video-to-video + depth clips into the cached ``.pt`` format.

Reads a JSONL manifest and writes one ``.pt`` per clip holding the target clip's
VAE latent, the source clip's VAE latent (the video-to-video condition), the T5
text embedding, and the depth latents. Training then never loads a VAE or a text
encoder.

Manifest, one JSON object per line::

    {"target": "3a/0001.mp4",
     "source": "simple/0001.mp4",
     "depth": "depth/0001.mp4",
     "depth_wide": "depth_wide/0001.mp4",
     "prompt": "a knight walks through a ruined cathedral",
     "depth_range": [0.1, 500.0]}

``depth_wide`` is optional and only needed when training with the wide-FOV
branch; it must be the same clip re-rendered with the focal length divided by
``wide_fov_scale`` (same resolution, same frames, same depth range).

Paths are resolved relative to ``--root``. ``depth_range`` overrides
``--depth-near/--depth-far`` for that clip.

Depth is stored as an encoded *video*, not as raw metres: the Wan VAE is the only
encoder available and it expects 3-channel [-1, 1] input, so depth is mapped to
disparity, normalized against a fixed near/far pair, and replicated to 3
channels. Fixing the range (rather than per-clip min/max) is what keeps the
control signal comparable across clips and across chunks of one streaming
rollout.

Usage::

    python scripts/v2v_depth/prepare_data/encode_v2v_depth_samples.py \\
        --manifest data/game_v2v_depth/manifest.jsonl \\
        --root data/game_v2v_depth/raw \\
        --output data/game_v2v_depth/train \\
        --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers \\
        --num-frames 81 --height 480 --width 832
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from fastvideo.dataset.v2v_depth_preprocess import (load_depth_clip, load_rgb_clip)

TEXT_MAX_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".", help="Base directory for relative manifest paths.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path",
                        required=True,
                        help="Wan diffusers snapshot (or HF id) providing vae/ and text_encoder/.")
    parser.add_argument("--num-frames",
                        type=int,
                        default=81,
                        help="Must satisfy (num_frames - 1) %% 4 == 0 for the Wan VAE.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    # Outer-edge crop BEFORE resize. Fractions of the source H/W removed from
    # each side. KOF / SF3-style HUD (measured on 1280x720 src): top≈0.175
    # clears portraits+HP+timer; bottom≈0.105 clears super meters.
    parser.add_argument("--crop-top", type=float, default=0.0)
    parser.add_argument("--crop-bottom", type=float, default=0.0)
    parser.add_argument("--crop-left", type=float, default=0.0)
    parser.add_argument("--crop-right", type=float, default=0.0)
    parser.add_argument("--depth-near", type=float, default=0.1)
    parser.add_argument("--depth-far", type=float, default=500.0)
    parser.add_argument("--depth-encoding", choices=("disparity", "linear"), default="disparity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--shard-index",
                        type=int,
                        default=0,
                        help="This worker's index, for splitting a manifest across GPUs.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (args.num_frames - 1) % 4 != 0:
        raise SystemExit(f"--num-frames must satisfy (n - 1) % 4 == 0 for the Wan VAE, got {args.num_frames}")
    if args.depth_far <= args.depth_near <= 0.0:
        raise SystemExit("--depth-near must be > 0 and --depth-far must exceed it")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    for name in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 0.5:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 0.5), got {value}")
    if args.crop_top + args.crop_bottom >= 1.0 or args.crop_left + args.crop_right >= 1.0:
        raise SystemExit("crop fractions leave an empty frame")
    return args


# ----------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------


class Encoders:
    """Wan VAE + UMT5 text encoder, loaded once and reused for every clip."""

    def __init__(self, model_path: str, *, device: str, dtype: torch.dtype) -> None:
        from diffusers import AutoencoderKLWan
        from transformers import AutoTokenizer, UMT5EncoderModel

        self.device = torch.device(device)
        self.dtype = dtype
        self.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae",
                                                    torch_dtype=torch.float32).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder",
                                                             torch_dtype=dtype).to(self.device).eval()

    @torch.no_grad()
    def encode_video(self, clip: torch.Tensor) -> torch.Tensor:
        """Encode ``[1, 3, T, H, W]`` pixels to ``[16, T', H', W']`` raw latents.

        The distribution mode is used rather than a sample: a cached condition
        should be deterministic, so two runs over the same clip agree.
        """
        latent = self.vae.encode(clip.to(self.device, dtype=torch.float32)).latent_dist.mode()
        return latent[0].to(torch.float32).cpu()

    @torch.no_grad()
    def encode_text(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mask = tokens.attention_mask.to(self.device)
        embeds = self.text_encoder(tokens.input_ids.to(self.device), attention_mask=mask).last_hidden_state
        # Zero the padding so a padded position can never leak a value through
        # a model that ignores the mask.
        embeds = embeds * mask.unsqueeze(-1).to(embeds.dtype)
        return embeds[0].to(torch.float32).cpu(), mask[0].to(torch.float32).cpu()


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    root = Path(args.root).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip() and not line.startswith("#")]
    entries = entries[args.shard_index::args.num_shards]
    if not entries:
        raise SystemExit(f"Manifest shard {args.shard_index}/{args.num_shards} is empty")

    encoders = Encoders(args.model_path, device=args.device, dtype=dtype)

    written = skipped = failed = 0
    for index, entry in enumerate(entries):
        # Flatten ids like ``kof-video-0809/seg_0001`` — the train loader only
        # scans one directory level for ``*.pt``.
        name = str(entry.get("name") or Path(str(entry["target"])).stem).replace("/", "__")
        out_path = output_dir / f"{name}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            near, far = entry.get("depth_range", (args.depth_near, args.depth_far))
            crop = {
                "crop_top": float(args.crop_top),
                "crop_bottom": float(args.crop_bottom),
                "crop_left": float(args.crop_left),
                "crop_right": float(args.crop_right),
            }
            common = {
                "num_frames": args.num_frames,
                "height": args.height,
                "width": args.width,
                **crop,
            }
            depth_common = {
                **common,
                "near": float(near),
                "far": float(far),
                "encoding": args.depth_encoding,
            }

            sample: dict[str, Any] = {
                "vae_latent": encoders.encode_video(load_rgb_clip(root / entry["target"], **common)),
                "control_latent": encoders.encode_video(load_rgb_clip(root / entry["source"], **common)),
            }
            if "depth" in entry:
                sample["depth_latent"] = encoders.encode_video(load_depth_clip(root / entry["depth"], **depth_common))
            if "depth_wide" in entry:
                sample["depth_wide_latent"] = encoders.encode_video(
                    load_depth_clip(root / entry["depth_wide"], **depth_common))

            embeds, mask = encoders.encode_text(str(entry.get("prompt", "")))
            sample["text_embedding"] = embeds
            sample["text_attention_mask"] = mask
            sample["info"] = {
                "name": name,
                "target": str(entry["target"]),
                "source": str(entry["source"]),
                "depth_range": [float(near), float(far)],
                "depth_encoding": args.depth_encoding,
                "num_frames": args.num_frames,
                "resolution": [args.height, args.width],
                "crop": crop,
            }

            shapes = {key: tuple(value.shape) for key, value in sample.items() if torch.is_tensor(value)}
            latent_shapes = {key: shape for key, shape in shapes.items() if key.endswith("latent")}
            if len(set(latent_shapes.values())) > 1:
                raise ValueError(f"latents disagree on shape: {latent_shapes}")

            tmp_path = out_path.with_suffix(".pt.tmp")
            torch.save(sample, tmp_path)
            os.replace(tmp_path, out_path)
            written += 1
        except Exception as error:  # noqa: BLE001 - one bad clip must not end the pass
            failed += 1
            print(f"[{index + 1}/{len(entries)}] FAILED {name}: {type(error).__name__}: {error}", flush=True)
            continue

        if written % 20 == 0 or index + 1 == len(entries):
            print(f"[{index + 1}/{len(entries)}] written={written} skipped={skipped} failed={failed}", flush=True)

    print(f"Done: written={written} skipped={skipped} failed={failed} -> {output_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
