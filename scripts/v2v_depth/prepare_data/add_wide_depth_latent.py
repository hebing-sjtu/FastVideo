# SPDX-License-Identifier: Apache-2.0
"""Backfill ``depth_wide_latent`` into an existing cached ``.pt`` dataset.

The wide-FOV branch is usually enabled only at the DMD stage, by which point the
narrow cache already exists. Rather than re-encode every clip, this reads the
wide-FOV depth renders and adds one key per sample in place.

The wide render must match its narrow counterpart frame for frame and use the
same depth range; only the focal length differs (divided by ``wide_fov_scale``,
so the same resolution covers a proportionally wider field of view).

Usage::

    python scripts/v2v_depth/prepare_data/add_wide_depth_latent.py \\
        --cache data/game_v2v_depth/train \\
        --wide-depth-dir data/game_v2v_depth/raw/depth_wide \\
        --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from encode_v2v_depth_samples import Encoders, load_depth_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="Directory of cached *.pt samples to modify in place.")
    parser.add_argument("--wide-depth-dir",
                        required=True,
                        help="Wide-FOV depth renders, named <sample>.mp4 or <sample>/ per cached sample.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode samples that already carry the key.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def _resolve_wide_source(wide_dir: Path, name: str) -> Path:
    candidates = [wide_dir / name, *(wide_dir / f"{name}{suffix}" for suffix in (".mp4", ".webm", ".mkv", ".avi"))]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No wide-FOV depth render for {name!r} under {wide_dir}")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache).expanduser()
    wide_dir = Path(args.wide_depth_dir).expanduser()
    sample_paths = sorted(p for p in cache_dir.iterdir() if p.suffix == ".pt")
    if not sample_paths:
        raise SystemExit(f"No cached *.pt samples under {cache_dir}")
    sample_paths = sample_paths[args.shard_index::args.num_shards]

    encoders = Encoders(args.model_path, device=args.device, dtype=torch.bfloat16)

    written = skipped = failed = 0
    for index, path in enumerate(sample_paths):
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if "depth_wide_latent" in sample and not args.overwrite:
            skipped += 1
            continue
        if "depth_latent" not in sample:
            raise SystemExit(f"{path} has no 'depth_latent'; the wide branch conditions alongside the narrow one.")

        info = sample.get("info", {}) if isinstance(sample.get("info"), dict) else {}
        near, far = info.get("depth_range", (0.1, 500.0))
        height, width = info.get("resolution", (480, 832))
        num_frames = info.get("num_frames")
        if num_frames is None:
            # Latent temporal length maps back to pixel frames through the Wan
            # VAE's 4x causal temporal compression.
            num_frames = (int(sample["depth_latent"].shape[1]) - 1) * 4 + 1

        try:
            clip = load_depth_clip(
                _resolve_wide_source(wide_dir, path.stem),
                num_frames=int(num_frames),
                height=int(height),
                width=int(width),
                near=float(near),
                far=float(far),
                encoding=str(info.get("depth_encoding", "disparity")),
            )
            latent = encoders.encode_video(clip)
            if tuple(latent.shape) != tuple(sample["depth_latent"].shape):
                raise ValueError(f"wide latent {tuple(latent.shape)} does not match narrow "
                                 f"{tuple(sample['depth_latent'].shape)}")
            sample["depth_wide_latent"] = latent
            tmp_path = path.with_suffix(".pt.tmp")
            torch.save(sample, tmp_path)
            os.replace(tmp_path, path)
            written += 1
        except Exception as error:  # noqa: BLE001 - one bad clip must not end the pass
            failed += 1
            print(f"[{index + 1}/{len(sample_paths)}] FAILED {path.stem}: {type(error).__name__}: {error}", flush=True)
            continue

    print(f"Done: written={written} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
