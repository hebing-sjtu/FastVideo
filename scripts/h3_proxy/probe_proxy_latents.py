#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How far outside the VAE's natural latent range does a DUV encoding land?

A proxy is a synthetic frame pushed through a VAE trained on photographic video. The transformer
then has to read the result with attention weights learned from natural latents. The further the
proxy latents sit from that distribution, the more of a from-scratch LoRA's capacity goes into
re-learning to *read* the conditioning rather than into following it -- which is what an adapter
concentrated in early attention blocks, with orthogonal increments, looks like.

Every cache carries both tensors, so this is answerable with no GPU and no re-encoding:
``vae_latent`` is the real clip and is the in-distribution reference, ``proxy_latent`` is the DUV
render. Both went through the same VAE in the same encode, so the comparison is internal to each
cache and needs no cross-cache calibration.

That matters, because two caches built by different commands differ in more than their DUV
convention -- target canvas, anchor canvas, and regenerated captions all move. Per-channel latent
*distribution* statistics do not depend on the spatial dimensions, so comparing each cache's proxy
against its own natural-video reference isolates the encoding. Absolute loss on two such caches
would not.

Reported per cache, over the channel dimension:

* ``shift``  |mean(proxy) - mean(real)| / std(real), how far off-centre the proxy sits
* ``spread`` std(proxy) / std(real), whether it is flatter or more extreme than natural video
* ``tail``   the fraction of proxy values outside the real latents' 0.1-99.9 percentile range. A
  proxy drawn from the same distribution as the reference would sit near 0.002 by construction, but
  the quantiles come from a finite sample, so read it by comparing the two caches rather than
  against that figure.

Usage::

    python scripts/h3_proxy/probe_proxy_latents.py \\
        /data/binghe/h3_proxy/cache/gta_v2_train \\
        /data/binghe/h3_proxy/cache/gta_v2_cwm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", nargs="+", help="Cache directories holding <name>.pt.")
    parser.add_argument("--sample", type=int, default=24, help="How many clips to open per cache.")
    parser.add_argument("--per-channel", type=int, default=0, help="Also list this many worst channels.")
    return parser.parse_args()


def pick(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or count >= len(paths):
        return paths
    return paths[::max(1, len(paths) // count)][:count]


def channel_stats(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std per channel, flattening time and space."""
    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
    return flat.mean(dim=1), flat.std(dim=1)


def describe(directory: Path, sample: int, per_channel: int) -> None:
    paths = pick(sorted(directory.glob("*.pt")), sample)
    print(f"\n{directory}")
    if not paths:
        print("  no .pt files")
        return

    shifts, spreads, tails = [], [], []
    for path in paths:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        real, proxy = blob.get("vae_latent"), blob.get("proxy_latent")
        if real is None or proxy is None:
            continue
        real_mean, real_std = channel_stats(real)
        proxy_mean, proxy_std = channel_stats(proxy)
        # A channel the VAE leaves constant on real video has no scale to compare against, so it
        # cannot say anything about the proxy and is dropped rather than dividing by ~0.
        live = real_std > 1e-6
        if not bool(live.any()):
            continue
        shifts.append(((proxy_mean - real_mean).abs() / real_std)[live])
        spreads.append((proxy_std / real_std)[live])

        flat_real = real.detach().float().reshape(real.shape[0], -1)
        low = torch.quantile(flat_real, 0.001, dim=1)
        high = torch.quantile(flat_real, 0.999, dim=1)
        flat_proxy = proxy.detach().float().reshape(proxy.shape[0], -1)
        outside = ((flat_proxy < low[:, None]) | (flat_proxy > high[:, None])).float().mean(dim=1)
        tails.append(outside[live])

    if not shifts:
        print("  no clip carried both vae_latent and proxy_latent")
        return

    shift = torch.stack(shifts).mean(dim=0)
    spread = torch.stack(spreads).mean(dim=0)
    tail = torch.stack(tails).mean(dim=0)
    print(f"  {len(shifts)} clips, {shift.numel()} live channels")
    print(f"  shift   mean {shift.mean():.3f}  max {shift.max():.3f}   (in units of the real latents' std)")
    print(f"  spread  mean {spread.mean():.3f}  min {spread.min():.3f}  max {spread.max():.3f}   (1.0 = natural)")
    print(f"  tail    mean {tail.mean():.4f}  max {tail.max():.4f}   (compare between caches, not to a constant)")

    if per_channel:
        # Ranking by one metric hides the others' outliers. A channel can sit dead centre and still
        # have half its values off the end of the range, which is the case worth seeing.
        for label, metric in (("shift", shift), ("tail", tail), ("spread", (spread - 1.0).abs())):
            order = torch.argsort(metric, descending=True)[:per_channel]
            print(f"  worst {per_channel} channels by {label}:")
            for index in order.tolist():
                print(f"    channel {index:3d}  shift {shift[index]:.3f}  spread {spread[index]:.3f}  "
                      f"tail {tail[index]:.4f}")


def main() -> None:
    args = parse_args()
    for name in args.cache:
        describe(Path(name).expanduser(), args.sample, args.per_channel)
    print("\nThe cache whose proxy sits closer to its own vae_latent is the encoding the base model\n"
          "already knows how to read. The larger shift, spread further from 1, and larger tail mean\n"
          "the transformer receives conditioning outside the range its attention was trained on, and\n"
          "a from-scratch LoRA pays for that before it can learn anything about following the proxy.\n"
          "This ranks the encodings; it does not measure the loss. If the two come out close, the\n"
          "paired forward-loss probe is the one that settles it.")


if __name__ == "__main__":
    main()
