#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What geometry was this proxy cache written at, and which flags must consume it?

A cache is a directory of ``.pt`` files whose geometry is fixed at encode time. Nothing at train or
sample time re-derives it, so ``training.data.num_height``, ``training.data.num_width`` and
``callbacks.validation.anchor_short_edge`` have to be set to the values the encoder used. Set them
to anything else and the run fails on a shape mismatch, or worse, silently conditions on a canvas
the cache never contained.

The encoder records the target canvas in ``info["pixel_size"]`` but not the proxy grid or the anchor
short edge, so those two come back from the latent shapes: every latent axis is the pixel axis over
16, being the VAE's 8x spatial compression and the transformer's 2x2 patch.

Reading every ``.pt`` would mean pulling the text embeddings too -- tens of megabytes per clip -- so
only a stride-sampled handful is opened unless ``--all`` is given. That is enough to catch a cache
built by two different commands, which is the failure this is for.

Usage::

    python scripts/h3_proxy/describe_cache.py /data/binghe/h3_proxy/cache/gta_v2_train

    # and confirm no clip went missing on the way in
    python scripts/h3_proxy/describe_cache.py /data/binghe/h3_proxy/cache/gta_v2_train \\
        --manifest /data/binghe/h3_proxy/gta_v2_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

# VAE spatial compression 8, times the transformer's 2x2 patch.
LATENT_TO_PIXEL = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", nargs="+", help="Cache directories holding <name>.pt.")
    parser.add_argument("--manifest", help="Encode manifest to reconcile names against.")
    parser.add_argument("--sample", type=int, default=8, help="How many .pt to open per cache.")
    parser.add_argument("--all", action="store_true", help="Open every .pt. Slow; reads the embeddings too.")
    parser.add_argument("--list-missing", type=int, default=20, help="How many missing names to print.")
    return parser.parse_args()


def pick(paths: list[Path], count: int, take_all: bool) -> list[Path]:
    """A stride-sampled subset, so a cache finished by a second command is still caught."""
    if take_all or count <= 0 or count >= len(paths):
        return paths
    return paths[::max(1, len(paths) // count)][:count]


def geometry(sample: dict[str, Any]) -> dict[str, Any]:
    """The encode-time flags this sample implies."""
    found: dict[str, Any] = {}
    info = sample.get("info") or {}
    if isinstance(info, dict):
        size = info.get("pixel_size")
        if size is not None:
            found["target"] = (int(size[0]), int(size[1]))
        if info.get("num_frames") is not None:
            found["num_frames"] = int(info["num_frames"])
        if info.get("cwm_system") is not None:
            found["cwm_system"] = str(info["cwm_system"])
    target = sample.get("vae_latent")
    if target is not None and "target" not in found:
        found["target"] = (target.shape[-2] * LATENT_TO_PIXEL, target.shape[-1] * LATENT_TO_PIXEL)
    if target is not None:
        found["latent_frames"] = int(target.shape[1])
    proxy = sample.get("proxy_latent")
    if proxy is not None:
        found["proxy"] = (proxy.shape[-2] * LATENT_TO_PIXEL, proxy.shape[-1] * LATENT_TO_PIXEL)
    anchor = sample.get("anchor_latent")
    if anchor is not None:
        pixels = (anchor.shape[-2] * LATENT_TO_PIXEL, anchor.shape[-1] * LATENT_TO_PIXEL)
        found["anchor_canvas"] = pixels
        found["anchor_short_edge"] = min(pixels)
    found["camera"] = "extrinsics" in sample
    return found


# The anchor keeps its source aspect at a fixed short edge, so its long edge legitimately differs
# per clip. Every other axis is set by a flag and must come back single-valued.
MAY_VARY = frozenset({"anchor_canvas"})


def render(label: str, values: Counter) -> str:
    """One value prints bare; more than one is a mixed cache unless the axis is free to vary."""
    if len(values) == 1:
        return f"  {label}: {next(iter(values))}"
    listed = ", ".join(f"{value} x{count}" for value, count in values.most_common())
    prefix = "varies" if label in MAY_VARY else "MIXED --"
    return f"  {label}: {prefix} {listed}"


def manifest_names(path: Path) -> list[str]:
    names = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            name = row.get("name") or Path(str(row["target"])).stem
            names.append(str(name).replace("/", "__"))
    return names


def reconcile(directory: Path, manifest: Path, present: set[str], show: int) -> None:
    wanted = manifest_names(manifest)
    missing = [name for name in wanted if name not in present]
    extra = sorted(present - set(wanted))
    print(f"  manifest {manifest}: {len(wanted)} rows, {len(wanted) - len(missing)} cached")
    if missing:
        # A shard that died, or a sharded encode run with the wrong --num-shards. The loader scans
        # the directory rather than the manifest, so neither shows up as an error at train time.
        print(f"  MISSING {len(missing)}: {', '.join(missing[:show])}"
              f"{' ...' if len(missing) > show else ''}")
    if extra:
        print(f"  not in manifest {len(extra)}: {', '.join(extra[:show])}"
              f"{' ...' if len(extra) > show else ''}")


def describe(directory: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    paths = sorted(directory.glob("*.pt"))
    print(f"\n{directory}")
    if not paths:
        print("  no .pt files")
        return None
    chosen = pick(paths, args.sample, args.all)
    print(f"  {len(paths)} clips, {len(chosen)} opened")

    collected: dict[str, Counter] = {}
    for path in chosen:
        for key, value in geometry(torch.load(path, map_location="cpu", weights_only=False)).items():
            collected.setdefault(key, Counter())[value] += 1

    for key in ("num_frames", "latent_frames", "target", "proxy", "anchor_canvas", "anchor_short_edge",
                "cwm_system", "camera"):
        if key in collected:
            print(render(key, collected[key]))

    if args.manifest:
        reconcile(directory, Path(args.manifest), {path.stem for path in paths}, args.list_missing)
    return {key: values for key, values in collected.items()}


def print_flags(collected: dict[str, Any]) -> None:
    """The flags a train or eval job must carry to consume this cache."""
    target = collected.get("target")
    anchor = collected.get("anchor_short_edge")
    if not target or not anchor:
        return
    if len(target) > 1 or len(anchor) > 1:
        print("\n  cache is mixed, so no single set of flags consumes it -- re-encode before training")
        return
    height, width = next(iter(target))
    print("\n  consume it with:")
    print(f"    --training.data.num_height {height} --training.data.num_width {width} \\")
    print(f"    --callbacks.validation.anchor_short_edge {next(iter(anchor))}")

    proxy = collected.get("proxy")
    frames = collected.get("num_frames")
    if proxy and frames and len(proxy) == 1 and len(frames) == 1:
        proxy_h, proxy_w = next(iter(proxy))
        print("\n  reproduce it with:")
        print(f"    --num-frames {next(iter(frames))} --height {height} --width {width} \\")
        print(f"    --proxy-height {proxy_h} --proxy-width {proxy_w} \\")
        print(f"    --anchor-short-edge {next(iter(anchor))}")


def main() -> None:
    args = parse_args()
    for name in args.cache:
        collected = describe(Path(name).expanduser(), args)
        if collected is not None:
            print_flags(collected)


if __name__ == "__main__":
    main()
