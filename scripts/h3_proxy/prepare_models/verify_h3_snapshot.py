#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a local MiniMax-H3 snapshot against the published file list.

Catches the two failure modes that otherwise surface only after a 32B text encoder has loaded:
an interrupted download missing whole components, and a truncated copy where every file is present
but some are short. Both are cheap to detect here and expensive to detect later.

Profiles select which components must be complete:

``ref2va``
    What proxy-to-video training and the encoder actually read: ``transformer_ref``, ``vae``,
    ``text_encoder``, ``tokenizer``, ``processor``, ``audio_vae``, both schedulers, and the
    manifests. ``transformer/`` is checked for existence only -- ``model_index.json`` declares it,
    so :func:`fastvideo.utils.verify_model_config_and_directory` requires the directory, but the
    Ref2VA path never loads its weights. 61.7 GB of shards for an ``os.path.exists`` call is not
    worth downloading.

``full``
    Every component the root manifest declares, weights included.

Neither profile covers ``Ref2VA/`` or ``FL2VA/``. Those are the upstream MiniMax packaging, not the
diffusers layout: they carry ``video_vae/`` rather than ``vae/`` and name their classes
``MiniMaxH3DiTModel``, so FastVideo cannot load them. They are also complete duplicates of the root
components, 134 GB each.

Usage::

    # after downloading, and again after a bucket round-trip
    python scripts/h3_proxy/prepare_models/verify_h3_snapshot.py --path /workspace/models/MiniMax-H3

    # on a node without Hub access, reuse a manifest fetched earlier
    python .../verify_h3_snapshot.py --path /data/models/MiniMax-H3 \\
        --manifest-cache /workspace/h3_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

REPO_DEFAULT = "MiniMaxAI/MiniMax-H3"

# Weight-bearing components each profile requires to be byte-complete.
PROFILE_COMPONENTS = {
    "ref2va": (
        "model_index.json",
        "modular_model_index.json",
        "tokenizer",
        "processor",
        "scheduler",
        "audio_scheduler",
        "text_encoder",
        "vae",
        "audio_vae",
        "transformer_ref",
    ),
    "full": (
        "model_index.json",
        "modular_model_index.json",
        "tokenizer",
        "processor",
        "scheduler",
        "audio_scheduler",
        "text_encoder",
        "vae",
        "audio_vae",
        "transformer_ref",
        "transformer",
    ),
}
# Declared in model_index.json, so the directory must exist even when its weights are not needed.
PLACEHOLDER_ONLY = {"ref2va": ("transformer", ), "full": ()}
# Upstream packaging that cannot be loaded and duplicates the root components.
UNUSABLE = ("Ref2VA", "FL2VA")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True, help="Local snapshot directory.")
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--profile", choices=tuple(PROFILE_COMPONENTS), default="ref2va")
    p.add_argument("--manifest-cache",
                   default=None,
                   help="Read the published file list from here if present, else fetch and write it.")
    p.add_argument("--fix-placeholders",
                   action="store_true",
                   help="Create any placeholder-only directory, with a .keep so it survives object storage.")
    return p.parse_args()


def load_manifest(repo: str, cache: str | None) -> dict[str, int]:
    """Map published path -> size in bytes."""
    if cache and os.path.isfile(cache):
        with open(cache, encoding="utf-8") as handle:
            return {str(k): int(v) for k, v in json.load(handle).items()}
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
        payload = json.load(response)
    sizes = {str(f["rfilename"]): int(f.get("size") or 0) for f in payload.get("siblings", [])}
    if not sizes:
        raise SystemExit(f"{repo} reported no files; cannot verify.")
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(sizes, handle)
        print(f"Cached published file list -> {cache}")
    return sizes


def component_of(path: str) -> str:
    head, _, tail = path.partition("/")
    return head if tail else path


def main() -> int:
    args = parse_args()
    root = Path(args.path).expanduser()
    if not root.is_dir():
        raise SystemExit(f"--path is not a directory: {root}")
    required = PROFILE_COMPONENTS[args.profile]
    placeholders = PLACEHOLDER_ONLY[args.profile]
    sizes = load_manifest(args.repo, args.manifest_cache)

    expected: dict[str, list[tuple[str, int]]] = {name: [] for name in required}
    for path, size in sizes.items():
        component = component_of(path)
        if component in expected:
            expected[component].append((path, size))

    print(f"Profile '{args.profile}' against {root}\n")
    print(f"{'component':26} {'files':>11}  {'GB':>7}  status")
    missing: list[str] = []
    truncated: list[str] = []
    total_bytes = 0
    for name in required:
        entries = sorted(expected[name])
        if not entries:
            raise SystemExit(f"The published manifest lists nothing under {name!r}; --repo may be wrong.")
        present = 0
        component_bytes = 0
        for path, size in entries:
            local = root / path
            try:
                actual = local.stat().st_size
            except OSError:
                missing.append(path)
                continue
            present += 1
            component_bytes += actual
            # Exact match, not a threshold: a short file is a truncated transfer, and the only
            # reason a published size would differ otherwise is a revision mismatch.
            if actual != size:
                truncated.append(f"{path} ({actual} bytes, published {size})")
        total_bytes += component_bytes
        status = "OK" if present == len(entries) else "INCOMPLETE"
        print(f"{name:26} {present:5d}/{len(entries):<5d}  {component_bytes / 2**30:7.2f}  {status}")

    print(f"{'TOTAL':26} {'':11}  {total_bytes / 2**30:7.2f}")
    print()

    for name in placeholders:
        directory = root / name
        if directory.is_dir():
            print(f"placeholder {name}/: OK")
            continue
        if args.fix_placeholders:
            directory.mkdir(parents=True, exist_ok=True)
            # Object storage has no empty directories, so a bare mkdir would not survive a bucket
            # round-trip. The .keep gives the prefix an object to exist as.
            (directory / ".keep").write_text("", encoding="utf-8")
            print(f"placeholder {name}/: created (with .keep)")
        else:
            print(f"placeholder {name}/: MISSING -- rerun with --fix-placeholders, or "
                  f"`mkdir -p {directory} && touch {directory}/.keep`")
            missing.append(f"{name}/ (directory placeholder)")

    stale = [name for name in UNUSABLE if (root / name).is_dir()]
    if stale:
        print(f"\nUnusable upstream packaging still present: {', '.join(stale)}. "
              f"These cannot be loaded by FastVideo and are ~134 GB each; `rm -rf` them.")

    if missing:
        print(f"\nMISSING {len(missing)} files:")
        for path in missing[:15]:
            print(f"  {path}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")
    if truncated:
        print(f"\nTRUNCATED {len(truncated)} files:")
        for line in truncated[:15]:
            print(f"  {line}")
        if len(truncated) > 15:
            print(f"  ... and {len(truncated) - 15} more")

    if missing or truncated:
        print("\nFAIL: snapshot is not usable. Re-run the download; it skips complete files.")
        return 1
    print("\nPASS: every file required by this profile is present at its published size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
