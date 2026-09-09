#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Did this DCP checkpoint actually move LoRA off initialization?

At init, LoRA-B is zeros, so the adapted model equals the base. If B is still
near zero at step 250, qualitative failure matching step 0 is expected.

DCP also stores Adam hyperparams under ``optimizers.*.lora_A.amsgrad`` etc.
Those are not weights. This script only measures ``roles.*`` LoRA matrices.

Usage (one CPU process, no torchrun)::

    python scripts/h3_proxy/inspect_lora_checkpoint.py \\
        /data/binghe/h3_proxy/runs/h3_proxy_bd_lora_abot_cwm/checkpoints/checkpoint-250
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROLE_PREFIX = "roles."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="checkpoint-<step> directory (contains dcp/)")
    parser.add_argument(
        "--load-norms",
        action="store_true",
        default=True,
        help="Load role LoRA tensors and print norms (default: on).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip tensor reads; print key counts and shapes only.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "dcp":
        path = path.parent
    if not (path / "dcp").is_dir():
        raise SystemExit(f"No dcp/ under {path}")
    return path


def lora_slot(key: str) -> str | None:
    """Return 'A' or 'B' if this FQN is a LoRA matrix, else None.

    Matches FastVideo ``lora_A`` / ``lora_B`` parameters and PEFT
    ``lora_A.weight``. Does not match optimizer leaves such as
    ``lora_A.amsgrad`` or ``lora_A.exp_avg``.
    """
    parts = key.split(".")
    leaf = parts[-1]
    if leaf in {"lora_A", "lora_a"}:
        return "A"
    if leaf in {"lora_B", "lora_b"}:
        return "B"
    if leaf == "weight" and len(parts) >= 2:
        parent = parts[-2]
        if parent in {"lora_A", "lora_a"}:
            return "A"
        if parent in {"lora_B", "lora_b"}:
            return "B"
    return None


def is_role_lora_weight(key: str) -> bool:
    return key.startswith(ROLE_PREFIX) and lora_slot(key) is not None


def top_level(key: str) -> str:
    return key.split(".", 1)[0]


def tensor_size(meta: Any) -> tuple[int, ...] | None:
    size = getattr(meta, "size", None)
    if size is None:
        return None
    return tuple(int(dim) for dim in size)


def load_role_lora_tensors(dcp_dir: Path, keys: list[str]) -> dict[str, Any]:
    """Read only the requested DCP tensors on one process (no flatten-to-disk)."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    class _Selected(DefaultLoadPlanner):
        def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
            assert metadata is not None
            built: dict[str, Any] = {}
            for key in keys:
                meta = metadata.state_dict_metadata[key]
                if isinstance(meta, TensorStorageMetadata):
                    built[key] = torch.empty(meta.size, device="cpu", dtype=meta.properties.dtype)
            super().set_up_planner(built, metadata, is_coordinator)

    planner = _Selected()
    holder: dict[str, Any] = {}
    _load_state_dict(
        holder,
        storage_reader=dcp.FileSystemReader(str(dcp_dir)),
        planner=planner,
        no_dist=True,
    )
    loaded = getattr(planner, "state_dict", None) or holder
    return {key: value for key, value in loaded.items() if key in keys and hasattr(value, "norm")}


def report(name: str, items: list[tuple[str, Any]]) -> None:
    import torch

    if not items:
        print(f"{name}: n=0")
        return
    norms = torch.tensor([tensor.detach().float().norm().item() for _, tensor in items])
    absmax = torch.tensor([tensor.detach().float().abs().max().item() for _, tensor in items])
    print(f"{name}: n={len(items)}  mean||.||={norms.mean():.6g}  "
          f"median||.||={norms.median():.6g}  max|w|={absmax.max():.6g}  "
          f"frac_near0={(absmax < 1e-8).float().mean():.3f}")


def main() -> None:
    args = parse_args()
    ckpt = resolve(Path(args.checkpoint))
    meta_file = ckpt / "metadata.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text())
        print(f"metadata.step = {meta.get('step')}")
    else:
        print("metadata.json missing")

    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(str(ckpt / "dcp"))
    stored = reader.read_metadata().state_dict_metadata
    keys = sorted(stored)

    prefixes: dict[str, int] = {}
    for key in keys:
        prefixes[top_level(key)] = prefixes.get(top_level(key), 0) + 1
    print("dcp tensors:", len(keys))
    print("top-level:", " ".join(f"{name}={count}" for name, count in sorted(prefixes.items())))

    role_lora = [key for key in keys if is_role_lora_weight(key)]
    role_a = [key for key in role_lora if lora_slot(key) == "A"]
    role_b = [key for key in role_lora if lora_slot(key) == "B"]
    optimizer_lora_leaves = [
        key for key in keys if key.startswith("optimizers.") and "lora_" in key.lower()
    ]
    print(f"role LoRA weights: {len(role_lora)}  (A={len(role_a)}, B={len(role_b)})")
    print(f"optimizer keys that mention lora (ignored): {len(optimizer_lora_leaves)}")

    if not role_lora:
        print("NO role LoRA WEIGHTS. Sampling this checkpoint is base Ref2VA.")
        print("first 20 keys:")
        for key in keys[:20]:
            print(f"  {key}")
        raise SystemExit(2)

    print("sample role LoRA keys:")
    for key in role_a[:4] + role_b[:4]:
        size = tensor_size(stored[key])
        print(f"  {key}  shape={size}")

    if args.metadata_only or not args.load_norms:
        print("Re-run without --metadata-only to measure ||lora_B||.")
        return

    print("loading role LoRA tensors only (CPU, no flatten)...")
    loaded = load_role_lora_tensors(ckpt / "dcp", role_lora)
    a_tensors = [(key, loaded[key]) for key in role_a if key in loaded]
    b_tensors = [(key, loaded[key]) for key in role_b if key in loaded]
    if len(b_tensors) != len(role_b):
        print(f"loaded {len(loaded)}/{len(role_lora)} role LoRA tensors "
              f"(B {len(b_tensors)}/{len(role_b)}). Planner may need a newer PyTorch.")
        raise SystemExit(3)

    report("lora_A", a_tensors)
    report("lora_B", b_tensors)
    if all(tensor.detach().float().abs().max().item() < 1e-7 for _, tensor in b_tensors):
        print("VERDICT: lora_B is still ~0. The adapter is a no-op; output matches step 0 / base.")
        raise SystemExit(4)
    print("VERDICT: lora_B has left zero. Weights moved. Same failure mode is the model, not a dead adapter.")


if __name__ == "__main__":
    main()
