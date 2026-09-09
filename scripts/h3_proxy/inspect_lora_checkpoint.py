#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Did this DCP checkpoint actually move LoRA off initialization?

At init, LoRA-B is zeros, so the adapted model equals the base. If B is still
near zero at step 250, qualitative failure matching step 0 is expected.

Usage (one CPU process, no torchrun)::

    python scripts/h3_proxy/inspect_lora_checkpoint.py \\
        /data/binghe/h3_proxy/runs/h3_proxy_bd_lora_abot_cwm/checkpoints/checkpoint-250
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="checkpoint-<step> directory (contains dcp/)")
    parser.add_argument("--load-norms", action="store_true", help="Materialize LoRA tensors and print norms. Uses RAM.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "dcp":
        path = path.parent
    if not (path / "dcp").is_dir():
        raise SystemExit(f"No dcp/ under {path}")
    return path


def main() -> None:
    args = parse_args()
    ckpt = resolve(Path(args.checkpoint))
    meta_file = ckpt / "metadata.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text())
        print(f"metadata.step = {meta.get('step')}")
    else:
        print("metadata.json missing")

    reader = dcp.FileSystemReader(str(ckpt / "dcp"))
    stored = reader.read_metadata().state_dict_metadata
    keys = sorted(stored)
    lora = [key for key in keys if "lora" in key.lower()]
    lora_a = [key for key in lora if "lora_A" in key or "lora_a" in key]
    lora_b = [key for key in lora if "lora_B" in key or "lora_b" in key]
    print(f"dcp tensors: {len(keys)}")
    print(f"lora keys:   {len(lora)}  (A={len(lora_a)}, B={len(lora_b)})")
    if not lora:
        print("NO LoRA KEYS. This checkpoint cannot be a trained adapter; sampling is base Ref2VA.")
        print("first 20 keys:")
        for key in keys[:20]:
            print(f"  {key}")
        raise SystemExit(2)
    print("sample LoRA keys:")
    for key in lora[:8]:
        print(f"  {key}")

    if not args.load_norms:
        print("Re-run with --load-norms to measure ||lora_B|| (near-zero means still at init).")
        return

    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    flattened = ckpt / "_inspect_flat.pt"
    print(f"flattening DCP -> {flattened} (one-time, large)")
    dcp_to_torch_save(str(ckpt / "dcp"), str(flattened))
    payload = torch.load(flattened, map_location="cpu", weights_only=False)

    def tensors_matching(needles: tuple[str, ...]) -> list[tuple[str, torch.Tensor]]:
        found = []

        def walk(prefix: str, value: object) -> None:
            if torch.is_tensor(value):
                if any(needle in prefix for needle in needles):
                    found.append((prefix, value.detach().float().cpu()))
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(f"{prefix}.{key}" if prefix else str(key), child)

        walk("", payload)
        return found

    b_tensors = tensors_matching(("lora_B", "lora_b"))
    a_tensors = tensors_matching(("lora_A", "lora_a"))
    if not b_tensors:
        print("Flattened state has LoRA key names in metadata but no lora_B tensors walked.")
        raise SystemExit(3)

    def report(name: str, items: list[tuple[str, torch.Tensor]]) -> None:
        norms = torch.tensor([tensor.norm().item() for _, tensor in items])
        absmax = torch.tensor([tensor.abs().max().item() for _, tensor in items])
        print(f"{name}: n={len(items)}  mean||.||={norms.mean():.6g}  "
              f"median||.||={norms.median():.6g}  max|w|={absmax.max():.6g}  "
              f"frac_near0={(absmax < 1e-8).float().mean():.3f}")

    report("lora_A", a_tensors)
    report("lora_B", b_tensors)
    flattened.unlink(missing_ok=True)
    if all(tensor.abs().max().item() < 1e-7 for _, tensor in b_tensors):
        print("VERDICT: lora_B is still ~0. The adapter is a no-op; output matches step 0 / base.")
        raise SystemExit(4)
    print("VERDICT: lora_B has left zero. Weights moved. Same failure mode is the model, not a dead adapter.")


if __name__ == "__main__":
    main()
