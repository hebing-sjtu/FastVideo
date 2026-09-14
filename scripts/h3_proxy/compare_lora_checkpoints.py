#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is the adapter still learning between two checkpoints, or has it stopped moving?

Two samples that look alike have three very different causes, and the checkpoints tell them apart
without generating anything:

* the adapter never left zero -- ``inspect_lora_checkpoint.py`` answers this one;
* it moved, and the later checkpoint is a larger step in the *same* direction, so the run is simply
  early and the schedule is doing what it was told;
* it moved, and the two deltas point in unrelated directions, so the updates are churning and more
  steps will not accumulate into anything.

The discriminator is the angle, not the magnitude. A norm that grew while the direction held means
one consistent descent direction is being integrated; a norm that grew while the direction turned
over means each batch is pulling somewhere else.

What is measured is the effective weight delta ``B @ A``. ``MergedLoRALinear.forward`` computes
``x @ A.T @ B.T`` and scales by ``alpha / rank``, which the H3 recipe pins to 128/128 = 1, so the
delta is exactly ``B @ A``.

It is never materialised. ``B @ A`` for a 5120-wide projection is 100 MB and there are 200 of them,
but every quantity here is a trace of a product of two rank-sized matrices::

    <B1 A1, B2 A2> = tr(A1^T B1^T B2 A2) = tr((B1^T B2)(A2 A1^T))

so each module costs two 128x128 products regardless of how wide the projection is.

Usage::

    python scripts/h3_proxy/compare_lora_checkpoints.py \\
        /data/binghe/h3_proxy/runs/h3_gta_v2/checkpoints/checkpoint-50 \\
        /data/binghe/h3_proxy/runs/h3_gta_v2/checkpoints/checkpoint-150 \\
        --warmup-steps 100 --max-steps 288
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_lora_checkpoint import (  # noqa: E402
    is_role_lora_weight, load_role_lora_tensors, lora_slot, resolve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("early", help="The earlier checkpoint-<step> directory.")
    parser.add_argument("late", help="The later checkpoint-<step> directory.")
    parser.add_argument("--top", type=int, default=8, help="List this many modules by delta norm.")
    parser.add_argument("--warmup-steps",
                        type=int,
                        default=0,
                        help="training.optimizer.lr_warmup_steps. With --max-steps this prints how much of the "
                        "schedule's learning rate each checkpoint had actually integrated, which is what "
                        "sets the expectation for the norm ratio.")
    parser.add_argument("--max-steps", type=int, default=0, help="training.loop.max_train_steps.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.05, help="training.optimizer.min_lr_ratio.")
    return parser.parse_args()


def module_name(key: str) -> str:
    """The owning module, so lora_A and lora_B of one projection pair up."""
    parts = key.split(".")
    cut = len(parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() in {"lora_a", "lora_b"}:
            cut = index
            break
    return ".".join(parts[:cut])


def read_pairs(checkpoint: Path) -> dict[str, tuple]:
    """``{module: (A, B)}`` for every role LoRA projection in one checkpoint."""
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(str(checkpoint / "dcp"))
    keys = [key for key in sorted(reader.read_metadata().state_dict_metadata) if is_role_lora_weight(key)]
    if not keys:
        raise SystemExit(f"{checkpoint} holds no role LoRA weights. Sampling it is base Ref2VA.")
    loaded = load_role_lora_tensors(checkpoint / "dcp", keys)
    halves: dict[str, dict[str, object]] = {}
    for key in keys:
        if key in loaded:
            halves.setdefault(module_name(key), {})[lora_slot(key)] = loaded[key]
    pairs = {name: (half["A"], half["B"]) for name, half in halves.items() if "A" in half and "B" in half}
    if not pairs:
        raise SystemExit(f"{checkpoint} has LoRA tensors but no module with both halves.")
    return pairs


def lr_integral(step: int, warmup: int, max_steps: int, min_ratio: float) -> float:
    """Peak-learning-rate-equivalent steps the schedule had spent by ``step``.

    Linear warmup then cosine to ``min_ratio``, matching ``cosine_with_min_lr``. This is the
    quantity two checkpoints differ by -- not their step numbers -- so it is what a delta norm
    should be compared against.
    """
    total = 0.0
    for index in range(step):
        if warmup and index < warmup:
            total += (index + 1) / warmup
            continue
        span = max(1, max_steps - warmup)
        progress = min(1.0, (index - warmup) / span)
        total += min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return total


def main() -> None:
    args = parse_args()
    import torch

    early_path, late_path = resolve(Path(args.early)), resolve(Path(args.late))
    print(f"early: {early_path.name}\nlate:  {late_path.name}")
    early, late = read_pairs(early_path), read_pairs(late_path)

    shared = sorted(set(early) & set(late))
    print(f"role LoRA modules: early {len(early)}, late {len(late)}, shared {len(shared)}")
    if len(early) != len(late):
        print("  WARNING: the two checkpoints adapt different module sets; only shared ones are compared.")
    if not shared:
        raise SystemExit("The two checkpoints share no LoRA module.")

    rows: list[tuple[str, float, float, float]] = []
    for name in shared:
        a_early, b_early = (tensor.detach().float() for tensor in early[name])
        a_late, b_late = (tensor.detach().float() for tensor in late[name])
        # tr((B1^T B2)(A2 A1^T)) == <B1 A1, B2 A2>, at rank-sized cost.
        norm_early = math.sqrt(max(0.0, torch.trace(b_early.T @ b_early @ (a_early @ a_early.T)).item()))
        norm_late = math.sqrt(max(0.0, torch.trace(b_late.T @ b_late @ (a_late @ a_late.T)).item()))
        inner = torch.trace((b_early.T @ b_late) @ (a_late @ a_early.T)).item()
        rows.append((name, norm_early, norm_late, inner))

    total_early = math.sqrt(sum(row[1]**2 for row in rows))
    total_late = math.sqrt(sum(row[2]**2 for row in rows))
    total_inner = sum(row[3] for row in rows)
    dead_early = sum(1 for row in rows if row[1] < 1e-8)
    dead_late = sum(1 for row in rows if row[2] < 1e-8)

    print(f"\n||B@A|| over all modules: early {total_early:.6g}, late {total_late:.6g}")
    print(f"  modules still at zero: early {dead_early}, late {dead_late}")
    if total_early < 1e-8 or total_late < 1e-8:
        print("\nVERDICT: one checkpoint's adapter is a no-op. Compare with inspect_lora_checkpoint.py; "
              "a zero adapter samples as base Ref2VA no matter how many steps ran.")
        return

    growth = total_late / total_early
    cosine = total_inner / (total_early * total_late)
    # ||D_late - D_early||, from the same three quantities.
    moved = math.sqrt(max(0.0, total_late**2 - 2 * total_inner + total_early**2))
    print(f"  growth ||late||/||early||: {growth:.3f}")
    print(f"  direction cos(early, late): {cosine:.4f}")
    print(f"  ||late - early|| / ||early||: {moved / total_early:.3f}")

    if args.warmup_steps and args.max_steps:
        early_step, late_step = (int(path.name.rsplit("-", 1)[-1]) for path in (early_path, late_path))
        integral_early = lr_integral(early_step, args.warmup_steps, args.max_steps, args.min_lr_ratio)
        integral_late = lr_integral(late_step, args.warmup_steps, args.max_steps, args.min_lr_ratio)
        print(f"\nschedule: warmup {args.warmup_steps}, max {args.max_steps}, min ratio {args.min_lr_ratio}")
        print(f"  peak-rate-equivalent steps: step {early_step} -> {integral_early:.1f}, "
              f"step {late_step} -> {integral_late:.1f}  (ratio {integral_late / max(integral_early, 1e-9):.2f})")
        if early_step < args.warmup_steps:
            print(f"  NOTE: step {early_step} is mid-warmup, so its learning rate was "
                  f"{early_step / args.warmup_steps:.0%} of peak and its integral is small. Two checkpoints "
                  "inside or straddling warmup are much closer in training than their step numbers suggest.")

    print()
    if cosine > 0.9:
        print("VERDICT: the later delta is a larger step in the same direction "
              f"(cos {cosine:.3f}). The updates are accumulating, so identical-looking samples mean the "
              "magnitude has not yet reached the point of changing the output -- a question of steps and "
              "learning rate, not of a broken signal.")
    elif cosine > 0.5:
        print(f"VERDICT: the direction is holding but drifting (cos {cosine:.3f}). Learning is directed; "
              "expect slower accumulation than the norm growth alone suggests.")
    else:
        print(f"VERDICT: the two deltas are nearly unrelated (cos {cosine:.3f}). The updates are churning "
              "rather than accumulating, which more steps will not fix. Suspect the learning rate, the "
              "batch, or a conditioning signal the loss cannot attribute.")

    rows.sort(key=lambda row: row[2], reverse=True)
    print(f"\ntop {args.top} modules by late ||B@A||:")
    for name, norm_early, norm_late, inner in rows[:args.top]:
        angle = inner / (norm_early * norm_late) if norm_early > 1e-12 and norm_late > 1e-12 else float("nan")
        print(f"  {name}\n    early {norm_early:.4g}  late {norm_late:.4g}  "
              f"growth {norm_late / max(norm_early, 1e-12):.2f}  cos {angle:.3f}")


if __name__ == "__main__":
    main()
