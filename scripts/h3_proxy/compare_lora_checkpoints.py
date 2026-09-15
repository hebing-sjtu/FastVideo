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

The discriminator is the angle between the *increment* and the accumulated delta. ``cos(early, late)``
is not it: whenever the norm barely grew, that cosine is pinned near 1 by construction and cannot
separate "kept pushing the same way" from "stopped moving". ``cos(late - early, early)`` can.

Magnitude still decides whether any of it matters, which is what ``--base-snapshot`` is for. A delta
of norm 3 against a backbone of norm 10000 changes nothing observable no matter how well aimed it is.

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
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_lora_checkpoint import (  # noqa: E402
    checkpoint_inventory, is_role_lora_weight, load_role_lora_tensors, lora_slot, resolve,
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
    parser.add_argument("--global-batch",
                        type=int,
                        default=0,
                        help="(num_gpus / sp_size) * train_batch_size * gradient_accumulation_steps. Required to "
                        "compare two runs whose batch differs: the per-step drift-to-noise carries a factor of "
                        "sqrt(batch), so without this a run gets credit for its batch size alone.")
    parser.add_argument("--peak-lr",
                        type=float,
                        default=0.0,
                        help="training.optimizer.learning_rate. Reports how much of the travel the schedule's "
                        "learning rate permits has been used, which is what separates 'the gradient disagrees "
                        "with itself' from 'the learning rate is the ceiling'.")
    parser.add_argument("--base-snapshot",
                        default="",
                        help="MiniMax-H3 snapshot, e.g. /data/models/MiniMax-H3. Reports ||B@A|| against the "
                        "norm of the base weight it is added to, which is what decides whether a converged "
                        "adapter is even large enough to change the output. Only matched tensors are read.")
    return parser.parse_args()


def base_weight_index(snapshot: Path) -> dict[str, Path]:
    """Base parameter name -> the file holding it, for the Ref2VA partition.

    ``transformer_ref/`` is the partition this stage trains. ``transformer/`` is T2VA/FL2VA and
    would quietly supply the wrong denominator.
    """
    directory = snapshot / "transformer_ref"
    if not directory.is_dir():
        raise SystemExit(f"{directory} does not exist. --base-snapshot wants the snapshot root, the "
                         "directory holding transformer_ref/.")
    for index_name in ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"):
        index = directory / index_name
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
            if weight_map:
                return {name: directory / shard for name, shard in weight_map.items()}
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No safetensors and no index under {directory}.")
    from safetensors import safe_open

    mapping: dict[str, Path] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle:
                mapping[name] = shard
    return mapping


def base_key_for(module: str, index: dict[str, Path]) -> str | None:
    """The base weight a LoRA module adapts, or None when the names do not line up.

    A training FQN carries wrappers the snapshot has no idea about -- a ``roles.student.transformer.``
    prefix and a ``.checkpointed`` segment from the gradient-checkpointing wrapper -- so the longest
    matching suffix is what identifies the parameter.

    ``.0.weight`` is tried alongside ``.weight`` because ``attn.to_out`` is an ``nn.ModuleList``
    whose Linear is ``to_out.0``, while LoRA wraps it under the list's own name. Without this, 50 of
    the 200 modules -- every ``to_out`` -- silently drop out of the ratio.
    """
    parts = [part for part in module.split(".") if part != "checkpointed"]
    for start in range(len(parts)):
        stem = ".".join(parts[start:])
        for candidate in (stem + ".weight", stem + ".0.weight"):
            if candidate in index:
                return candidate
    tail = ".".join(parts[-3:])
    for suffix in (tail + ".weight", tail + ".0.weight"):
        matches = [name for name in index if name.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
    return None


def base_norms(modules: list[str], snapshot: Path) -> tuple[dict[str, float], list[str]]:
    """Frobenius norm of each module's base weight, and the modules that did not match."""
    from safetensors import safe_open

    index = base_weight_index(snapshot)
    print(f"base snapshot: {len(index)} tensors under {snapshot / 'transformer_ref'}")
    wanted: dict[str, str] = {}
    unmatched: list[str] = []
    for module in modules:
        key = base_key_for(module, index)
        if key is None:
            unmatched.append(module)
        else:
            wanted[module] = key

    # Grouped by shard so each file is opened once rather than once per tensor.
    by_shard: dict[Path, set[str]] = {}
    for key in wanted.values():
        by_shard.setdefault(index[key], set()).add(key)
    cache: dict[str, float] = {}
    for shard, keys in by_shard.items():
        with safe_open(str(shard), framework="pt") as handle:
            for key in sorted(keys):
                cache[key] = handle.get_tensor(key).detach().float().norm().item()
    return {module: cache[key] for module, key in wanted.items()}, unmatched


def module_name(key: str) -> str:
    """The owning module, so lora_A and lora_B of one projection pair up."""
    parts = key.split(".")
    cut = len(parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() in {"lora_a", "lora_b"}:
            cut = index
            break
    return ".".join(parts[:cut])


def require_loadable(checkpoint: Path) -> None:
    """Fail before torch does, naming the checkpoints that can actually be read.

    An interrupted save leaves ``dcp/`` in place without ``.metadata``, which ``resolve`` accepts
    and DCP then reports as a bare ``FileNotFoundError`` on a path nobody asked for. Since the usual
    reason to be here is that a run was interrupted, the inventory is the useful part.
    """
    if (checkpoint / "dcp" / ".metadata").is_file():
        return
    print(f"{checkpoint.name} has a dcp/ directory but no dcp/.metadata, so its save did not finish.")
    print(checkpoint_inventory(checkpoint.parent))
    raise SystemExit(2)


def read_pairs(checkpoint: Path) -> dict[str, tuple]:
    """``{module: (A, B)}`` for every role LoRA projection in one checkpoint."""
    import torch.distributed.checkpoint as dcp

    require_loadable(checkpoint)
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


def report_lr_ceiling(late: dict[str, tuple], integral: float, peak_lr: float) -> None:
    """Which constraint is binding: the learning rate, or the gradient agreeing with itself.

    AdamW's per-parameter step is ``lr * m/sqrt(v)``, and that ratio sits near 1 wherever the
    gradient keeps its sign, so the learning rate integrated over the schedule is a ceiling on how
    far any single weight can travel. LoRA-B starts at exactly zero, which makes its current value
    the distance travelled and lets it be read against that ceiling directly.

    A run near the ceiling cannot be helped by more steps at this learning rate -- only by a larger
    integral. A run far below it is held back by increments that cancel, and more integral would buy
    proportionally less.
    """
    ceiling = peak_lr * integral
    if ceiling <= 0:
        return
    b_tensors = [pair[1].detach().float() for pair in late.values()]
    absmax = max(float(tensor.abs().max()) for tensor in b_tensors)
    elements = sum(tensor.numel() for tensor in b_tensors)
    rms = math.sqrt(sum(float(tensor.pow(2).sum()) for tensor in b_tensors) / max(elements, 1))
    print(f"\nAdamW travel ceiling: peak lr {peak_lr:.3g} over {integral:.1f} equivalent steps "
          f"= {ceiling:.3g} per weight")
    print(f"  lora_B max|w| {absmax:.3g}  ({absmax / ceiling:.0%} of ceiling)")
    print(f"  lora_B rms    {rms:.3g}  ({rms / ceiling:.1%} of ceiling)")
    if absmax / ceiling > 0.5:
        print("  The most consistently driven weights are already learning-rate bound, so more steps at this "
              "rate cannot grow the delta much. Raising the magnitude means raising lr * equivalent steps.")
    else:
        print("  No weight is near the ceiling, so the learning rate is not what limits the delta. Whether "
              "raising it would help is the verdict's question, not this one's: a delta held down by "
              "uncorrelated increments does not grow by taking larger uncorrelated steps.")


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


def report_base_relative(rows: list[tuple[str, float, float, float]], snapshot: Path, *, top: int) -> None:
    """How far each checkpoint moved its base weight, in units of that weight.

    ``||B@A||`` on its own says nothing: 0.7 is large next to a weight of norm 2 and invisible next
    to one of norm 200. The ratio is what says whether a converged adapter had the room to change
    the output at all, which is the difference between "the data does not ask for this" and "the
    adapter is too small to express it".
    """
    norms, unmatched = base_norms([row[0] for row in rows], snapshot)
    if unmatched:
        print(f"  {len(unmatched)} modules had no matching base weight, e.g. {unmatched[0]}")
    if not norms:
        print("  No module matched a base weight, so no ratio can be reported.")
        return

    matched = [row for row in rows if row[0] in norms]
    base_total = math.sqrt(sum(norms[row[0]]**2 for row in matched))
    early_total = math.sqrt(sum(row[1]**2 for row in matched))
    late_total = math.sqrt(sum(row[2]**2 for row in matched))
    print(f"\nbase-relative offset over {len(matched)} matched modules:")
    print(f"  ||W_base|| {base_total:.6g}")
    print(f"  ||B@A|| / ||W_base||: early {early_total / base_total:.3e}, late {late_total / base_total:.3e}")

    ratios = sorted(
        ((row[2] / norms[row[0]] if norms[row[0]] > 1e-12 else float("inf"), row[0], row[2], norms[row[0]])
         for row in matched),
        reverse=True)
    print(f"  largest {top} per-module ratios:")
    for ratio, name, delta, base in ratios[:top]:
        print(f"    {ratio:.3e}  {name}  (||B@A|| {delta:.4g} / ||W|| {base:.4g})")
    print(f"  smallest ratio: {ratios[-1][0]:.3e} at {ratios[-1][1]}")
    if ratios[-1][0] == 0.0:
        print("    A ratio of exactly zero is the float32 floor, not a dead parameter: the trace the norm "
              "comes from went slightly negative and was clamped. Read it as 'below what float32 resolves'.")
    print(f"\n  For scale: a LoRA that visibly changes a diffusion model's behaviour sits at 1e-2..1e-1 here.\n"
          f"  Reaching 1e-2 from {late_total / base_total:.1e} needs {1e-2 * base_total / max(late_total, 1e-12):.0f}x "
          "the current delta.")


def main() -> None:
    args = parse_args()
    # Paths first: torch takes tens of seconds to import, and a step number guessed from the
    # schedule rather than read off the disk is the common way to get here.
    early_path, late_path = resolve(Path(args.early)), resolve(Path(args.late))
    print(f"early: {early_path.name}\nlate:  {late_path.name}")
    # Both before either read, so an unfinished save is reported with the inventory rather than
    # after minutes of loading the other checkpoint's 200 tensors.
    require_loadable(early_path)
    require_loadable(late_path)

    import torch

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
    # <D_early, D_late - D_early> / (||D_early|| ||D_late - D_early||), from the same three scalars.
    # cos(early, late) is pinned near 1 whenever the norm barely grew -- at growth 1.01 it cannot
    # tell "kept pushing the same way" from "stopped moving" -- so the angle that carries the
    # information is between the *new* motion and what had already accumulated.
    advance = ((total_inner - total_early**2) / (total_early * moved)) if moved > 1e-12 else float("nan")
    print(f"  growth ||late||/||early||: {growth:.3f}")
    print(f"  direction cos(early, late): {cosine:.4f}"
          f"{'   <- uninformative at this growth; read the next line' if growth < 1.05 else ''}")
    print(f"  ||late - early|| / ||early||: {moved / total_early:.3f}")
    print(f"  cos(new motion, accumulated): {advance:.3f}")

    # The increment split along and across the delta it is added to. Independent noise increments are
    # perpendicular in high dimensions, so the parallel part is what a consistent gradient leaves
    # behind and the perpendicular part is what cancels out over a longer run. The null model is
    # sharp: a pure random walk has <D_early, D_late> = ||D_early||^2 exactly, hence a cosine of
    # ||D_early|| / ||D_late|| and nothing along the accumulated direction.
    along = advance * moved if moved > 1e-12 else 0.0
    across = math.sqrt(max(0.0, moved**2 - along**2))
    split = f"  increment {moved:.4g} = {along:.4g} along + {across:.4g} across"
    if along > 1e-12:
        split += f"  (1 : {across / along:.0f})"
    print(split)
    print(f"  random-walk null: cos(early, late) would be {1 / growth:.4f} if every increment were "
          f"independent noise; observed {cosine:.4f}")

    if args.warmup_steps and args.max_steps:
        early_step, late_step = (int(path.name.rsplit("-", 1)[-1]) for path in (early_path, late_path))
        integral_early = lr_integral(early_step, args.warmup_steps, args.max_steps, args.min_lr_ratio)
        integral_late = lr_integral(late_step, args.warmup_steps, args.max_steps, args.min_lr_ratio)
        print(f"\nschedule: warmup {args.warmup_steps}, max {args.max_steps}, min ratio {args.min_lr_ratio}")
        print(f"  peak-rate-equivalent steps: step {early_step} -> {integral_early:.1f}, "
              f"step {late_step} -> {integral_late:.1f}  (ratio {integral_late / max(integral_early, 1e-9):.2f})")
        # Only the aligned part of the increment survives averaging, so extrapolating it is what says
        # whether finishing the schedule can reach a useful magnitude or whether the run is already done.
        spent = integral_late - integral_early
        remaining = lr_integral(args.max_steps, args.warmup_steps, args.max_steps, args.min_lr_ratio) - integral_late
        if spent > 1e-9 and remaining > 0 and along > 0:
            projected = total_late + along * remaining / spent
            print(f"  at this interval's drift rate ({along:.3g} of aligned motion per {spent:.0f} "
                  f"equivalent steps), the {remaining:.0f} left in the schedule add "
                  f"{along * remaining / spent:.3g}, taking ||B@A|| {total_late:.3g} -> {projected:.3g}.")
        # along grows with the interval and across only with its square root, so the raw ratio is
        # larger for a wider interval at identical gradient quality. Dividing the square root back
        # out leaves a per-step drift-to-noise that two runs can be compared on even when their
        # intervals and schedules differ -- which is what makes it usable for ablating the
        # conditioning signal rather than just describing one run.
        if spent > 1e-9 and across > 1e-12:
            per_step = (along / across) / math.sqrt(spent)
            print(f"  per-step drift-to-noise, (along/across)/sqrt(equivalent steps): {per_step:.5f}")
            # Gradient noise falls as 1/sqrt(batch) while the true gradient does not, so the per-step
            # figure carries a factor of sqrt(batch). Two runs at different batch sizes cannot be read
            # against each other until that is divided out, and the run with the larger batch would
            # otherwise be credited for its batch alone -- which is exactly the confound in an
            # ablation that changes the conditioning signal and the batch at the same time.
            if args.global_batch > 0:
                print(f"  per-sample drift-to-noise, /sqrt(equivalent steps * batch {args.global_batch}): "
                      f"{per_step / math.sqrt(args.global_batch):.5f}  <- compare runs on this")
            else:
                print("  pass --global-batch to also print the batch-free figure, which is the one two "
                      "runs at different batch sizes can be compared on.")
        if early_step < args.warmup_steps:
            print(f"  NOTE: step {early_step} is mid-warmup, so its learning rate was "
                  f"{early_step / args.warmup_steps:.0%} of peak and its integral is small. Two checkpoints "
                  "inside or straddling warmup are much closer in training than their step numbers suggest.")
        if args.peak_lr > 0:
            report_lr_ceiling(late, integral_late, args.peak_lr)

    print()
    if moved / total_early < 0.02:
        print(f"VERDICT: the delta is standing still -- it moved {moved / total_early:.1%} of its own norm "
              "across this interval. Neither coherent descent nor a random walk is that slow, so the step "
              "size is the binding constraint, not the number of steps. cos(early, late) near 1 here says "
              "nothing about direction; it is what any near-stationary pair produces.")
    elif advance > 0.5:
        print(f"VERDICT: the new motion extends the accumulated delta (cos {advance:.3f}). The updates are "
              "accumulating, so identical-looking samples mean the magnitude has not yet reached the point "
              "of changing the output -- a question of steps and learning rate, not of a broken signal.")
    elif advance > 0.1:
        print(f"VERDICT: the new motion is mostly sideways to what had accumulated (cos {advance:.3f}). Some "
              "of each step lengthens the delta and most of it rotates it, so the norm will grow far more "
              "slowly than the step count suggests.")
    elif advance < -0.05:
        # Below the null, not merely at it. A random walk leaves <D_early, D_late> = ||D_early||^2 and
        # hence advance = 0 exactly, so a negative value is increments that undo what accumulated --
        # overshoot around a basin, which is the one case where a smaller step is the fix.
        print(f"VERDICT: the new motion points back against the accumulated delta (cos {advance:.3f}), which "
              "is *below* the random-walk null of 0. Independent noise would leave the accumulated direction "
              "untouched; actively eroding it means the steps are overshooting. Lower the learning rate. "
              "More steps at this rate will keep undoing the previous ones.")
    else:
        print(f"VERDICT: the new motion is orthogonal to the accumulated delta (cos {advance:.3f}), which is "
              "the random-walk null of 0 to within noise. The increments are not fighting each other, they "
              "carry no shared direction at all, so the norm still inflates while nothing accumulates -- read "
              "the growth above as a random walk, not as progress. This is the gradient's signal-to-noise "
              "floor, and a larger learning rate only takes larger uncorrelated steps: the levers are the "
              "batch, the conditioning the loss can attribute, and the adapter's capacity. Compare an earlier "
              "interval of the same run; a floor reached partway through shows up as a drift-to-noise that "
              "collapsed rather than one that was always low.")

    if args.base_snapshot:
        report_base_relative(rows, Path(args.base_snapshot).expanduser(), top=args.top)

    rows.sort(key=lambda row: row[2], reverse=True)
    print(f"\ntop {args.top} modules by late ||B@A||:")
    for name, norm_early, norm_late, inner in rows[:args.top]:
        angle = inner / (norm_early * norm_late) if norm_early > 1e-12 and norm_late > 1e-12 else float("nan")
        print(f"  {name}\n    early {norm_early:.4g}  late {norm_late:.4g}  "
              f"growth {norm_late / max(norm_early, 1e-12):.2f}  cos {angle:.3f}")


if __name__ == "__main__":
    main()
