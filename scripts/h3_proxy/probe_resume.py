#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Would this checkpoint's weights actually land in the model this config builds?

A resume can report success and change nothing. DCP asks the model which keys to read, matches them
against the save by name, and ``strict=False`` skips the rest without a word, so a checkpoint whose
key names disagree with the model loads zero tensors while ``maybe_resume`` still returns its step
and the validation filenames still carry it. With a zero-initialised ``lora_B`` the result is the
base model, which looks exactly like a checkpoint that learned nothing.

Three settings rename or reshape every LoRA key, and all three live in the config rather than in the
checkpoint:

* ``enable_gradient_checkpointing_type`` inserts a ``.checkpointed.`` segment into every block key.
* ``lora.rank`` changes each tensor's shape, which DCP rejects per tensor.
* ``lora.target_modules`` decides which keys exist at all.

``CheckpointManager._write_metadata`` stores the whole training config in the checkpoint's
``metadata.json``, so all of this is answerable from disk, with no GPU and no model build. Run it
before paying for a sampling job, not after.

Usage::

    python scripts/h3_proxy/probe_resume.py \\
        /data/binghe/h3_proxy/runs/gta_v2_cwm/checkpoints/checkpoint-138 \\
        --config examples/train/scenario/h3_proxy/proxy_bd_finetune.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

# Settings that change the *names* or *shapes* of the saved tensors. A disagreement in any of these
# is not a tuning difference, it is a load that will silently match nothing.
KEY_SHAPING_FIELDS = (
    ("models", "student", "trainable"),
    ("models", "student", "enable_gradient_checkpointing_type"),
    ("models", "student", "lora", "enable"),
    ("models", "student", "lora", "rank"),
    ("models", "student", "lora", "alpha"),
    ("models", "student", "lora", "target_modules"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="A checkpoint-<step> directory.")
    parser.add_argument("--config", help="The YAML the eval would run with. Omit to only describe the checkpoint.")
    parser.add_argument("--list-keys", type=int, default=4, help="How many sample LoRA key names to print.")
    return parser.parse_args()


def dig(tree: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        if not isinstance(tree, dict):
            return None
        tree = tree.get(part)
    return tree


def load_training_config(checkpoint: Path) -> dict[str, Any]:
    meta_path = checkpoint / "metadata.json"
    if not meta_path.is_file():
        raise SystemExit(f"No metadata.json in {checkpoint}, so the training config it was saved with is unknown.")
    with open(meta_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise SystemExit(f"{meta_path} holds no 'config', so there is nothing to compare against.")
    print(f"checkpoint step {metadata.get('step')!r} from {checkpoint}")
    return config


def describe_saved_tensors(checkpoint: Path, list_keys: int) -> None:
    """What is actually in the save, by name, straight from the DCP metadata."""
    import torch.distributed.checkpoint as dcp

    dcp_dir = checkpoint / "dcp"
    if not (dcp_dir / ".metadata").is_file():
        raise SystemExit(f"No dcp/.metadata under {checkpoint}; this is an unfinished save and cannot be loaded.")
    names = list(dcp.FileSystemReader(str(dcp_dir)).read_metadata().state_dict_metadata)

    lora_a = sorted(name for name in names if "lora_A" in name)
    lora_b = sorted(name for name in names if "lora_B" in name)
    print(f"\nsaved tensors: {len(names)} total, {len(lora_a)} lora_A, {len(lora_b)} lora_B")
    if not lora_b:
        print("  NO lora_B IN THE SAVE. Resuming this cannot produce anything but the base model.")
    for name in lora_b[:list_keys]:
        print(f"  {name}")

    checkpointed = sum(1 for name in lora_b if ".checkpointed." in name)
    if lora_b:
        if checkpointed == len(lora_b):
            print("  every LoRA key carries '.checkpointed.', so this was trained with "
                  "enable_gradient_checkpointing_type set, and the eval config must set it too.")
        elif checkpointed:
            print(f"  MIXED: {checkpointed}/{len(lora_b)} keys carry '.checkpointed.'.")
        else:
            print("  no LoRA key carries '.checkpointed.', so this was trained without activation "
                  "checkpointing, and an eval config that enables it will match none of them.")


def compare(training: dict[str, Any], eval_config: dict[str, Any], where: str) -> bool:
    print(f"\nkey-shaping settings, checkpoint vs {where}:")
    agreed = True
    for path in KEY_SHAPING_FIELDS:
        mine, theirs = dig(training, path), dig(eval_config, path)
        dotted = ".".join(path)
        if mine == theirs:
            print(f"  ok        {dotted}: {mine!r}")
        else:
            agreed = False
            print(f"  MISMATCH  {dotted}: checkpoint {mine!r} != config {theirs!r}")
    return agreed


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    training = load_training_config(checkpoint)
    describe_saved_tensors(checkpoint, args.list_keys)

    if not args.config:
        return
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        raise SystemExit(f"--config not found: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        eval_config = yaml.safe_load(handle) or {}

    if compare(training, eval_config, config_path.name):
        print("\nEvery setting that shapes a key agrees, so the names will match and the weights will land.\n"
              "If the output is still identical to the base model, the delta is real but too small to see -- "
              "measure it with compare_lora_checkpoints.py rather than by eye.")
    else:
        raise SystemExit("\nAt least one setting that renames or reshapes the saved tensors disagrees. Resuming "
                         "this checkpoint with this config would load only the keys that happen to match and "
                         "silently skip the rest. Evaluate with the config the run trained under.")


if __name__ == "__main__":
    main()
