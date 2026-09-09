# SPDX-License-Identifier: Apache-2.0
"""Key filters for inspect_lora_checkpoint — no torch required."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "h3_proxy" / "inspect_lora_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("inspect_lora_checkpoint", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_INSPECT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INSPECT)

lora_slot = _INSPECT.lora_slot
is_role_lora_weight = _INSPECT.is_role_lora_weight


def test_role_lora_parameters_are_weights():
    assert lora_slot("roles.student.transformer.blocks.0.attn.to_k.lora_A") == "A"
    assert lora_slot("roles.student.transformer.blocks.0.attn.to_k.lora_B") == "B"
    assert is_role_lora_weight("roles.student.transformer.blocks.0.attn.to_k.lora_B")


def test_peft_style_weight_suffix():
    assert lora_slot("roles.student.transformer.to_k.lora_A.weight") == "A"
    assert is_role_lora_weight("roles.student.transformer.to_k.lora_B.weight")


def test_optimizer_hyperparams_are_not_weights():
    key = ("optimizers.student.param_groups.transformer.transformer_blocks.0."
           "checkpointed.attn.to_k.lora_A.amsgrad")
    assert lora_slot(key) is None
    assert not is_role_lora_weight(key)


def test_optimizer_moments_are_not_role_weights():
    key = "optimizers.student.state.transformer.blocks.0.attn.to_k.lora_B.exp_avg"
    assert not is_role_lora_weight(key)
