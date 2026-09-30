# SPDX-License-Identifier: Apache-2.0
"""Load a SolarWM MiniMax-H3 proxy LoRA into a FastVideo eval model."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_SLOT = re.compile(
    r"(?:^|\.)transformer_blocks\.(?P<block>\d+)\."
    r"(?:[^.]+\.)*attn\.to_(?P<projection>q|k|v|out)"
    r"(?:\.0)?(?:\.base_layer)?\.lora_(?P<slot>A|B)"
    r"(?:\.default)?(?:\.weight)?$"
)
_EXPECTED_BLOCKS = 50
_EXPECTED_PROJECTIONS = ("q", "k", "v", "out")


def lora_identity(name: str) -> tuple[int, str, str] | None:
    """Return a wrapper-independent H3 attention-LoRA identity."""

    normalized = str(name).replace("._checkpoint_wrapped_module.", ".").replace(".checkpointed.", ".")
    match = _SLOT.search(normalized)
    if match is None:
        return None
    return (
        int(match.group("block")),
        str(match.group("projection")),
        str(match.group("slot")),
    )


def _expected_identities() -> set[tuple[int, str, str]]:
    return {
        (block, projection, slot)
        for block in range(_EXPECTED_BLOCKS)
        for projection in _EXPECTED_PROJECTIONS
        for slot in ("A", "B")
    }


def _index_tensors(values: Mapping[str, Any], *, label: str) -> dict[tuple[int, str, str], tuple[str, Any]]:
    indexed: dict[tuple[int, str, str], tuple[str, Any]] = {}
    for name, value in values.items():
        identity = lora_identity(str(name))
        if identity is None:
            continue
        if identity in indexed:
            raise RuntimeError(f"{label} has duplicate LoRA identity {identity}: {indexed[identity][0]!r} and {name!r}")
        indexed[identity] = (str(name), value)
    expected = _expected_identities()
    if set(indexed) != expected:
        raise RuntimeError(
            f"{label} LoRA topology differs: "
            f"missing={sorted(expected - set(indexed))[:8]} "
            f"extra={sorted(set(indexed) - expected)[:8]} "
            f"observed={len(indexed)} expected={len(expected)}"
        )
    return indexed


def _load_source(root: Path, weight_source: str) -> Mapping[str, Any]:
    import torch

    if weight_source == "live":
        payload = torch.load(
            root / "adapter.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        metadata = payload.get("metadata", {})
        if (
            metadata.get("schema") != "solarwm.minimax-h3-lora.v1"
            or int(metadata.get("rank", 0)) != 128
            or int(metadata.get("alpha", 0)) != 128
            or int(metadata.get("target_count", 0)) != 200
        ):
            raise RuntimeError("SolarWM adapter metadata is not the H3 proxy LoRA-128 profile")
        values = payload.get("state")
    else:
        payload = torch.load(
            root / "ema.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if payload.get("schema") != "solarwm.minimax-h3-ema.v1" or not bool(payload.get("trainable_only")):
            raise RuntimeError("SolarWM EMA is not a trainable-only H3 EMA checkpoint")
        values = payload.get("shadow")
    if not isinstance(values, Mapping):
        raise RuntimeError(f"SolarWM {weight_source} checkpoint has no tensor mapping")
    return values


def _validate_checkpoint(root: Path, weight_source: str) -> int:
    if weight_source not in {"live", "ema"}:
        raise ValueError("SolarWM weight source must be 'live' or 'ema'")
    for name in ("COMPLETE.json", "checkpoint-manifest.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"SolarWM checkpoint is incomplete; missing {root / name}")
    component = "adapter.pt" if weight_source == "live" else "ema.pt"
    if not (root / component).is_file():
        raise FileNotFoundError(f"SolarWM checkpoint is missing {root / component}")
    manifest = json.loads((root / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    contract = manifest.get("contract", {})
    if (
        contract.get("family") != "minimax_h3"
        or contract.get("stage") != "stage0p5"
        or contract.get("parameterization") != "peft-lora-r128-alpha128"
        or contract.get("data_generation") != "h3.ref2va-proxy.124f.v1"
    ):
        raise RuntimeError("SolarWM checkpoint is not the H3 124f Ref2VA proxy profile")
    step = int(manifest.get("step", 0))
    if step < 1:
        raise RuntimeError(f"SolarWM checkpoint manifest has invalid step {step}")
    return step


def load_solarwm_h3_proxy_lora(
    transformer: Any,
    checkpoint: str | Path,
    *,
    weight_source: str = "ema",
) -> int:
    """Strictly copy one SolarWM proxy adapter into a built FastVideo model."""

    import torch

    root = Path(checkpoint).expanduser().resolve()
    source_name = str(weight_source).strip().lower()
    step = _validate_checkpoint(root, source_name)
    source = _index_tensors(_load_source(root, source_name), label="SolarWM checkpoint")
    target_values = {
        name: parameter for name, parameter in transformer.named_parameters() if lora_identity(name) is not None
    }
    target = _index_tensors(target_values, label="FastVideo eval model")

    before_b_sq = 0.0
    after_b_sq = 0.0
    with torch.no_grad():
        for identity in sorted(source):
            source_key, value = source[identity]
            target_key, parameter = target[identity]
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            if tuple(value.shape) != tuple(local.shape):
                raise RuntimeError(
                    f"SolarWM tensor shape differs for {identity}: "
                    f"{source_key}={tuple(value.shape)} {target_key}={tuple(local.shape)}"
                )
            if identity[2] == "B":
                before_b_sq += float(local.detach().float().pow(2).sum())
            local.copy_(value.to(device=local.device, dtype=local.dtype))
            if identity[2] == "B":
                after_b_sq += float(local.detach().float().pow(2).sum())
    if after_b_sq == 0.0:
        raise RuntimeError("SolarWM LoRA-B norm is zero after loading; refusing to sample the base model")
    logger.info(
        "Loaded SolarWM H3 proxy %s weights at step %d: 400/400 tensors; local LoRA-B norm %.6g -> %.6g",
        source_name,
        step,
        before_b_sq**0.5,
        after_b_sq**0.5,
    )
    return step


__all__ = ["load_solarwm_h3_proxy_lora", "lora_identity"]
