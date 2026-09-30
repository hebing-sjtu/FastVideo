# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from fastvideo.train.utils.solarwm_lora import _index_tensors, lora_identity


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "base_model.model.transformer_blocks.7.attn.to_q.lora_A.weight",
            (7, "q", "A"),
        ),
        (
            "base_model.model.transformer_blocks.49.attn.to_out.0.lora_B.default.weight",
            (49, "out", "B"),
        ),
        (
            "transformer_blocks.3.checkpointed.attn.to_v.lora_A",
            (3, "v", "A"),
        ),
        ("token_refiner.refiner_blocks.0.attn.to_q.lora_A.weight", None),
    ],
)
def test_lora_identity_normalizes_solarwm_and_fastvideo_names(name: str, expected: tuple[int, str, str] | None) -> None:
    assert lora_identity(name) == expected


def test_index_tensors_requires_the_complete_400_tensor_proxy_topology() -> None:
    values = {
        f"base_model.model.transformer_blocks.{block}.attn.to_{projection}"
        f"{'.0' if projection == 'out' else ''}.lora_{slot}.weight": object()
        for block in range(50)
        for projection in ("q", "k", "v", "out")
        for slot in ("A", "B")
    }
    assert len(_index_tensors(values, label="test")) == 400

    values.pop(next(iter(values)))
    with pytest.raises(RuntimeError, match="topology differs"):
        _index_tensors(values, label="test")
