# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

import fastvideo.train.utils.lora as lora_utils
from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.training.training_utils import (
    get_cosine_schedule_with_min_lr,
)


def test_training_lora_weights_stay_fp32_over_bf16_base() -> None:
    base = torch.nn.Linear(8, 12, bias=False, dtype=torch.bfloat16)
    layer = BaseLayerWithLoRA(
        base,
        lora_rank=4,
        lora_alpha=4,
        training_mode=True,
    )

    assert layer.base_layer.weight.dtype == torch.bfloat16
    assert layer.lora_A.dtype == torch.float32
    assert layer.lora_B.dtype == torch.float32


def test_cosine_with_min_lr_scales_cosine_to_floor() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = get_cosine_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=100,
        min_lr_ratio=0.1,
    )

    lr_lambda = scheduler.lr_lambdas[0]
    assert lr_lambda(0) == 1.0
    assert lr_lambda(50) == 0.55
    assert lr_lambda(100) == 0.1


def test_replicated_lora_gradients_are_averaged_over_mesh(monkeypatch) -> None:
    class FakeDTensor:

        def __init__(self, local, *, mesh=None, placements=None):
            self._local = local
            self.device_mesh = mesh
            self.placements = placements
            self.grad = None

        def to_local(self):
            return self._local

    class FakeLoRALayer:
        pass

    class FakeMesh:
        ndim = 2

        def get_group(self, mesh_dim):
            return f"group-{mesh_dim}"

    mesh = FakeMesh()
    placements = [lora_utils.Replicate(), lora_utils.Replicate()]
    grad = FakeDTensor(torch.tensor([3.0]), mesh=mesh, placements=placements)
    param = FakeDTensor(torch.tensor([1.0]), mesh=mesh, placements=placements)
    param.grad = grad
    layer = FakeLoRALayer()
    layer.lora_A = param
    layer.lora_B = None
    transformer = SimpleNamespace(modules=lambda: [layer])

    monkeypatch.setattr(lora_utils, "DTensor", FakeDTensor)
    monkeypatch.setattr(lora_utils, "BaseLayerWithLoRA", FakeLoRALayer)
    monkeypatch.setattr(lora_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(lora_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        lora_utils.dist,
        "get_world_size",
        lambda group: 1 if group == "group-0" else 2,
    )

    def fake_all_reduce(tensor, *, op, group):
        del op
        assert group == "group-1"
        tensor.add_(5.0)

    monkeypatch.setattr(lora_utils.dist, "all_reduce", fake_all_reduce)

    count = lora_utils.synchronize_lora_gradients(transformer)

    assert count == 1
    torch.testing.assert_close(grad.to_local(), torch.tensor([4.0]))
