# SPDX-License-Identifier: Apache-2.0
"""CPU-only integration tests for Trainer validation hook dispatch."""

from __future__ import annotations

from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Any

import torch

from fastvideo.train.callbacks.validation import ValidationCallback
from fastvideo.train.trainer import Trainer, _distributed_mean_scalars
from fastvideo.train.utils.training_config import TrainingConfig


class _RecordingValidationCallback(ValidationCallback):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[int] = []

    def _run_validation(self, method: Any, step: int) -> None:  # type: ignore[override]
        del method
        self.run_calls.append(step)


class _DummyTracker:

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logs.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class _DummyMethod:

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.train_start_calls = 0
        self.zero_grad_steps: list[int] = []
        self.optimizer_steps: list[int] = []
        self.synchronize_steps: list[int] = []
        self.backward_calls = 0
        self.tracker = None

    def set_tracker(self, tracker: Any) -> None:
        self.tracker = tracker

    def on_train_start(self) -> None:
        self.train_start_calls += 1

    def manages_optimization(self) -> bool:
        return False

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
        assert batch["sample"] == "x"
        loss = self.weight * 0.0 + 1.0
        return {"total_loss": loss}, {}, {"iteration_seen": float(iteration)}

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int,
    ) -> None:
        del outputs
        self.backward_calls += 1
        (loss_map["total_loss"] / grad_accum_rounds).backward()

    def optimizers_schedulers_step(self, iteration: int) -> None:
        self.optimizer_steps.append(iteration)

    def optimizer_lr_metrics(self) -> dict[str, float]:
        return {"learning_rate": 1e-4}

    def synchronize_gradients(self, iteration: int) -> None:
        self.synchronize_steps.append(iteration)

    def optimizers_zero_grad(self, iteration: int) -> None:
        self.zero_grad_steps.append(iteration)
        self.weight.grad = None


def test_distributed_mean_scalars_averages_accumulation_and_ranks() -> None:
    class Group:
        device = torch.device("cpu")
        world_size = 2

        @staticmethod
        def all_reduce(values: torch.Tensor) -> torch.Tensor:
            # The other rank contributes [6, 10].
            return values + torch.tensor([6.0, 10.0])

    metrics = _distributed_mean_scalars(
        {
            "a_loss": torch.tensor(2.0),
            "b_loss": 6.0,
        },
        divisor=2,
        world_group=Group(),
    )

    assert metrics == {
        "a_loss": 2.0,
        "b_loss": 4.0,
    }


def test_distributed_mean_scalars_single_rank() -> None:
    class Group:
        world_size = 1

        @staticmethod
        def all_reduce(values: torch.Tensor) -> torch.Tensor:
            return values

    assert _distributed_mean_scalars(
        {"training_loss": torch.tensor(6.0)},
        divisor=3,
        world_group=Group(),
    ) == {"training_loss": 2.0}


def test_rolling_loss_metrics_uses_last_100_global_steps() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer._metric_history = defaultdict(lambda: deque(maxlen=100))

    result = {}
    for step in range(101):
        result = trainer._rolling_loss_metrics({
            "training_loss": float(step),
            "learning_rate": 2e-5,
        })

    assert result == {"rolling_100/training_loss": 50.5}
    assert "learning_rate" not in trainer._metric_history


def test_trainer_runs_validation_callback_during_training(monkeypatch, ) -> None:
    tracker = _DummyTracker()
    group = SimpleNamespace(rank=0, local_rank=0, rank_in_group=0, world_size=1)

    monkeypatch.setattr("fastvideo.train.trainer.get_world_group", lambda: group)
    monkeypatch.setattr("fastvideo.train.trainer.get_sp_group", lambda: group)
    monkeypatch.setattr(
        "fastvideo.train.callbacks.validation.get_world_group",
        lambda: group,
    )
    monkeypatch.setattr(
        "fastvideo.train.callbacks.validation.get_sp_group",
        lambda: group,
    )
    monkeypatch.setattr(
        "fastvideo.train.trainer.build_tracker",
        lambda *args, **kwargs: tracker,
    )

    cfg = TrainingConfig()
    cfg.tracker.project_name = ""
    cfg.loop.gradient_accumulation_steps = 1
    callback_configs = {
        "validation": {
            "_target_": f"{__name__}._RecordingValidationCallback",
            "pipeline_target": "unused.pipeline.Target",
            "dataset_file": "unused.json",
            "every_steps": 2,
        }
    }
    trainer = Trainer(
        cfg,
        callback_configs=callback_configs,
    )
    method = _DummyMethod()

    trainer.run(
        method,
        dataloader=[{
            "sample": "x"
        }],
        max_steps=3,
    )

    validation = trainer.callbacks._callbacks["validation"]
    assert isinstance(validation, _RecordingValidationCallback)
    assert validation.run_calls == [0, 2]
    assert method.train_start_calls == 1
    assert method.backward_calls == 3
    assert method.zero_grad_steps == [0, 1, 2, 3]
    assert method.optimizer_steps == [1, 2, 3]
    assert method.synchronize_steps == [1, 2, 3]
    assert [step for _, step in tracker.logs] == [1, 2, 3]
    assert tracker.finished is True
