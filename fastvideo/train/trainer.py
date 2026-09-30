# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import CallbackDict
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.utils.tracking import build_tracker

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig,
    )

logger = init_logger(__name__)


def _distributed_mean_scalars(
    values: dict[str, float | torch.Tensor],
    *,
    divisor: int,
    world_group: Any,
) -> dict[str, float]:
    """Average scalar sums over accumulation rounds and every distributed rank."""
    if not values:
        return {}
    divisor = max(1, int(divisor))
    keys = sorted(values)
    first_tensor = next((value for value in values.values() if isinstance(value, torch.Tensor)), None)
    device = getattr(world_group, "device", None)
    if device is None:
        device = first_tensor.device if first_tensor is not None else torch.device("cpu")
    scalars = [torch.as_tensor(values[key], device=device, dtype=torch.float32).detach().reshape(()) for key in keys]
    packed = torch.stack(scalars)
    if int(world_group.world_size) > 1:
        packed = world_group.all_reduce(packed)
    packed.div_(divisor * int(world_group.world_size))
    materialized = packed.cpu().tolist()
    return {key: float(value) for key, value in zip(keys, materialized, strict=True)}


def _coerce_log_scalar(
    value: Any,
    *,
    where: str,
) -> float | torch.Tensor:
    """Coerce *value* to a loggable scalar.

    GPU tensors stay on device so we avoid a
    ``cudaDeviceSynchronize`` per accumulation step.
    The caller must materialize them to float when logging.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor at {where}, got shape={tuple(value.shape)}")
        return value.detach()
    if isinstance(value, float | int):
        return float(value)
    raise TypeError(f"Expected a scalar (float/int/Tensor) at {where}, got {type(value).__name__}")


@dataclass(slots=True)
class TrainLoopState:
    step: int
    accum_iter: int


class Trainer:
    def __init__(
        self,
        training_config: TrainingConfig,
        *,
        config: dict[str, Any] | None = None,
        callback_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.training_config = training_config
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.global_rank = self.world_group.rank
        self.local_rank = self.world_group.local_rank
        self.tracker = build_tracker(
            training_config.tracker,
            training_config.checkpoint,
            config=config,
        )
        self.callbacks = CallbackDict(
            callback_configs or {},
            training_config,
        )
        self._metric_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))

    def _rolling_loss_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Add an actual 100-step mean for noisy supervised-loss metrics."""
        smoothed: dict[str, float] = {}
        for key, value in metrics.items():
            if key != "total_loss" and not key.endswith("_loss"):
                continue
            history = self._metric_history[key]
            history.append(float(value))
            smoothed[f"rolling_100/{key}"] = sum(history) / len(history)
        return smoothed

    def _iter_dataloader(self, dataloader: Any) -> Iterator[dict[str, Any]]:
        data_iter = iter(dataloader)
        while True:
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            yield batch

    def _run_method_validation(
        self,
        method: TrainingMethod,
        iteration: int,
    ) -> None:
        hook = getattr(method, "on_validation_begin", None)
        if hook is None:
            return
        validation_metrics: dict[str, LogScalar] = hook(iteration)
        validation_metrics = {
            k: float(_coerce_log_scalar(v, where=(f"method.on_validation_begin().metrics[{k!r}]")))
            for k, v in validation_metrics.items()
        }
        if self.global_rank == 0 and validation_metrics:
            self.tracker.log(validation_metrics, iteration)

    def run(
        self,
        method: TrainingMethod,
        *,
        dataloader: Any,
        max_steps: int,
        start_step: int = 0,
        checkpoint_manager: Any | None = None,
    ) -> None:
        tc = self.training_config
        grad_accum = max(
            1,
            int(tc.loop.gradient_accumulation_steps or 1),
        )

        method.set_tracker(self.tracker)
        method.on_train_start()
        self.callbacks.on_train_start(
            method,
            iteration=start_step,
        )

        resume_from_checkpoint = tc.checkpoint.resume_from_checkpoint or ""
        solarwm_checkpoint = tc.checkpoint.solarwm_checkpoint or ""
        if resume_from_checkpoint and solarwm_checkpoint:
            raise ValueError("resume_from_checkpoint and solarwm_checkpoint are mutually exclusive")
        if solarwm_checkpoint:
            from fastvideo.train.utils.solarwm_lora import (
                load_solarwm_h3_proxy_lora,
            )

            start_step = load_solarwm_h3_proxy_lora(
                method.student.transformer,
                solarwm_checkpoint,
                weight_source=tc.checkpoint.solarwm_weight_source,
            )
            if int(max_steps) != start_step:
                raise ValueError(
                    "SolarWM checkpoint loading is eval-only: "
                    f"max_train_steps={max_steps} must equal checkpoint step {start_step}"
                )
        if checkpoint_manager is not None:
            if resume_from_checkpoint:
                method.seed_optimizer_state_for_resume()
            resumed_step = checkpoint_manager.maybe_resume(resume_from_checkpoint=(resume_from_checkpoint))
            if resumed_step is not None:
                start_step = int(resumed_step)
        # An eval-only job names the step it wants by resuming into it, and an empty
        # resume_from_checkpoint is not an error -- it is how a fresh run starts. The two are then
        # only distinguishable from the step baked into the validation filenames, which is read
        # after the sampling has been paid for, so say which one this is before spending it.
        if solarwm_checkpoint:
            logger.info(
                "solarwm_checkpoint=%r resolved to step %s using %s weights; "
                "the training loop is empty and validation runs once.",
                solarwm_checkpoint,
                start_step,
                tc.checkpoint.solarwm_weight_source,
            )
        elif resume_from_checkpoint:
            logger.info(
                "resume_from_checkpoint=%r resolved to step %s; validation and training continue from there.",
                resume_from_checkpoint,
                start_step,
            )
        else:
            logger.info(
                "No resume_from_checkpoint, so this run starts at step %s with weights as initialized. "
                "A LoRA adapter is zero here and samples identically to the base model -- if you meant to "
                "evaluate a checkpoint, pass --training.checkpoint.resume_from_checkpoint.",
                start_step,
            )
        self.callbacks.on_validation_begin(
            method,
            iteration=start_step,
        )
        self._run_method_validation(method, start_step)
        method.optimizers_zero_grad(start_step)

        data_stream = self._iter_dataloader(dataloader)

        # Restore the RNG snapshot LAST — after dcp.load,
        # after iter(dataloader), after everything that may
        # have advanced the RNG as a side-effect.
        if checkpoint_manager is not None and resume_from_checkpoint:
            checkpoint_manager.load_rng_snapshot(
                resume_from_checkpoint,
            )
        progress = tqdm(
            range(start_step + 1, max_steps + 1),
            initial=start_step,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        # Allow method-specific optimization flow (e.g. DiffusionNFT).
        method_manages_optimization = bool(method.manages_optimization())
        for step in progress:
            t0 = time.perf_counter()

            # Accumulate on GPU during grad-accum; materialise
            # to CPU once per step right before logging.
            loss_sums: dict[str, float | torch.Tensor] = {}
            metric_sums: dict[str, float | torch.Tensor] = {}
            optimizer_metrics: dict[str, float] = {}
            if method_manages_optimization:
                loss_map, outputs, step_metrics = method.managed_train_step(
                    data_stream,
                    step,
                )
                for k, v in loss_map.items():
                    if isinstance(v, torch.Tensor):
                        loss_sums[k] = v.detach()
                for k, v in step_metrics.items():
                    if k in loss_sums:
                        raise ValueError(
                            f"Metric key {k!r} collides "
                            "with loss key. Use a "
                            "different name (e.g. prefix "
                            "with 'train/')."
                        )
                    metric_sums[k] = _coerce_log_scalar(
                        v,
                        where=(f"method.managed_train_step().metrics[{k!r}]"),
                    )
            else:
                for accum_iter in range(grad_accum):
                    batch = next(data_stream)
                    loss_map, outputs, step_metrics = method.single_train_step(
                        batch,
                        step,
                    )

                    method.backward(
                        loss_map,
                        outputs,
                        grad_accum_rounds=grad_accum,
                    )

                    for k, v in loss_map.items():
                        if isinstance(v, torch.Tensor):
                            prev = loss_sums.get(k, 0.0)
                            loss_sums[k] = prev + v.detach()
                    for k, v in step_metrics.items():
                        if k in loss_sums:
                            raise ValueError(
                                f"Metric key {k!r} collides "
                                "with loss key. Use a "
                                "different name (e.g. prefix "
                                "with 'train/')."
                            )
                        prev = metric_sums.get(k, 0.0)
                        metric_sums[k] = prev + _coerce_log_scalar(
                            v,
                            where=(f"method.single_train_step().metrics[{k!r}]"),
                        )

            if not method_manages_optimization:
                optimizer_metrics = method.optimizer_lr_metrics()
                # LoRA adapters are attached after FSDP has captured its parameter groups,
                # so their replicated gradients need an explicit data-parallel average.
                # This must precede clipping so every rank clips the same global gradient.
                method.synchronize_gradients(step)
                self.callbacks.on_before_optimizer_step(
                    method,
                    iteration=step,
                )
                method.optimizers_schedulers_step(step)
                method.optimizers_zero_grad(step)

            # Single CPU sync point: materialise GPU tensors
            # to float right before logging.
            divisor = 1 if method_manages_optimization else grad_accum
            scalar_sums = dict(loss_sums)
            scalar_sums.update(metric_sums)
            metrics = _distributed_mean_scalars(
                scalar_sums,
                divisor=divisor,
                world_group=self.world_group,
            )
            if self.global_rank == 0:
                metrics.update(self._rolling_loss_metrics(metrics))
            metrics.update(optimizer_metrics)
            metrics["step_time_sec"] = time.perf_counter() - t0
            metrics["vsa_sparsity"] = float(tc.vsa_sparsity)
            if self.global_rank == 0 and metrics:
                self.tracker.log(metrics, step)

            self.callbacks.on_training_step_end(
                method,
                metrics,
                iteration=step,
            )

            if checkpoint_manager is not None:
                checkpoint_manager.maybe_save(step)

            self.callbacks.on_validation_begin(
                method,
                iteration=step,
            )
            self._run_method_validation(method, step)
            self.callbacks.on_validation_end(
                method,
                iteration=step,
            )

        self.callbacks.on_train_end(
            method,
            iteration=max_steps,
        )

        if checkpoint_manager is not None:
            checkpoint_manager.save_final(max_steps)

        self.tracker.finish()
