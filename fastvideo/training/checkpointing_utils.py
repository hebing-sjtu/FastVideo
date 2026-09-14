# SPDX-License-Identifier: Apache-2.0
import random
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint.stateful
from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
                                                     set_model_state_dict, set_optimizer_state_dict)

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _local_shard(param: torch.Tensor) -> torch.Tensor:
    """This rank's slice of a possibly-sharded parameter, as a plain tensor.

    Anything that touches a DTensor's full value is a collective. These numbers are gathered from
    inside ``dcp.load``, where issuing one would interleave with DCP's own collectives and desync
    the ranks, so the shard is all that may be read.
    """
    to_local = getattr(param, "to_local", None)
    return to_local() if callable(to_local) else param


def lora_b_local_stats(model: torch.nn.Module) -> tuple[float, int]:
    """Sum of squares and element count of this rank's ``lora_B`` shards. No collectives.

    ``lora_B`` is zero-initialised so the adapter starts as the identity. That makes its norm the
    one number separating "a checkpoint's weights are in this model" from "this model is the base
    model", which no key count can establish: a load that matched nothing and a load that was never
    asked for anything both leave it at exactly zero.
    """
    total, elements = 0.0, 0
    for name, param in model.named_parameters():
        if "lora_B" not in name:
            continue
        shard = _local_shard(param.detach())
        if shard.numel():
            total += float(shard.float().pow(2).sum())
            elements += int(shard.numel())
    return total, elements


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        # Read by `CheckpointManager.maybe_resume` after the load, where collectives are safe.
        self.load_report: dict[str, Any] | None = None

    def _requested_keys(self) -> set[str]:
        """The keys `state_dict` would return, computed without the all-gather it needs to return
        their values. `named_parameters` is local metadata; `get_model_state_dict` is a collective.
        """
        return {
            k.replace("._checkpoint_wrapped_module.", ".")
            for k, v in self.model.named_parameters() if v.requires_grad
        }

    def state_dict(self) -> dict[str, Any]:
        state_dict = get_model_state_dict(self.model)

        filtered_state_dict = {k: v for k, v in state_dict.items() if k in self._requested_keys()}

        return filtered_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Two things here fail silently, and both end with the base model being sampled by a job
        # whose logs and output filenames say it resumed a checkpoint.
        #
        # DCP asks `state_dict()` which keys to read, and that filters to `requires_grad`. A model
        # built frozen therefore requests nothing, DCP reads nothing, and this arrives empty. Then
        # `strict=False` -- which is needed, since the save holds only trainable params while the
        # model also has frozen ones -- means any key that does not match is skipped without a word.
        # A renamed module, activation checkpointing enabled on one side only (it inserts a
        # `.checkpointed.` segment), or a different LoRA rank all land here as a quiet no-op.
        #
        # So record what was asked for and measure the adapter rather than trusting key bookkeeping.
        # DCP calls this from inside `dcp.load`, between its own collectives, so everything gathered
        # here is rank-local and the verdict is left to `maybe_resume` once the load has joined.
        before, elements = lora_b_local_stats(self.model)
        requested = self._requested_keys()
        supplied = set(state_dict)
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        after, _ = lora_b_local_stats(self.model)

        missing = sorted(requested - supplied)
        self.load_report = {
            "requested": len(requested),
            "supplied": len(supplied),
            "applied": len(requested & supplied),
            "missing": missing,
            "lora_b_sq_before": before,
            "lora_b_sq_after": after,
            "lora_b_elements": elements,
        }
        logger.info(
            "Loaded %d/%d requested model tensors (%d supplied, %d unmatched); "
            "local lora_B norm %.6g -> %.6g over %d elements",
            len(requested & supplied),
            len(requested),
            len(supplied),
            len(missing),
            before**0.5,
            after**0.5,
            elements,
        )
        if missing:
            logger.warning(
                "%d requested tensors were absent from the checkpoint and keep their constructed value, "
                "e.g. %s",
                len(missing),
                ", ".join(missing[:4]),
            )


class OptimizerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        return get_optimizer_state_dict(  # type: ignore[no-any-return]
            self.model,
            self.optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


class SchedulerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])


class RandomStateWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, noise_generator: torch.Generator | None = None) -> None:
        self.noise_generator = noise_generator

    def state_dict(self) -> dict[str, Any]:
        state = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
            if torch.cuda.device_count() > 1:
                state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        if self.noise_generator is not None:
            state["noise_generator_state"] = self.noise_generator.get_state()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])

        if "numpy_rng_state" in state_dict:
            np.random.set_state(state_dict["numpy_rng_state"])

        if "python_rng_state" in state_dict:
            random.setstate(state_dict["python_rng_state"])

        # Restore CUDA random state
        if torch.cuda.is_available():
            if "cuda_rng_state" in state_dict:
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            if "cuda_rng_state_all" in state_dict:
                torch.cuda.set_rng_state_all(state_dict["cuda_rng_state_all"])

        # Restore noise generator state
        if "noise_generator_state" in state_dict and self.noise_generator is not None:
            self.noise_generator.set_state(state_dict["noise_generator_state"])
