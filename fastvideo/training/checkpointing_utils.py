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


def _lora_b_norm(model: torch.nn.Module) -> tuple[float, int]:
    """Total ``lora_B`` norm and count. Zero norm with a nonzero count means an inert adapter.

    ``lora_B`` is zero-initialised so the adapter starts as the identity. That makes its norm the
    one number that separates "a checkpoint's weights are in this model" from "this model is the
    base model", which no key count can establish: a load that matches nothing and a load that was
    never asked for anything both leave it at exactly zero.
    """
    total, count = 0.0, 0
    for name, param in model.named_parameters():
        if "lora_B" in name:
            total += float(param.detach().float().pow(2).sum())
            count += 1
    return total**0.5, count


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        state_dict = get_model_state_dict(self.model)

        param_requires_grad = {
            k.replace("._checkpoint_wrapped_module.", ".")
            for k, v in self.model.named_parameters() if v.requires_grad
        }

        filtered_state_dict = {k: v for k, v in state_dict.items() if k in param_requires_grad}

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
        # So report what was actually asked for and applied, and measure the adapter rather than
        # trusting the key bookkeeping.
        before, lora_params = _lora_b_norm(self.model)
        requested = set(self.state_dict())
        supplied = set(state_dict)
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        after, _ = _lora_b_norm(self.model)

        missing = requested - supplied
        logger.info(
            "Loaded %d/%d requested model tensors (%d supplied, %d unmatched); lora_B norm %.6g -> %.6g",
            len(requested & supplied),
            len(requested),
            len(supplied),
            len(missing),
            before,
            after,
        )
        if missing:
            logger.warning(
                "%d requested tensors were absent from the checkpoint and keep their constructed value, "
                "e.g. %s",
                len(missing),
                ", ".join(sorted(missing)[:4]),
            )
        if lora_params and after == 0.0:
            raise RuntimeError(
                f"Resumed a checkpoint into a model with {lora_params} lora_B tensors, but their total norm is "
                f"still exactly 0, so the adapter is the identity and this would sample the base model while "
                f"reporting a resumed step. DCP was asked for {len(requested)} tensors and {len(supplied)} came "
                f"back. Either the checkpoint holds no LoRA weights, or its key names disagree with this model -- "
                f"compare the LoRA rank, target_modules and enable_gradient_checkpointing_type in the config used "
                f"to evaluate against the one used to train, since the last inserts a '.checkpointed.' segment "
                f"into every key.")


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
