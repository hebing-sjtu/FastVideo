# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(getattr(obj, "load_state_dict", None))


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _parse_step_from_dir(checkpoint_dir: Path) -> int:
    match = _CHECKPOINT_DIR_RE.match(checkpoint_dir.name)
    if not match:
        raise ValueError(f"Invalid checkpoint directory name {checkpoint_dir.name!r}; "
                         "expected 'checkpoint-<step>'")
    return int(match.group(1))


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if not _CHECKPOINT_DIR_RE.match(child.name):
            continue
        if not (child / "dcp").is_dir():
            continue
        try:
            step = _parse_step_from_dir(child)
        except Exception:
            continue
        candidates.append((step, child))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_resume_checkpoint(resume_from_checkpoint: str, *, output_dir: str) -> Path | None:
    """Resolve a user-provided resume path to a concrete checkpoint dir.

    Accepted values:
    - "latest" (auto-pick latest checkpoint-*/dcp under output_dir,
      or ``None`` if no checkpoint exists yet — starts from scratch)
    - /path/to/output_dir/checkpoint-<step>
    - /path/to/output_dir/checkpoint-<step>/dcp
    - /path/to/output_dir (auto-pick latest checkpoint-*/dcp)
    """

    if str(resume_from_checkpoint).strip().lower() == "latest":
        out = Path(os.path.expanduser(str(output_dir))).resolve()
        latest = _find_latest_checkpoint(out)
        if latest is None:
            logger.info(
                "resume_from_checkpoint='latest' but no "
                "checkpoints found under %s; starting from "
                "scratch.",
                out,
            )
        return latest

    raw = os.path.expanduser(str(resume_from_checkpoint))
    path = Path(raw).resolve()
    if not path.exists():
        raise FileNotFoundError(f"resume_from_checkpoint not found: {path}")

    if path.is_dir() and path.name == "dcp":
        path = path.parent

    if path.is_dir() and _CHECKPOINT_DIR_RE.match(path.name):
        if not (path / "dcp").is_dir():
            raise FileNotFoundError(f"Missing dcp dir under checkpoint: {path / 'dcp'}")
        return path

    # Treat as output_dir -> pick latest.
    latest = _find_latest_checkpoint(path)
    if latest is not None:
        return latest

    # Give a clearer error message.
    out = Path(os.path.expanduser(str(output_dir))).resolve()
    raise ValueError("Could not resolve resume checkpoint. Expected a checkpoint directory "
                     f"named 'checkpoint-<step>' (with 'dcp/' inside), or an output_dir "
                     f"containing such checkpoints. Got: {path} (output_dir={out}).")


def _resolve_model_weight_dcp_dir(checkpoint_path: str) -> Path:
    """Resolve an explicit ``checkpoint-<step>`` or ``dcp/`` path."""
    path = Path(os.path.expanduser(str(checkpoint_path))).resolve()
    dcp_dir = path if path.name == "dcp" else path / "dcp"
    if not dcp_dir.is_dir():
        raise FileNotFoundError(f"Model-weight checkpoint has no dcp/ directory: {path}")
    if not (dcp_dir / ".metadata").is_file():
        raise FileNotFoundError(f"Model-weight checkpoint is incomplete (missing {dcp_dir / '.metadata'}).")
    return dcp_dir


class _PrefixedModelWeightLoader(Stateful):
    """Map a prefixed DCP model state onto a direct target module.

    The wrapper deliberately exposes only tensor keys found under one source
    model prefix, so DCP never plans reads for optimizer, scheduler, dataloader,
    callback, RNG, or any sibling model state.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        source_metadata: dict[str, Any],
        source_prefix: str,
        required_target_prefixes: tuple[str, ...] | None = None,
        optional_target_substrings: tuple[str, ...] | None = None,
    ) -> None:
        self.model = model
        self.source_prefix = source_prefix
        self.loaded_tensor_count = 0
        self.ignored_missing_target_count = 0
        # `optional_target_substrings` mark model keys allowed to be absent from
        # the checkpoint, e.g. a newly-added branch layered onto an older save.
        # Substring rather than prefix matching so mid-key markers like `.wca.`
        # are covered. Declaring any forces a non-strict `set_model_state_dict`
        # and leaves those params at their constructed value for the caller.
        self.optional_target_substrings = tuple(optional_target_substrings or ())
        self._strict_load = (required_target_prefixes is None and not self.optional_target_substrings)

        target_state = get_model_state_dict(model)
        self.source_to_target: dict[str, str] = {}
        unexpected_source_keys: list[str] = []
        shape_mismatches: list[str] = []
        for source_name, metadata in source_metadata.items():
            if not source_name.startswith(source_prefix):
                continue
            target_name = source_name[len(source_prefix):]
            target_tensor = target_state.get(target_name)
            if target_tensor is None:
                unexpected_source_keys.append(target_name)
                continue
            source_size = getattr(metadata, "size", None)
            if source_size is None:
                raise TypeError(f"DCP model entry {source_name!r} is not tensor metadata")
            if tuple(source_size) != tuple(target_tensor.shape):
                shape_mismatches.append(
                    f"{target_name}: checkpoint={tuple(source_size)}, target={tuple(target_tensor.shape)}")
                continue
            self.source_to_target[source_name] = target_name

        covered_targets = set(self.source_to_target.values())
        all_missing_target_keys = sorted(set(target_state).difference(covered_targets))
        missing_target_keys = all_missing_target_keys
        if required_target_prefixes is not None:
            # A partial overlay names the subtree it insists on, e.g.
            # `depth_controlnet.` layered on a backbone loaded from `init_from`.
            missing_target_keys = [key for key in missing_target_keys if key.startswith(required_target_prefixes)]
        if self.optional_target_substrings:
            missing_target_keys = [
                key for key in missing_target_keys if not any(sub in key for sub in self.optional_target_substrings)
            ]
        self.ignored_missing_target_count = len(all_missing_target_keys) - len(missing_target_keys)
        if unexpected_source_keys or shape_mismatches or missing_target_keys:
            details = []
            if unexpected_source_keys:
                details.append("checkpoint keys absent from target: " + ", ".join(unexpected_source_keys[:8]))
            if shape_mismatches:
                details.append("shape mismatches: " + "; ".join(shape_mismatches[:8]))
            if missing_target_keys:
                details.append("checkpoint missing target keys: " + ", ".join(missing_target_keys[:8]))
            raise ValueError("Checkpoint does not match this model; align the enabled control modalities and the "
                             "ControlNet architecture before loading (" + " | ".join(details) + ")")
        if not self.source_to_target:
            raise ValueError(f"No model tensors matched DCP source prefix {source_prefix!r}")

    def state_dict(self) -> dict[str, Any]:
        target_state = get_model_state_dict(self.model)
        return {source_name: target_state[target_name] for source_name, target_name in self.source_to_target.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        missing = set(self.source_to_target).difference(state_dict)
        if missing:
            raise KeyError(f"DCP did not return requested model tensors: {sorted(missing)[:8]}")
        target_state = {
            self.source_to_target[source_name]: tensor
            for source_name, tensor in state_dict.items() if source_name in self.source_to_target
        }
        set_model_state_dict(
            self.model,
            model_state_dict=target_state,
            options=StateDictOptions(strict=self._strict_load),
        )
        self.loaded_tensor_count = len(target_state)


def load_prefixed_model_weights_from_dcp(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    state_key: str,
    source_prefix: str,
    required_target_prefixes: tuple[str, ...] | None = None,
    optional_target_substrings: tuple[str, ...] | None = None,
) -> int:
    """Load only one prefixed model state from a DCP training checkpoint.

    This never constructs or loads optimizer, scheduler, dataloader, callback, or
    RNG state; extra DCP entries are ignored by omitting them from the requested
    state dictionary. By default every target model key must be present under
    ``source_prefix``; ``required_target_prefixes`` narrows that check for
    partial overlays such as a ControlNet-only checkpoint layered on top of a
    base backbone loaded from ``init_from``.
    """
    dcp_dir = _resolve_model_weight_dcp_dir(checkpoint_path)
    normalized_prefix = source_prefix.rstrip(".")
    if normalized_prefix:
        normalized_prefix = f"{normalized_prefix}."
    metadata_root = f"{state_key}."
    reader = dcp.FileSystemReader(str(dcp_dir))
    metadata = reader.read_metadata()
    source_metadata = {
        full_name[len(metadata_root):]: value
        for full_name, value in metadata.state_dict_metadata.items()
        if full_name.startswith(f"{metadata_root}{normalized_prefix}")
    }
    if not source_metadata:
        raise KeyError(f"No model weights under {state_key}.{normalized_prefix} in {dcp_dir}")

    loader = _PrefixedModelWeightLoader(
        model,
        source_metadata=source_metadata,
        source_prefix=normalized_prefix,
        required_target_prefixes=required_target_prefixes,
        optional_target_substrings=optional_target_substrings,
    )
    if loader.ignored_missing_target_count:
        logger.info(
            "Ignoring %d target model key(s) absent from %s; required target prefixes=%s.",
            loader.ignored_missing_target_count,
            dcp_dir,
            required_target_prefixes,
        )
    logger.info(
        "Loading %d model tensors only from %s (state=%s, prefix=%s); "
        "optimizer/scheduler/dataloader/callback/RNG states are not requested.",
        len(loader.source_to_target),
        dcp_dir,
        state_key,
        normalized_prefix,
    )
    dcp.load({state_key: loader}, storage_reader=reader)
    _barrier()
    if loader.loaded_tensor_count != len(loader.source_to_target):
        raise RuntimeError(f"Expected {len(loader.source_to_target)} model tensors, "
                           f"loaded {loader.loaded_tensor_count}")
    return loader.loaded_tensor_count


def dcp_model_has_key_substring(
    checkpoint_path: str,
    *,
    state_key: str,
    source_prefix: str,
    substring: str,
) -> bool:
    """Return True if the DCP save has any model tensor key containing ``substring``.

    Reads only DCP metadata (no tensor reads, no collectives), so a warm-start
    seed can be applied only when the branch is genuinely absent.
    """
    dcp_dir = _resolve_model_weight_dcp_dir(checkpoint_path)
    normalized_prefix = source_prefix.rstrip(".")
    if normalized_prefix:
        normalized_prefix = f"{normalized_prefix}."
    full_prefix = f"{state_key}.{normalized_prefix}"
    metadata = dcp.FileSystemReader(str(dcp_dir)).read_metadata()
    return any(
        full_name.startswith(full_prefix) and substring in full_name for full_name in metadata.state_dict_metadata)


class _RoleModuleContainer(torch.nn.Module):
    """Ephemeral container to expose multiple role modules as a single
    ``nn.Module``.

    Used by ``OptimizerWrapper`` which expects a single root module
    covering all parameters owned by the optimizer.
    """

    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        super().__init__()
        for name, module in modules.items():
            self.add_module(name, module)


class _FullModelState(Stateful):
    """DCP wrapper that saves frozen model parameters too.

    The shared ``ModelWrapper`` intentionally filters to ``requires_grad``
    parameters. Frozen-but-mutated roles (e.g. DiffusionNFT's old policy,
    causal-CD's EMA target) must still be restored on resume, so they need
    full model state.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model)  # type: ignore[no-any-return]

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )


class _CallbackStateWrapper:
    """Wraps a CallbackDict for DCP save/load."""

    def __init__(self, callbacks: Any) -> None:
        self._callbacks = callbacks

    def state_dict(self) -> dict[str, Any]:
        return self._callbacks.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        self._callbacks.load_state_dict(state_dict)


@dataclass(slots=True)
class CheckpointConfig:
    save_steps: int
    keep_last: int


class CheckpointManager:
    """Role-based checkpoint manager for training runtime.

    - Checkpoint policy lives in YAML (via TrainingArgs fields).
    - Resume path is typically provided via CLI (``--resume-from-checkpoint``).
    """

    def __init__(
        self,
        *,
        method: Any,
        dataloader: Any,
        output_dir: str,
        config: CheckpointConfig,
        callbacks: Any | None = None,
        raw_config: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.dataloader = dataloader
        self.output_dir = str(output_dir)
        self.config = config
        self._callbacks = callbacks
        self._raw_config = raw_config
        self._last_saved_step: int | None = None

    def _build_states(self) -> dict[str, Any]:
        states: dict[str, Any] = self.method.checkpoint_state()

        # Dataloader (optional but recommended for exact resume).
        if _is_stateful(self.dataloader):
            states["dataloader"] = self.dataloader

        # Callback state (e.g. EMA shadow weights, validation RNG).
        if self._callbacks is not None and _is_stateful(self._callbacks):
            states["callbacks"] = _CallbackStateWrapper(self._callbacks, )

        return states

    def _checkpoint_dir(self, step: int) -> Path:
        return Path(self.output_dir) / f"checkpoint-{step}"

    def _dcp_dir(self, step: int) -> Path:
        return self._checkpoint_dir(step) / "dcp"

    def maybe_save(self, step: int) -> None:
        save_steps = int(self.config.save_steps or 0)
        if save_steps <= 0:
            return
        if step % save_steps != 0:
            return
        if self._last_saved_step == step:
            return
        self.save(step)

    def save_final(self, step: int) -> None:
        save_steps = int(self.config.save_steps or 0)
        if save_steps <= 0:
            return
        self.save(step)

    def save(self, step: int) -> None:
        checkpoint_dir = self._checkpoint_dir(step)
        dcp_dir = self._dcp_dir(step)
        os.makedirs(dcp_dir, exist_ok=True)

        states = self._build_states()
        if _rank() == 0:
            logger.info(
                "Saving checkpoint to %s",
                checkpoint_dir,
            )
            self._write_metadata(checkpoint_dir, step)
        dcp.save(states, checkpoint_id=str(dcp_dir))
        _barrier()

        # Save RNG state AFTER dcp.save so it captures the
        # exact state the continuous run continues with.
        # dcp.save triggers FSDP all-gather ops that can
        # advance the RNG between when DCP captures it and
        # when the save completes.
        self._save_rng_snapshot(checkpoint_dir)
        _barrier()

        self._last_saved_step = step

        self._cleanup_old_checkpoints()

    def _write_metadata(
        self,
        checkpoint_dir: Path,
        step: int,
    ) -> None:
        metadata: dict[str, Any] = {"step": step}
        if self._raw_config is not None:
            metadata["config"] = self._raw_config
        meta_path = checkpoint_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_metadata(checkpoint_dir: str | Path, ) -> dict[str, Any]:
        """Read ``metadata.json`` from a checkpoint dir."""
        meta_path = Path(checkpoint_dir) / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No metadata.json in {checkpoint_dir}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def _save_rng_snapshot(self, checkpoint_dir: Path) -> None:
        """Save per-rank RNG state to a separate file.

        Called AFTER ``dcp.save`` so the snapshot reflects
        the exact state the continuous run continues with.
        Each rank saves its own file because CUDA RNG and
        custom generators differ across ranks.
        """
        rank = _rank()
        rng: dict[str, Any] = {
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        rng["cuda_rng"] = torch.cuda.get_rng_state()
        rng["gen_cuda"] = self.method.cuda_generator.get_state()
        torch.save(
            rng,
            checkpoint_dir / f"rng_state_rank{rank}.pt",
        )

    def load_rng_snapshot(
        self,
        checkpoint_path: str,
    ) -> None:
        """Restore per-rank RNG state from the snapshot file.

        Must be called AFTER ``dcp.load`` **and** after
        ``iter(dataloader)`` so no later operation can
        clobber the restored state.
        """
        resolved = _resolve_resume_checkpoint(
            checkpoint_path,
            output_dir=self.output_dir,
        )
        if resolved is None:
            return
        rank = _rank()
        rng_path = resolved / f"rng_state_rank{rank}.pt"
        if not rng_path.is_file():
            # Fall back to legacy single-file snapshot.
            rng_path = resolved / "rng_state.pt"
        if not rng_path.is_file():
            logger.warning(
                "No rng_state in %s; skipping "
                "RNG snapshot restore.",
                resolved,
            )
            return

        rng = torch.load(
            rng_path,
            map_location="cpu",
            weights_only=False,
        )
        if "torch_rng" in rng:
            torch.set_rng_state(rng["torch_rng"])
        if "python_rng" in rng:
            random.setstate(rng["python_rng"])
        if "numpy_rng" in rng:
            np.random.set_state(rng["numpy_rng"])

        torch.cuda.set_rng_state(rng["cuda_rng"])
        self.method.cuda_generator.set_state(rng["gen_cuda"])
        logger.info(
            "Restored RNG snapshot from %s",
            rng_path,
        )

    def maybe_resume(self, *, resume_from_checkpoint: str | None) -> int | None:
        if not resume_from_checkpoint:
            return None

        resolved = _resolve_resume_checkpoint(
            resume_from_checkpoint,
            output_dir=self.output_dir,
        )
        if resolved is None:
            return None
        step = _parse_step_from_dir(resolved)

        states = self._build_states()
        logger.info("Loading Phase 2 checkpoint from %s", resolved)
        dcp.load(states, checkpoint_id=str(resolved / "dcp"))
        _barrier()
        logger.info("Checkpoint loaded; resuming from step=%s", step)
        return step

    def _cleanup_old_checkpoints(self) -> None:
        keep_last = int(self.config.keep_last or 0)
        if keep_last <= 0:
            return

        if _rank() != 0:
            _barrier()
            return

        output_dir = Path(self.output_dir)
        candidates: list[tuple[int, Path]] = []
        for child in output_dir.iterdir():
            if not child.is_dir():
                continue
            if not _CHECKPOINT_DIR_RE.match(child.name):
                continue
            try:
                step = _parse_step_from_dir(child)
            except Exception:
                continue
            candidates.append((step, child))

        candidates.sort(key=lambda x: x[0])
        to_delete = candidates[:-keep_last] if len(candidates) > keep_last else []
        for step, path in to_delete:
            logger.info("Removing old checkpoint (keep_last=%s): %s", keep_last, path)
            shutil.rmtree(path, ignore_errors=True)

        _barrier()
