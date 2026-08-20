# SPDX-License-Identifier: Apache-2.0
"""Cached ``.pt`` dataset for Wan video-to-video depth-ControlNet training.

One cached sample holds the target clip's VAE latent, the pixel-aligned source
clip's VAE latent (the video-to-video condition), the text embedding, and
optionally the depth latents. Everything is pre-encoded, so training never loads
a VAE or a text encoder.

Expected keys per ``.pt``:

===========================  ====================================  ==========
key                          shape                                 required
===========================  ====================================  ==========
``vae_latent``               ``[16, T, H, W]`` target clip          yes
``control_latent``           ``[16, T, H, W]`` source clip          yes
``text_embedding``           ``[L, 4096]``                          yes
``text_attention_mask``      ``[L]``                                yes
``depth_latent``             ``[16, T, H, W]``                      when depth on
``depth_wide_latent``        ``[16, T, H, W]``                      when wide on
``clip_feature``             ``[257, 1280]``                        Wan 2.1 I2V only
``info``                     dict                                   no
===========================  ====================================  ==========

All latents share one ``(T, H, W)`` grid: depth is the depth of the *target*
clip's geometry, and the wide latent is the same clip re-rendered at a wider FOV
(same resolution, same frames, only ``fx,fy`` scaled by ``wide_fov_scale``).
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_TENSOR_KEYS = (
    "vae_latent",
    "control_latent",
    "text_embedding",
    "text_attention_mask",
    "clip_feature",
    "depth_latent",
    "depth_wide_latent",
)
_REQUIRED_KEYS = (
    "vae_latent",
    "control_latent",
    "text_embedding",
    "text_attention_mask",
)


def _list_dir_or_manifest(path: str) -> list[str]:
    expanded = os.path.expanduser(str(path))
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"v2v depth data_path does not exist: {expanded!r}")
    if os.path.isdir(expanded):
        paths = [os.path.join(expanded, name) for name in sorted(os.listdir(expanded)) if name.endswith(".pt")]
        if not paths:
            raise FileNotFoundError(f"No '*.pt' cached samples found under {expanded!r}")
        return paths

    base = os.path.dirname(expanded)
    paths = []
    with open(expanded, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line if os.path.isabs(line) else os.path.join(base, line))
    if not paths:
        raise FileNotFoundError(f"Manifest {expanded!r} lists no samples")
    return paths


def list_sample_paths(data_path: str | list[str] | dict[str, int]) -> list[str]:
    """Resolve ``data_path`` into a flat list of cached sample paths.

    A dict maps one source to an integer repeat count, which is how a mixture is
    weighted here: repeating a source's paths keeps the sampler itself uniform
    and the epoch deterministic, so resume stays exact.
    """
    if isinstance(data_path, dict):
        paths: list[str] = []
        for source, repeat in data_path.items():
            count = int(repeat)
            if count < 1:
                raise ValueError(f"data_path weight for {source!r} must be >= 1, got {repeat!r}")
            paths.extend(_list_dir_or_manifest(source) * count)
        return paths
    if isinstance(data_path, (list, tuple)):
        paths = []
        for source in data_path:
            paths.extend(_list_dir_or_manifest(source))
        return paths
    return _list_dir_or_manifest(data_path)


class WanV2VDepthCachedDataset(Dataset):
    """Map-style dataset over pre-encoded video-to-video control samples."""

    def __init__(
        self,
        data_path: str | list[str] | dict[str, int],
        *,
        include_depth: bool = True,
        include_wide: bool = False,
        sample_paths: list[str] | None = None,
    ) -> None:
        self.sample_paths = sample_paths if sample_paths is not None else list_sample_paths(data_path)
        self.include_depth = bool(include_depth)
        self.include_wide = bool(include_wide)
        logger.info("WanV2VDepthCachedDataset: %d samples (depth=%s, wide=%s)", len(self.sample_paths),
                    self.include_depth, self.include_wide)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # A truncated shard should cost one sample, not the run, so walk forward
        # until something loads.
        for offset in range(len(self.sample_paths)):
            path = self.sample_paths[(index + offset) % len(self.sample_paths)]
            try:
                sample = torch.load(path, map_location="cpu", weights_only=False)
                break
            except (EOFError, OSError, RuntimeError) as error:
                logger.warning("Skipping unreadable cached sample %s: %s", path, error)
        else:
            raise RuntimeError("No readable v2v depth cached samples remain.")

        if not isinstance(sample, dict):
            raise TypeError(f"Cached sample {path!r} must be a dict, got {type(sample).__name__}")
        missing = [key for key in _REQUIRED_KEYS if key not in sample]
        if missing:
            raise KeyError(f"Cached sample {path!r} missing required keys: {missing}")
        if self.include_depth and "depth_latent" not in sample:
            raise KeyError(f"enable_depth=true but cached sample {path!r} has no 'depth_latent'")
        if self.include_wide and "depth_wide_latent" not in sample:
            raise KeyError(f"enable_wide_fov=true but cached sample {path!r} has no 'depth_wide_latent'; "
                           "run scripts/v2v_depth/prepare_data/add_wide_depth_latent.py to backfill it.")

        sample.setdefault("info", {})
        if isinstance(sample["info"], dict):
            sample["info"].setdefault("sample_path", path)
        # Drop what this run did not ask for, so a disabled branch can never be
        # fed by a stale cache key.
        if not self.include_depth:
            sample.pop("depth_latent", None)
        if not self.include_wide:
            sample.pop("depth_wide_latent", None)
        if "clip_feature" not in sample:
            sample["clip_feature"] = torch.empty(0)
        return sample


def v2v_depth_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    for key in _TENSOR_KEYS:
        if key in batch[0]:
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    collated["info_list"] = [sample.get("info", {}) for sample in batch]
    return collated


def build_v2v_depth_train_dataloader(
    data_config: Any,
    *,
    num_sp_groups: int,
    sp_world_size: int,
    global_rank: int,
    include_depth: bool = True,
    include_wide: bool = False,
) -> StatefulDataLoader:
    from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler

    dataset = WanV2VDepthCachedDataset(
        data_config.data_path,
        include_depth=include_depth,
        include_wide=include_wide,
    )
    batch_size = int(getattr(data_config, "train_batch_size", 1) or 1)
    sampler = DP_SP_BatchSampler(
        batch_size=batch_size,
        dataset_size=len(dataset),
        num_sp_groups=int(num_sp_groups),
        sp_world_size=int(sp_world_size),
        global_rank=int(global_rank),
        drop_last=True,
        seed=int(getattr(data_config, "seed", 0) or 0),
    )
    if len(sampler) == 0:
        raise ValueError(f"v2v depth dataloader is empty: {len(dataset)} samples cannot fill one batch "
                         f"of {batch_size} per SP group (num_sp_groups={num_sp_groups}).")
    return StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=v2v_depth_collate,
        num_workers=int(getattr(data_config, "dataloader_num_workers", 0) or 0),
        pin_memory=False,
    )
