# SPDX-License-Identifier: Apache-2.0
"""Cached ``.pt`` dataset for MiniMax-H3 proxy-to-video training.

One cached sample holds the high-quality target clip's VAE latent, the proxy render's VAE latent
(the Ref2VA video reference), an optional RGB anchor frame latent, the Qwen3-VL text embedding with
its per-token modality tags, and an optional camera trajectory. Everything is pre-encoded, so
training loads neither a VAE nor a text encoder.

Expected keys per ``.pt``:

===========================  ==========================================  ==============
key                          shape                                       required
===========================  ==========================================  ==============
``vae_latent``               ``[24, T, H, W]`` target clip               yes
``proxy_latent``             ``[24, T_p, H_p, W_p]`` proxy render        yes
``text_embedding``           ``[L, 5120]``                               yes
``text_token_tags``          ``[L]`` int64, 0=vision 1=text              yes
``anchor_latent``            ``[24, 1, H_a, W_a]`` RGB first frame       when anchor on
``camera_extrinsics``        ``[F, 4, 4]`` world-to-camera               when camera on
``camera_intrinsics``        ``[F, 3, 3]`` pixel units                   when camera on
``depth_latent``             ``[24, T, H, W]`` target-clip depth         when depth on
``audio_latent``             ``[2, 32, A]``                              no
``info``                     dict, must carry ``pixel_size``             when camera on
===========================  ==========================================  ==============

The proxy latent deliberately does *not* share the target's grid. It is a reference, packed into
its own rows with its own rotary coordinates, and a coarser render is both cheaper and enough:
layout and motion survive downsampling in a way appearance would not.

Camera poses are stored rather than a precomputed ray field. Poses are four orders of magnitude
smaller on disk, and the normalization applied to them (:func:`fastvideo.pipelines.basic.minimax_h3
.camera.normalize_camera_trajectory`) is a modelling choice that should not be frozen into the
cache — changing it must not mean re-encoding every clip.
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
    "proxy_latent",
    "anchor_latent",
    "text_embedding",
    "text_token_tags",
    "camera_extrinsics",
    "camera_intrinsics",
    "depth_latent",
    "audio_latent",
)
_REQUIRED_KEYS = (
    "vae_latent",
    "proxy_latent",
    "text_embedding",
    "text_token_tags",
)


def _list_dir_or_manifest(path: str) -> list[str]:
    expanded = os.path.expanduser(str(path))
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"H3 proxy data_path does not exist: {expanded!r}")
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

    A dict maps one source to an integer repeat count, which is how a mixture is weighted here:
    repeating a source's paths keeps the sampler uniform and the epoch deterministic, so resume
    stays exact.
    """
    if isinstance(data_path, dict):
        paths: list[str] = []
        for source, repeat in data_path.items():
            count = int(repeat)
            if count < 1:
                raise ValueError(f"data_path weight for {source!r} must be >= 1, got {repeat!r}")
            paths.extend(_list_dir_or_manifest(source) * count)
        return paths
    if isinstance(data_path, list | tuple):
        paths = []
        for source in data_path:
            paths.extend(_list_dir_or_manifest(source))
        return paths
    return _list_dir_or_manifest(data_path)


class MiniMaxH3ProxyCachedDataset(Dataset):
    """Map-style dataset over pre-encoded proxy-to-video samples."""

    def __init__(
        self,
        data_path: str | list[str] | dict[str, int],
        *,
        include_anchor: bool = True,
        include_camera: bool = True,
        include_depth: bool = False,
        sample_paths: list[str] | None = None,
    ) -> None:
        self.sample_paths = sample_paths if sample_paths is not None else list_sample_paths(data_path)
        self.include_anchor = bool(include_anchor)
        self.include_camera = bool(include_camera)
        self.include_depth = bool(include_depth)
        logger.info("MiniMaxH3ProxyCachedDataset: %d samples (anchor=%s, camera=%s, depth=%s)", len(self.sample_paths),
                    self.include_anchor, self.include_camera, self.include_depth)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # A truncated shard should cost one sample, not the run, so walk forward until something
        # loads.
        for offset in range(len(self.sample_paths)):
            path = self.sample_paths[(index + offset) % len(self.sample_paths)]
            try:
                sample = torch.load(path, map_location="cpu", weights_only=False)
                break
            except (EOFError, OSError, RuntimeError) as error:
                logger.warning("Skipping unreadable cached sample %s: %s", path, error)
        else:
            raise RuntimeError("No readable H3 proxy cached samples remain.")

        if not isinstance(sample, dict):
            raise TypeError(f"Cached sample {path!r} must be a dict, got {type(sample).__name__}")
        missing = [key for key in _REQUIRED_KEYS if key not in sample]
        if missing:
            raise KeyError(f"Cached sample {path!r} missing required keys: {missing}")
        if self.include_anchor and "anchor_latent" not in sample:
            raise KeyError(f"enable_anchor=true but cached sample {path!r} has no 'anchor_latent'")
        if not self.include_anchor and "anchor_latent" in sample:
            # The cached text embedding was tokenized around a `<Picture 1>` block. Packing the
            # layout without the anchor would leave those vision tokens describing a reference that
            # is not in the sequence, which nothing downstream can detect.
            raise KeyError(f"enable_anchor=false but cached sample {path!r} was encoded with an anchor; re-encode "
                           "without one rather than dropping it here.")
        if self.include_camera and not {"camera_extrinsics", "camera_intrinsics"} <= sample.keys():
            raise KeyError(f"enable_camera_controlnet=true but cached sample {path!r} has no camera trajectory; "
                           "add a 'camera' field naming an .npz to the encoder's manifest and re-encode.")
        if self.include_depth and "depth_latent" not in sample:
            raise KeyError(f"enable_control_depth=true but cached sample {path!r} has no 'depth_latent'")

        sample.setdefault("info", {})
        if isinstance(sample["info"], dict):
            sample["info"].setdefault("sample_path", path)
        # Drop what this run did not ask for, so a disabled branch can never be fed by a stale
        # cache key.
        if not self.include_anchor:
            sample.pop("anchor_latent", None)
        if not self.include_camera:
            sample.pop("camera_extrinsics", None)
            sample.pop("camera_intrinsics", None)
        if not self.include_depth:
            sample.pop("depth_latent", None)
        return sample


def proxy_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    for key in _TENSOR_KEYS:
        if key in batch[0]:
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    collated["info_list"] = [sample.get("info", {}) for sample in batch]
    return collated


def build_proxy_train_dataloader(
    data_config: Any,
    *,
    num_sp_groups: int,
    sp_world_size: int,
    global_rank: int,
    include_anchor: bool = True,
    include_camera: bool = True,
    include_depth: bool = False,
) -> StatefulDataLoader:
    from fastvideo.dataset.parquet_dataset_map_style import DP_SP_BatchSampler

    dataset = MiniMaxH3ProxyCachedDataset(
        data_config.data_path,
        include_anchor=include_anchor,
        include_camera=include_camera,
        include_depth=include_depth,
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
        raise ValueError(f"H3 proxy dataloader is empty: {len(dataset)} samples cannot fill one batch of "
                         f"{batch_size} per SP group (num_sp_groups={num_sp_groups}).")
    return StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=proxy_collate,
        num_workers=int(getattr(data_config, "dataloader_num_workers", 0) or 0),
        pin_memory=False,
    )


__all__ = [
    "MiniMaxH3ProxyCachedDataset",
    "build_proxy_train_dataloader",
    "list_sample_paths",
    "proxy_collate",
]
