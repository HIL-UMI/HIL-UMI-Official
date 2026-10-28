# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Single-Franka tcp_rot6d SFT data loader for the openpi_pytorch path.

The loader owns dataset access, distributed sampling, and collation into the
openpi_pytorch ``Observation`` type. Input semantics come from the canonical
RLinf OpenPI config selected by ``config_name``::

    repack LeRobot keys
    -> SingleFrankaTcpRot6dInputs (camera map, 10-d slice, pad)
    -> RigidBodyDeltaActions (body-frame SE(3); gripper absolute)
    -> quantile Normalize
    -> ModelTransformFactory (resize + tokenize)
    -> collate -> (Observation, actions [B, H, D])
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import typing
from pathlib import Path

import numpy as np
import torch
from openpi.transforms import DataTransformFn, compose
from torch.utils.data.distributed import DistributedSampler

from training.data.datasets.openpi_pytorch.single_franka.acbc import (
    PromptFromLeRobotTask,
    validate_acbc_dataset,
)
from training.data.datasets.openpi_pytorch.single_franka.single_franka_sft_dataset import (
    SingleFrankaSftDataset,
)
from training.data.datasets.two_stream_batch_sampler import TwoStreamBatchSampler
from training.data.lerobot_paths import resolve_lerobot_repo_id
from training.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from training.models.embodiment.openpi_pytorch.transforms_pipeline import (
    build_openpi_transforms,
)
from training.utils.logging import get_logger

logger = get_logger()

__all__ = [
    "SingleFrankaSftDataConfig",
    "SingleFrankaSftDataLoader",
    "build_single_franka_sft_dataloader",
    "collate_single_franka_sft_items",
    "create_single_franka_sft_data_loader",
]

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

_REPACK_KEYS = {
    "observation/image": ("image",),
    "observation/extra_view_image": ("extra_view_image", "extra_view_image-0"),
    "observation/state": ("state",),
    "actions": ("actions",),
}


class _Repack(DataTransformFn):
    """Map raw single-Franka LeRobot keys to canonical OpenPI input keys."""

    def __call__(self, frame: dict[str, typing.Any]) -> dict[str, typing.Any]:
        data: dict[str, typing.Any] = {}
        for destination, sources in _REPACK_KEYS.items():
            source = next((key for key in sources if key in frame), None)
            if source is None:
                raise KeyError(
                    f"single_franka SFT frame missing one of {sources!r} "
                    f"(needed for {destination!r}); "
                    f"available keys={sorted(frame.keys())}"
                )
            data[destination] = np.asarray(frame[source])

        prompt = frame.get("prompt", frame.get("task"))
        if prompt is None:
            raise ValueError(
                "single_franka SFT frame is missing both 'prompt' and 'task'; "
                "ensure the LeRobot dataset contains task annotations."
            )
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        data["prompt"] = prompt
        return data


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: SingleFrankaSftDataset,
        transform: typing.Callable[[typing.Any], typing.Any],
    ) -> None:
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, idx: int) -> typing.Any:
        return self._transform(self._dataset[idx])

    def __len__(self) -> int:
        return len(self._dataset)


class _TwoDataset(torch.utils.data.Dataset):
    """Index two datasets as one, preserving the first dataset's index range."""

    def __init__(
        self,
        primary: torch.utils.data.Dataset,
        correction: torch.utils.data.Dataset,
    ) -> None:
        self._primary = primary
        self._correction = correction
        self._correction_offset = len(primary)

    @property
    def correction_offset(self) -> int:
        """Return the index offset where the correction dataset starts."""
        return self._correction_offset

    def __getitem__(self, idx: int) -> typing.Any:
        if idx < self._correction_offset:
            return self._primary[idx]
        return self._correction[idx - self._correction_offset]

    def __len__(self) -> int:
        return self._correction_offset + len(self._correction)


def collate_single_franka_sft_items(
    items: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> tuple[Observation, torch.Tensor]:
    """Collate transformed items into ``(Observation, actions)``."""
    if not items:
        raise ValueError("Cannot collate an empty single_franka SFT batch.")

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    batch = {
        "image": images,
        "image_mask": image_masks,
        "state": torch.from_numpy(
            np.stack([np.asarray(item["state"], dtype=np.float32) for item in items])
        ),
        "tokenized_prompt": torch.from_numpy(
            np.stack(
                [np.asarray(item["tokenized_prompt"], dtype=np.int64) for item in items]
            )
        ).long(),
        "tokenized_prompt_mask": torch.from_numpy(
            np.stack(
                [
                    np.asarray(item["tokenized_prompt_mask"], dtype=np.bool_)
                    for item in items
                ]
            )
        ),
    }
    actions = torch.from_numpy(
        np.stack([np.asarray(item["actions"], dtype=np.float32) for item in items])
    )
    return Observation.from_dict(batch), actions


@dataclasses.dataclass(frozen=True)
class SingleFrankaSftDataConfig:
    """Resolved metadata exposed by :class:`SingleFrankaSftDataLoader`."""

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int
    prompt_from_task: bool
    acbc: bool
    mix_acbc: bool


def _worker_init_fn(worker_id: int) -> None:
    del worker_id


def _as_bool(value: typing.Any) -> bool:
    """Parse bool-like Hydra CLI values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def create_single_franka_sft_data_loader(
    *,
    data_path: str,
    model_path: str,
    config_name: str,
    assets_dir: str,
    asset_id: str,
    action_dim: int,
    action_horizon: int,
    max_token_len: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    fps: int | None,
    tolerance_s: float,
    dist_rank: int,
    dist_world_size: int,
    data_kwargs: dict | None = None,
    prompt_from_task: bool = False,
    acbc: bool = False,
    mix_acbc_data_path: str | None = None,
    mix_acbc_primary_ratio: float = 0.5,
    mix_acbc_secondary_ratio: float = 0.5,
    post_train_data_path: str | None = None,
    post_train_demo_ratio: float = 0.5,
    post_train_correction_ratio: float = 0.5,
) -> "SingleFrankaSftDataLoader":
    """Build a single-franka SFT loader yielding ``(Observation, actions)``."""
    if acbc and not prompt_from_task:
        raise ValueError("ACBC requires prompt_from_task=True.")
    if mix_acbc_data_path is not None and not acbc:
        raise ValueError("Mix ACBC requires acbc=True.")
    if mix_acbc_data_path is not None and post_train_data_path is not None:
        raise ValueError(
            "Mix ACBC and post-train secondary datasets cannot be enabled together."
        )
    secondary_data_path = mix_acbc_data_path or post_train_data_path
    mix_acbc = mix_acbc_data_path is not None
    primary_ratio = mix_acbc_primary_ratio if mix_acbc else post_train_demo_ratio
    secondary_ratio = (
        mix_acbc_secondary_ratio if mix_acbc else post_train_correction_ratio
    )
    dataset = SingleFrankaSftDataset(
        data_path=data_path,
        action_horizon=action_horizon,
        fps=fps,
        tolerance_s=tolerance_s,
    )

    input_transforms, _ = build_openpi_transforms(
        model_path,
        config_name,
        data_kwargs=data_kwargs,
        norm_stats_dir=assets_dir,
        norm_stats_asset_id=asset_id,
    )
    transform_steps: list[typing.Callable[[typing.Any], typing.Any]] = []
    if prompt_from_task:
        transform_steps.append(PromptFromLeRobotTask(dataset.meta.tasks))
    transform = compose([*transform_steps, _Repack(), *input_transforms])
    source = _TransformedDataset(
        dataset,
        transform,
    )

    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(seed)

    if secondary_data_path is None:
        sampler: typing.Any = DistributedSampler(
            source,
            num_replicas=dist_world_size,
            rank=dist_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        logger.info(
            "single_franka SFT data loader: batch_size=%d, num_workers=%d, "
            "action_horizon=%d, transforms=%s, prompt_from_task=%s, acbc=%s",
            batch_size,
            num_workers,
            action_horizon,
            config_name,
            prompt_from_task,
            acbc,
        )

        torch_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, source),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=collate_single_franka_sft_items,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            pin_memory=True,
        )
    else:
        correction_dataset = SingleFrankaSftDataset(
            data_path=secondary_data_path,
            action_horizon=action_horizon,
            fps=fps,
            tolerance_s=tolerance_s,
        )
        correction_transform_steps: list[typing.Callable[[typing.Any], typing.Any]] = []
        if prompt_from_task:
            correction_transform_steps.append(
                PromptFromLeRobotTask(correction_dataset.meta.tasks)
            )
        correction_transform = compose(
            [*correction_transform_steps, _Repack(), *input_transforms]
        )
        correction_source = _TransformedDataset(
            correction_dataset, correction_transform
        )
        mixed_source = _TwoDataset(source, correction_source)
        primary_sampler = DistributedSampler(
            source,
            num_replicas=dist_world_size,
            rank=dist_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        correction_sampler = DistributedSampler(
            correction_source,
            num_replicas=dist_world_size,
            rank=dist_rank,
            shuffle=shuffle,
            seed=seed + 1,
            drop_last=False,
        )
        sampler = TwoStreamBatchSampler(
            primary_sampler=primary_sampler,
            secondary_sampler=correction_sampler,
            secondary_offset=mixed_source.correction_offset,
            batch_size=batch_size,
            primary_ratio=primary_ratio,
            secondary_ratio=secondary_ratio,
        )
        first_primary_batch_size, first_secondary_batch_size = sampler.first_batch_split
        mode_name = "Mix ACBC" if mix_acbc else "post-train SFT"
        logger.info(
            "single_franka %s data loader: batch_size=%d "
            "(requested ratio %.6g primary : %.6g secondary; first batch "
            "%d primary + %d secondary; epoch samples %d primary + %d "
            "secondary), num_workers=%d, action_horizon=%d, transforms=%s, "
            "primary_path=%s, secondary_path=%s",
            mode_name,
            batch_size,
            primary_ratio,
            secondary_ratio,
            first_primary_batch_size,
            first_secondary_batch_size,
            sampler.primary_samples_per_epoch,
            sampler.secondary_samples_per_epoch,
            num_workers,
            action_horizon,
            config_name,
            dataset.dataset_root,
            correction_dataset.dataset_root,
        )

        torch_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, mixed_source),
            batch_sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=collate_single_franka_sft_items,
            worker_init_fn=_worker_init_fn,
            generator=generator,
            pin_memory=True,
        )

    data_config = SingleFrankaSftDataConfig(
        repo_id=str(dataset.dataset_root),
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        prompt_from_task=prompt_from_task,
        acbc=acbc,
        mix_acbc=mix_acbc,
    )
    return SingleFrankaSftDataLoader(torch_loader, data_config, sampler)


class SingleFrankaSftDataLoader:
    """Infinite ``(Observation, actions)`` loop over single-franka SFT data."""

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: SingleFrankaSftDataConfig,
        sampler: typing.Any | None = None,
    ) -> None:
        self._torch_loader = torch_loader
        self._data_config = data_config
        self._sampler = sampler
        self._epoch = 0

    def data_config(self) -> SingleFrankaSftDataConfig:
        """Return the resolved single-Franka data configuration."""
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """Return the underlying PyTorch dataloader."""
        return self._torch_loader

    def __iter__(
        self,
    ) -> typing.Iterator[tuple[Observation, torch.Tensor]]:
        while True:
            if self._sampler is not None:
                self._sampler.set_epoch(self._epoch)
                self._epoch += 1
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def build_single_franka_sft_dataloader(
    cfg: typing.Any,
    world_size: int,
    rank: int,
    data_paths: typing.Any,
    eval_dataset: bool = False,
) -> tuple[SingleFrankaSftDataLoader, SingleFrankaSftDataConfig]:
    """Build the single-franka openpi_pytorch loader for the SFT worker."""
    from omegaconf import OmegaConf

    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError(
            "openpi_pytorch single_franka SFT requires data.train_data_paths."
        )

    model_cfg = cfg.actor.model
    data_cfg = cfg.data
    openpi_cfg = model_cfg.openpi

    repo_id = OmegaConf.select(data_cfg, "repo_id", default=None)
    if repo_id:
        from training.data.lerobot_paths import default_hf_lerobot_home

        candidate = default_hf_lerobot_home() / str(repo_id)
        nested = Path(data_path) / str(repo_id)
        if (candidate / "meta" / "info.json").is_file():
            data_path = str(candidate)
        elif (nested / "meta" / "info.json").is_file():
            data_path = str(nested)

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    fps = OmegaConf.select(data_cfg, "fps", default=None)
    fps = int(fps) if fps is not None else None
    tolerance_s = float(OmegaConf.select(data_cfg, "tolerance_s", default=1e-4))
    post_train = _as_bool(OmegaConf.select(data_cfg, "post_train", default=False))
    mix_acbc = _as_bool(OmegaConf.select(data_cfg, "mix_acbc", default=False))
    if post_train and mix_acbc:
        raise ValueError(
            "data.post_train=true cannot be combined with data.mix_acbc=true."
        )
    if mix_acbc and not _as_bool(OmegaConf.select(data_cfg, "acbc", default=False)):
        raise ValueError("data.mix_acbc=true requires data.acbc=true.")
    post_train_data_path = None
    if post_train and not eval_dataset:
        post_train_data_path = resolve_lerobot_repo_id(
            OmegaConf.select(data_cfg, "post_train_data_path", default=None)
        )
        if not post_train_data_path:
            raise ValueError(
                "data.post_train=true requires data.post_train_data_path to point "
                "to a LeRobot-format online correction dataset."
            )
    post_train_demo_ratio = float(
        OmegaConf.select(data_cfg, "post_train_demo_ratio", default=0.5)
    )
    post_train_correction_ratio = float(
        OmegaConf.select(data_cfg, "post_train_correction_ratio", default=0.5)
    )
    mix_acbc_data_path = None
    if mix_acbc and not eval_dataset:
        mix_acbc_data_path = resolve_lerobot_repo_id(
            OmegaConf.select(data_cfg, "mix_acbc_data_path", default=None)
        )
        if not mix_acbc_data_path:
            raise ValueError(
                "data.mix_acbc=true requires data.mix_acbc_data_path to point "
                "to a second ACBC dataset."
            )
    mix_acbc_primary_ratio = float(
        OmegaConf.select(data_cfg, "mix_acbc_primary_ratio", default=0.5)
    )
    mix_acbc_secondary_ratio = float(
        OmegaConf.select(data_cfg, "mix_acbc_secondary_ratio", default=0.5)
    )
    prompt_from_task = _as_bool(
        OmegaConf.select(data_cfg, "prompt_from_task", default=False)
    )
    acbc = _as_bool(OmegaConf.select(data_cfg, "acbc", default=False))
    if acbc:
        if not prompt_from_task:
            raise ValueError("data.acbc=true requires data.prompt_from_task=true.")
        repo_root = Path(__file__).resolve().parents[5]
        summary = validate_acbc_dataset(data_path, repo_root=repo_root)
        logger.info(
            "Validated ACBC dataset: root=%s, episodes=%d, frames=%d, "
            "videos=%d, task_counts=%s, tasks=%s",
            summary.dataset_root,
            summary.total_episodes,
            summary.total_frames,
            summary.video_files,
            summary.task_counts,
            summary.tasks,
        )
        if mix_acbc_data_path is not None:
            secondary_summary = validate_acbc_dataset(
                mix_acbc_data_path, repo_root=repo_root
            )
            logger.info(
                "Validated secondary Mix ACBC dataset: root=%s, episodes=%d, "
                "frames=%d, videos=%d, task_counts=%s, tasks=%s",
                secondary_summary.dataset_root,
                secondary_summary.total_episodes,
                secondary_summary.total_frames,
                secondary_summary.video_files,
                secondary_summary.task_counts,
                secondary_summary.tasks,
            )

    loader = create_single_franka_sft_data_loader(
        data_path=str(data_path),
        post_train_data_path=(
            str(post_train_data_path) if post_train_data_path is not None else None
        ),
        mix_acbc_data_path=(
            str(mix_acbc_data_path) if mix_acbc_data_path is not None else None
        ),
        model_path=str(model_cfg.model_path),
        config_name=str(openpi_cfg.config_name),
        assets_dir=str(openpi_cfg.assets_dir),
        asset_id=str(openpi_cfg.asset_id),
        action_dim=int(openpi_cfg.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(openpi_cfg.max_token_len),
        batch_size=(
            int(cfg.actor.eval_batch_size)
            if eval_dataset
            else int(cfg.actor.micro_batch_size)
        ),
        num_workers=int(data_cfg.num_workers),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        fps=fps,
        tolerance_s=tolerance_s,
        dist_rank=rank,
        dist_world_size=world_size,
        data_kwargs=typing.cast(dict | None, data_kwargs),
        prompt_from_task=prompt_from_task,
        acbc=acbc,
        mix_acbc_primary_ratio=mix_acbc_primary_ratio,
        mix_acbc_secondary_ratio=mix_acbc_secondary_ratio,
        post_train_demo_ratio=post_train_demo_ratio,
        post_train_correction_ratio=post_train_correction_ratio,
    )
    return loader, loader.data_config()
