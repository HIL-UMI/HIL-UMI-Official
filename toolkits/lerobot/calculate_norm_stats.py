# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import math
import multiprocessing
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import torch
import tqdm
import tyro
from openpi.training.config import DataConfig

from training.data.datasets.openpi_pytorch.single_franka.single_franka_sft_dataset import (
    SingleFrankaSftDataset,
)
from training.data.lerobot_paths import resolve_lerobot_dataset_root
from training.models.embodiment.openpi.dataconfig import get_openpi_config


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def _collate_stats_items(items: Sequence[dict]) -> dict[str, np.ndarray]:
    """Collate only arrays consumed by the normalization-statistics loop."""
    return {
        key: np.stack([np.asarray(item[key]) for item in items], axis=0)
        for key in ("state", "actions")
    }


def _stats_worker_init_fn(worker_id: int) -> None:
    """Disable JAX GPU preallocation inside data-loader workers."""
    del worker_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


_SINGLE_FRANKA_REPACK_KEYS = {
    "observation/image": ("image",),
    "observation/extra_view_image": ("extra_view_image", "extra_view_image-0"),
    "observation/state": ("state",),
    "actions": ("actions",),
}


class SingleFrankaStatsRepack(transforms.DataTransformFn):
    """Match the single-Franka SFT loader repack before computing stats."""

    def __call__(self, frame: dict) -> dict:
        data = {}
        for destination, sources in _SINGLE_FRANKA_REPACK_KEYS.items():
            source = next((key for key in sources if key in frame), None)
            if source is None:
                raise KeyError(
                    f"single_franka stats frame missing one of {sources!r} "
                    f"(needed for {destination!r}); "
                    f"available keys={sorted(frame.keys())}"
                )
            data[destination] = np.asarray(frame[source])

        prompt = frame.get("prompt", frame.get("task"))
        if prompt is None:
            raise ValueError(
                "single_franka stats frame is missing both 'prompt' and 'task'; "
                "ensure the LeRobot dataset contains task annotations."
            )
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        data["prompt"] = prompt
        return data


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
        return self._correction_offset

    def __getitem__(self, idx: int):
        if idx < self._correction_offset:
            return self._primary[idx]
        return self._correction[idx - self._correction_offset]

    def __len__(self) -> int:
        return self._correction_offset + len(self._correction)


def _normalize_post_train_ratios(
    demo_ratio: float,
    correction_ratio: float,
) -> tuple[float, float]:
    demo_ratio = float(demo_ratio)
    correction_ratio = float(correction_ratio)
    if not math.isfinite(demo_ratio) or demo_ratio <= 0:
        raise ValueError(
            "--post-train-demo-ratio must be a positive finite number, "
            f"got {demo_ratio!r}."
        )
    if not math.isfinite(correction_ratio) or correction_ratio <= 0:
        raise ValueError(
            "--post-train-correction-ratio must be a positive finite number, "
            f"got {correction_ratio!r}."
        )
    total = demo_ratio + correction_ratio
    return demo_ratio / total, correction_ratio / total


def _format_post_train_ratio_suffix(
    demo_ratio: float,
    correction_ratio: float,
) -> str:
    demo_fraction, correction_fraction = _normalize_post_train_ratios(
        demo_ratio, correction_ratio
    )

    def _format_percent(value: float) -> str:
        text = f"{value * 100:.6f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")

    return f"{_format_percent(demo_fraction)}_{_format_percent(correction_fraction)}"


def _format_post_train_output_task_name(
    task_name: str,
    post_train_data_path: str,
    demo_ratio: float,
    correction_ratio: float,
) -> str:
    _normalize_post_train_ratios(demo_ratio, correction_ratio)

    def _format_ratio(value: float) -> str:
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text or "0"

    collection_name = Path(post_train_data_path).expanduser().name
    if not collection_name:
        raise ValueError(
            "--post-train-data-path must have a non-empty final path component."
        )
    return (
        f"{task_name}_{_format_ratio(demo_ratio)}_"
        f"{_format_ratio(correction_ratio)}_{collection_name}"
    )


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _ratio_demo_count_for_batch(
    batch_index: int,
    batch_size: int,
    demo_fraction: float,
) -> int:
    previous_total = _round_half_up(batch_index * batch_size * demo_fraction)
    next_total = _round_half_up((batch_index + 1) * batch_size * demo_fraction)
    return max(0, min(batch_size, next_total - previous_total))


def _build_demo_batch_counts(
    *,
    demo_len: int,
    batch_size: int,
    demo_fraction: float,
) -> list[int]:
    if demo_len <= 0:
        raise ValueError("Post-train demo dataset is empty.")
    if batch_size <= 0:
        raise ValueError(f"post_train batch_size must be positive, got {batch_size}.")

    counts: list[int] = []
    consumed = 0
    batch_index = 0
    expected_batches = max(
        1,
        int(math.ceil(demo_len / max(batch_size * demo_fraction, 1e-12))) + 2,
    )
    max_batches = expected_batches + batch_size + 8
    while consumed < demo_len and batch_index < max_batches:
        demo_count = _ratio_demo_count_for_batch(batch_index, batch_size, demo_fraction)
        if demo_count > demo_len - consumed:
            break
        counts.append(demo_count)
        consumed += demo_count
        batch_index += 1

    if not counts:
        raise ValueError(
            "Post-train demo dataset is too small for the requested stats ratio: "
            f"len={demo_len}, batch_size={batch_size}, demo_fraction={demo_fraction:.6f}."
        )
    return counts


class RatioMixedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield mixed demo/correction batches with the requested cumulative ratio."""

    def __init__(
        self,
        *,
        demo_len: int,
        correction_len: int,
        correction_offset: int,
        batch_size: int,
        demo_ratio: float,
        correction_ratio: float,
        seed: int = 0,
    ) -> None:
        if correction_len <= 0:
            raise ValueError("Post-train correction dataset is empty.")
        demo_fraction, correction_fraction = _normalize_post_train_ratios(
            demo_ratio, correction_ratio
        )
        self._batch_size = batch_size
        self._correction_len = correction_len
        self._correction_offset = correction_offset
        self._demo_batch_counts = _build_demo_batch_counts(
            demo_len=demo_len,
            batch_size=batch_size,
            demo_fraction=demo_fraction,
        )
        self._demo_fraction = demo_fraction
        self._correction_fraction = correction_fraction
        self._demo_samples = sum(self._demo_batch_counts)
        self._correction_samples = (
            len(self._demo_batch_counts) * batch_size - self._demo_samples
        )
        generator = torch.Generator()
        # Match the training loader convention: demo uses seed, correction uses
        # seed + 1.
        generator.manual_seed(int(seed) + 1)
        self._correction_indices = torch.randperm(
            correction_len,
            generator=generator,
        ).tolist()

    @property
    def demo_samples(self) -> int:
        return self._demo_samples

    @property
    def correction_samples(self) -> int:
        return self._correction_samples

    @property
    def first_batch_split(self) -> tuple[int, int]:
        demo_count = self._demo_batch_counts[0]
        return demo_count, self._batch_size - demo_count

    def __iter__(self) -> Iterator[list[int]]:
        demo_start = 0
        correction_index = 0
        for demo_count in self._demo_batch_counts:
            demo_batch = list(range(demo_start, demo_start + demo_count))
            demo_start += demo_count

            correction_batch = []
            for _ in range(self._batch_size - demo_count):
                correction_batch.append(
                    self._correction_indices[correction_index]
                    + self._correction_offset
                )
                correction_index = (correction_index + 1) % self._correction_len

            yield demo_batch + correction_batch

    def __len__(self) -> int:
        return len(self._demo_batch_counts)


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _validate_lerobot_dataset(data_path: str, name: str) -> None:
    dataset_root = resolve_lerobot_dataset_root(data_path)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset not found for {name}={data_path!r} at {dataset_root}. "
            "Pass a local dataset path, a Hugging Face repo id with data under "
            "HF_LEROBOT_HOME (default: ~/.cache/huggingface/lerobot), or download "
            "the dataset first."
        )


def _get_output_base_path(
    config_name: str,
    config,
    data_config: DataConfig,
) -> Path:
    if config_name == "pi05_singlefranka_tcp_rot6d":
        repo_root = Path(__file__).resolve().parents[2]
        assets_dir = Path(
            os.environ.get(
                "UMI_LEFT_FRANKA_ASSETS_DIR",
                repo_root / "assets" / "pi05_singlefranka_tcp_rot6d",
            )
        ).expanduser()
        return assets_dir / data_config.asset_id
    return config.assets_dirs / data_config.asset_id


def create_transformed_torch_dataset(
    data_config: DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    repack_inputs: Sequence[transforms.DataTransformFn] | None = None,
    full_action_chunks_only: bool = False,
) -> torch.utils.data.Dataset:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if full_action_chunks_only:
        dataset = SingleFrankaSftDataset(
            data_path=str(data_config.repo_id),
            action_horizon=action_horizon,
        )
    else:
        dataset = _data_loader.create_torch_dataset(
            data_config, action_horizon, model_config
        )
    if repack_inputs is None:
        repack_inputs = data_config.repack_transforms.inputs
    return _data_loader.TransformedDataset(
        dataset,
        [
            *repack_inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    repack_inputs: Sequence[transforms.DataTransformFn] | None = None,
    full_action_chunks_only: bool = False,
) -> tuple[torch.utils.data.DataLoader, int]:
    dataset = create_transformed_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        repack_inputs,
        full_action_chunks_only=full_action_chunks_only,
    )
    if max_frames is not None and max_frames < len(dataset):
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(dataset), generator=generator)[
            :max_frames
        ].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    num_batches = math.ceil(len(dataset) / batch_size)
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    # OpenPI's training loader always drops the final incomplete batch. Stats
    # should cover every selected sample, so use a dedicated non-dropping loader.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_stats_items,
        worker_init_fn=_stats_worker_init_fn,
        drop_last=False,
    )
    return data_loader, num_batches


def create_post_train_torch_dataloaders(
    data_config: DataConfig,
    correction_data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    demo_ratio: float,
    correction_ratio: float,
    seed: int = 0,
    repack_inputs: Sequence[transforms.DataTransformFn] | None = None,
    full_action_chunks_only: bool = False,
) -> tuple[torch.utils.data.DataLoader, int, RatioMixedBatchSampler]:
    demo_dataset = create_transformed_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        repack_inputs,
        full_action_chunks_only=full_action_chunks_only,
    )
    correction_dataset = create_transformed_torch_dataset(
        correction_data_config,
        action_horizon,
        model_config,
        repack_inputs,
        full_action_chunks_only=full_action_chunks_only,
    )
    mixed_dataset = _TwoDataset(demo_dataset, correction_dataset)
    sampler = RatioMixedBatchSampler(
        demo_len=len(demo_dataset),
        correction_len=len(correction_dataset),
        correction_offset=mixed_dataset.correction_offset,
        batch_size=batch_size,
        demo_ratio=demo_ratio,
        correction_ratio=correction_ratio,
        seed=seed,
    )
    data_loader = torch.utils.data.DataLoader(
        mixed_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
    )
    return data_loader, len(sampler), sampler


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    repo_id: str,
    model_path: str | None = None,
    task_name: str | None = None,
    post_train: str = "false",
    post_train_data_path: str | None = None,
    post_train_demo_ratio: float = 0.5,
    post_train_correction_ratio: float = 0.5,
    post_train_seed: int = 0,
    output_task_name: str | None = None,
):
    _validate_lerobot_dataset(repo_id, "repo_id")
    post_train_enabled = _as_bool(post_train)
    if post_train_enabled and not post_train_data_path:
        raise ValueError(
            "--post-train true requires --post-train-data-path to point to the "
            "collection correction dataset."
        )
    if post_train_data_path is not None:
        _validate_lerobot_dataset(post_train_data_path, "post_train_data_path")

    config = get_openpi_config(
        config_name,
        model_path=model_path,
        repo_id=repo_id,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    repack_inputs = (
        [SingleFrankaStatsRepack()]
        if "singlefranka_tcp_rot6d" in config_name
        else None
    )
    full_action_chunks_only = config_name == "pi05_singlefranka_tcp_rot6d"
    if full_action_chunks_only:
        print(
            "Using full action chunks only: episode-tail starts that require "
            f"padding are excluded (action_horizon={config.model.action_horizon})."
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    if post_train_enabled:
        correction_config = get_openpi_config(
            config_name,
            model_path=model_path,
            repo_id=post_train_data_path,
        )
        correction_data_config = correction_config.data.create(
            correction_config.assets_dirs,
            correction_config.model,
        )
        if data_config.rlds_data_dir is not None:
            raise ValueError(
                "post_train norm stats currently supports torch data only."
            )
        if correction_data_config.rlds_data_dir is not None:
            raise ValueError(
                "post_train correction norm stats currently supports torch data only."
            )

        (
            data_loader,
            num_batches,
            sampler,
        ) = create_post_train_torch_dataloaders(
            data_config,
            correction_data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            demo_ratio=post_train_demo_ratio,
            correction_ratio=post_train_correction_ratio,
            seed=post_train_seed,
            repack_inputs=repack_inputs,
            full_action_chunks_only=full_action_chunks_only,
        )
        first_demo_batch_size, first_correction_batch_size = sampler.first_batch_split
        print(
            "Computing post-train stats with requested ratio: "
            f"{post_train_demo_ratio:g} demo : "
            f"{post_train_correction_ratio:g} correction; "
            f"first batch {first_demo_batch_size} demo + "
            f"{first_correction_batch_size} correction; "
            f"epoch samples {sampler.demo_samples} demo + "
            f"{sampler.correction_samples} correction; "
            f"deterministic correction shuffle seed {post_train_seed + 1}"
        )
        for batch in tqdm.tqdm(
            data_loader,
            total=num_batches,
            desc="Computing post-train stats",
        ):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            repack_inputs=repack_inputs,
            full_action_chunks_only=full_action_chunks_only,
        )

    if not post_train_enabled:
        for batch in tqdm.tqdm(
            data_loader,
            total=num_batches,
            desc="Computing stats",
        ):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = _get_output_base_path(config_name, config, data_config)
    if output_task_name is None:
        if post_train_enabled and task_name is not None:
            output_task_name = _format_post_train_output_task_name(
                task_name,
                post_train_data_path,
                post_train_demo_ratio,
                post_train_correction_ratio,
            )
        else:
            output_task_name = task_name
    if output_task_name is not None:
        # Nest stats under a per-task sub-directory so that stats computed
        # from different datasets (tasks) do not overwrite each other.
        # Training must then point its OpenPI assets path at this directory.
        output_path = output_path / output_task_name
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
