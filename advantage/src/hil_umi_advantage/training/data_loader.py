# Modified for HIL-UMI.
from collections.abc import Iterator, Sequence
import logging
import math
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import hil_umi_advantage.models.model as _model
from hil_umi_advantage.training.advantage_dataset import AdvantageLerobotDataset
from hil_umi_advantage.training.advantage_dataset import CanonicalAdvantageDataset
from hil_umi_advantage.training.advantage_dataset import detect_advantage_dataset_schema
import hil_umi_advantage.training.config as _config
import hil_umi_advantage.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class MixedAdvantageDataset(torch.utils.data.ConcatDataset):
    """Concatenated datasets with boundaries for two balanced source groups."""

    def __init__(self, group_a: Sequence[Dataset], group_b: Sequence[Dataset]):
        if not group_a or not group_b:
            raise ValueError("Both advantage mix groups must contain at least one dataset")
        self.group_lengths = (sum(map(len, group_a)), sum(map(len, group_b)))
        if any(length <= 0 for length in self.group_lengths):
            raise ValueError(f"Advantage mix groups must be non-empty; group lengths={self.group_lengths}")
        super().__init__([*group_a, *group_b])


class DistributedTwoGroupSampler(torch.utils.data.Sampler[int]):
    """Sample two contiguous dataset groups with total probability 0.5 each."""

    def __init__(
        self,
        group_lengths: tuple[int, int],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        if any(length <= 0 for length in group_lengths):
            raise ValueError(f"Group lengths must be positive, got {group_lengths}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.group_lengths = group_lengths
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(sum(group_lengths) / num_replicas)
        self.total_size = self.num_samples * num_replicas
        group_a_length, group_b_length = group_lengths
        self.weights = torch.cat(
            (
                torch.full((group_a_length,), 0.5 / group_a_length, dtype=torch.float64),
                torch.full((group_b_length,), 0.5 / group_b_length, dtype=torch.float64),
            )
        )

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        self.epoch += 1
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def _create_mixed_advantage_source(
    repo_id: str,
    *,
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    split: str,
) -> Dataset:
    dataset_path = Path(repo_id).expanduser()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Advantage mix dataset is not a local directory: {dataset_path}")
    repo_id = str(dataset_path.resolve())

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    if "stage_progress_gt" not in dataset_meta.features:
        raise ValueError(f"Advantage mix dataset is missing stage_progress_gt: {repo_id}")
    schema = detect_advantage_dataset_schema(dataset_meta.features)
    episodes = episodes_split_through_task(repo_id, split_type=split, shuffle=False)
    if not episodes:
        raise ValueError(f"Advantage mix dataset has no episodes for split={split!r}: {repo_id}")
    short_episodes = [ep_idx for ep_idx in episodes if int(dataset_meta.episodes[ep_idx]["length"]) < 2]
    if short_episodes:
        raise ValueError(f"Advantage mix dataset has episodes shorter than two frames: {repo_id}: {short_episodes}")

    dataset: Dataset = AdvantageLerobotDataset(
        repo_id=repo_id,
        episodes=episodes,
        delta_timestamps={
            schema.action_key: [t / dataset_meta.fps for t in range(action_horizon)],
        },
        video_backend="pyav",
        selected_video_keys=schema.video_keys,
    )
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    logging.info(
        "Loaded advantage mix source %s: schema=%s episodes=%d frames=%d cameras=%s",
        repo_id,
        schema.name,
        len(episodes),
        len(dataset),
        schema.video_keys,
    )
    return CanonicalAdvantageDataset(dataset, schema, state_dim=model_config.action_dim)


def _create_mixed_advantage_dataset(
    group_a: Sequence[str],
    group_b: Sequence[str],
    *,
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    split: str,
) -> MixedAdvantageDataset:
    resolved_groups = [[str(Path(repo_id).expanduser().resolve()) for repo_id in group] for group in (group_a, group_b)]

    all_repo_ids = [repo_id for group in resolved_groups for repo_id in group]
    duplicates = sorted({repo_id for repo_id in all_repo_ids if all_repo_ids.count(repo_id) > 1})
    if duplicates:
        raise ValueError(f"Advantage mix dataset paths must be unique: {duplicates}")

    datasets = [
        [
            _create_mixed_advantage_source(
                repo_id,
                data_config=data_config,
                action_horizon=action_horizon,
                model_config=model_config,
                split=split,
            )
            for repo_id in group
        ]
        for group in resolved_groups
    ]
    mixed_dataset = MixedAdvantageDataset(datasets[0], datasets[1])
    logging.info(
        "Advantage mix enabled: group_a=%d frames, group_b=%d frames, probabilities=(0.5, 0.5)",
        *mixed_dataset.group_lengths,
    )
    return mixed_dataset


def create_advantage_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    config: _config.TrainConfig = None,
) -> Dataset:
    """Create LeRobot dataset for advantage estimator (progress/value) training."""
    split = getattr(config, "split", "all") or "all"

    group_a = tuple(getattr(config, "advantage_dataset_group_a", ()))
    group_b = tuple(getattr(config, "advantage_dataset_group_b", ()))
    if bool(group_a) != bool(group_b):
        raise ValueError("Advantage mix requires both advantage_dataset_group_a and advantage_dataset_group_b")
    if group_a and group_b:
        return _create_mixed_advantage_dataset(
            group_a,
            group_b,
            data_config=data_config,
            action_horizon=action_horizon,
            model_config=model_config,
            split=split,
        )

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = AdvantageLerobotDataset(
        repo_id=repo_id,
        episodes=episodes_split_through_task(repo_id, split_type=split, shuffle=False),
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend="pyav",
    )
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def episodes_split_through_task(
    repo_id: str,
    split_ratio: float = 0.9,
    split_type: str = "train",
    *,
    shuffle: bool = False,
    random_seed: int = 42,
) -> list[int]:
    """Split the episodes by task."""
    assert split_type in ["all", "train", "val"], f"Invalid split type '{split_type}'. Must be 'all', 'train' or 'val'."
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    episodes_meta = dataset_meta.episodes
    episodes_index = list(episodes_meta)
    # tasks = [episodes_meta[idx]["tasks"] for idx in episodes_index]
    if split_type == "all":
        return episodes_index

    episodes_index_dict = {}
    for idx in episodes_index:
        task = "".join(episodes_meta[idx]["tasks"])
        episodes_index_dict.setdefault(task, []).append(idx)
    train_episodes = []
    val_episodes = []
    for task_episodes in episodes_index_dict.values():
        train_length = int(len(task_episodes) * split_ratio)
        train_episodes.extend(task_episodes[:train_length])
        val_episodes.extend(task_episodes[train_length:])
    if split_type == "train":
        return train_episodes
    return val_episodes


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        config=config,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    config: _config.TrainConfig = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        config: Full train config; used when advantage_estimator=True to create advantage dataset.
    """
    if config is not None and getattr(config, "advantage_estimator", False):
        dataset = create_advantage_torch_dataset(data_config, action_horizon, model_config, config)
    else:
        dataset = create_torch_dataset(data_config, action_horizon, model_config)
    mix_group_lengths = dataset.group_lengths if isinstance(dataset, MixedAdvantageDataset) else None
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if mix_group_lengths is not None:
            sampler = DistributedTwoGroupSampler(
                mix_group_lengths,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
            )
            local_batch_size = batch_size // world_size
        elif torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // world_size
        else:
            local_batch_size = batch_size
    else:
        if mix_group_lengths is not None:
            sampler = DistributedTwoGroupSampler(mix_group_lengths, seed=seed)
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            sampler: Optional sampler that controls dataset indices.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        available_samples = len(sampler) if sampler is not None else len(dataset)
        if available_samples < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than available samples ({available_samples})."
            )

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
