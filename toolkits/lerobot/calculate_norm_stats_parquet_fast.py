# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Fast, training-compatible norm stats for single-Franka rot6d parquet data.

This implementation deliberately matches ``calculate_norm_stats.py`` before
optimizing execution:

* only action-chunk starts with all ``action_horizon`` frames in one episode
  are eligible (there is no temporal action padding);
* normal stats include the final incomplete batch;
* post-train stats use the same fixed-size demo/correction batch schedule,
  ratio normalization, correction shuffle, and small demo-tail behavior;
* state/actions undergo the same 10-D slice, 32-D model padding, and body-frame
  SE(3) delta transform; and
* rank 0 updates OpenPI's official ``normalize.RunningStats`` in the original
  global batch order, preserving its streaming histogram quantile semantics.

Parquet I/O bypasses image decoding.  CUDA execution vectorizes the expensive
SE(3) transform in float64 and rounds at the same float32 boundaries as the
NumPy reference.  Under ``torchrun``, ranks transform consecutive batches in
parallel and gather them in deterministic batch order before statistics are
updated.  CPU and CUDA floating-point backends are not guaranteed to be
bitwise identical, but they operate on exactly the same samples with the same
statistics algorithm.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import math
import os
import sys
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

# Import OpenPI only after disabling JAX's eager GPU reservation.  The fast
# path owns CUDA through PyTorch and does not execute JAX kernels.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import openpi.shared.normalize as normalize
import torch
import tqdm
import tyro
from openpi.training.config import DataConfig

# When launched as ``python toolkits/lerobot/...`` or through ``torchrun``,
# Python puts this file's directory (rather than the repository root) on
# sys.path. Add the root before importing the local ``training`` package so the
# documented file-path entrypoint works without an editable installation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.data.lerobot_paths import resolve_lerobot_dataset_root  # noqa: E402
from training.models.embodiment.openpi.dataconfig import (  # noqa: E402
    get_openpi_config,
)
from training.models.embodiment.openpi.transforms import (  # noqa: E402
    RigidBodyDeltaActions,
    SINGLE_ARM_ROT6D_LAYOUT,
)

SUPPORTED_CONFIGS = {"pi05_singlefranka_tcp_rot6d"}
_REAL_ACTION_DIM = 10


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _normalize_post_train_ratios(
    demo_ratio: float,
    correction_ratio: float,
) -> tuple[float, float]:
    """Match ``calculate_norm_stats.py`` post-train ratio normalization."""
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


def _format_post_train_output_task_name(
    task_name: str,
    post_train_data_path: str,
    demo_ratio: float,
    correction_ratio: float,
) -> str:
    """Match the reference post-train norm-stats directory convention."""
    _normalize_post_train_ratios(demo_ratio, correction_ratio)

    def _format_ratio(value: float) -> str:
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text or "0"

    correction_name = Path(post_train_data_path).expanduser().name
    if not correction_name:
        raise ValueError(
            "--post-train-data-path must have a non-empty final path component."
        )
    return (
        f"{task_name}_{_format_ratio(demo_ratio)}_"
        f"{_format_ratio(correction_ratio)}_{correction_name}"
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
    """Copy the reference mixed-stats stopping rule exactly."""
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
            f"len={demo_len}, batch_size={batch_size}, "
            f"demo_fraction={demo_fraction:.6f}."
        )
    return counts


class RatioMixedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Reference-compatible post-train demo/correction batch sampler."""

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


def _validate_lerobot_dataset(data_path: str, name: str) -> Path:
    root = resolve_lerobot_dataset_root(data_path)
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset not found for {name}={data_path!r} at {root}. "
            "Pass a local dataset path, a Hugging Face repo id with data under "
            "HF_LEROBOT_HOME (default: ~/.cache/huggingface/lerobot), or download "
            "the dataset first."
        )
    return root


def _get_output_base_path(
    config_name: str,
    config: Any,
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_fixed_size_list_column(table: Any, name: str) -> np.ndarray:
    column = table[name].combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False)
    list_size = getattr(column.type, "list_size", None)
    if list_size is not None:
        return values.reshape(len(column), list_size)

    offsets = column.offsets.to_numpy(zero_copy_only=False)
    lengths = offsets[1:] - offsets[:-1]
    if len(set(lengths.tolist())) != 1:
        raise ValueError(f"Column {name!r} has variable-length rows.")
    return values.reshape(len(column), int(lengths[0]))


@dataclasses.dataclass(frozen=True)
class _Episode:
    index: int
    path: Path
    length: int
    valid_chunk_starts: int


class FullChunkParquetDataset:
    """State/action-only view matching ``SingleFrankaSftDataset`` indices."""

    def __init__(
        self,
        dataset_root: Path,
        action_horizon: int,
        *,
        episode_cache_size: int,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
        if episode_cache_size <= 0:
            raise ValueError(
                f"episode_cache_size must be positive, got {episode_cache_size}."
            )

        self.dataset_root = dataset_root
        self.action_horizon = int(action_horizon)
        self._episode_cache_size = int(episode_cache_size)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._episodes = self._build_episode_index()
        self._cumulative_valid: list[int] = []
        total = 0
        for episode in self._episodes:
            total += episode.valid_chunk_starts
            self._cumulative_valid.append(total)
        self._length = total
        if self._length == 0:
            raise ValueError(
                f"FullChunkParquetDataset {dataset_root} has no samples with a full "
                f"{self.action_horizon}-frame action chunk."
            )

    def _build_episode_index(self) -> list[_Episode]:
        info = json.loads(
            (self.dataset_root / "meta" / "info.json").read_text(encoding="utf-8")
        )
        data_template = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            raise FileNotFoundError(
                f"Strict parquet stats require {episodes_path} so episode boundaries "
                "match LeRobot's episode_data_index."
            )

        rows = sorted(_read_jsonl(episodes_path), key=lambda row: int(row["episode_index"]))
        expected_indices = list(range(len(rows)))
        actual_indices = [int(row["episode_index"]) for row in rows]
        if actual_indices != expected_indices:
            raise ValueError(
                "LeRobot episode indices must be contiguous from zero; "
                f"got {actual_indices[:10]}{'...' if len(actual_indices) > 10 else ''}."
            )

        episodes = []
        for row in rows:
            episode_index = int(row["episode_index"])
            episode_chunk = episode_index // int(info.get("chunks_size", 1000))
            relative_path = data_template.format(
                episode_chunk=episode_chunk,
                chunk_index=episode_chunk,
                episode_index=episode_index,
            )
            path = self.dataset_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Missing episode parquet file: {path}")
            length = int(row["length"])
            valid = max(0, length - self.action_horizon + 1)
            episodes.append(
                _Episode(
                    index=episode_index,
                    path=path,
                    length=length,
                    valid_chunk_starts=valid,
                )
            )
        return episodes

    def __len__(self) -> int:
        return self._length

    def _read_episode(self, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.pop(episode_index, None)
        if cached is not None:
            self._cache[episode_index] = cached
            return cached

        import pyarrow.parquet as pq

        episode = self._episodes[episode_index]
        parquet_file = pq.ParquetFile(episode.path)
        columns = set(parquet_file.schema_arrow.names)
        # These are the exact source keys accepted by SingleFrankaStatsRepack.
        missing = {"state", "actions"}.difference(columns)
        if missing:
            raise ValueError(
                f"{episode.path} is missing canonical single-Franka columns "
                f"{sorted(missing)}; found {sorted(columns)}. Convert the dataset "
                "with convert_umi_left_to_single_franka.py first."
            )
        table = parquet_file.read(columns=["state", "actions"])
        state = _read_fixed_size_list_column(table, "state").astype(
            np.float32, copy=False
        )
        actions = _read_fixed_size_list_column(table, "actions").astype(
            np.float32, copy=False
        )
        if len(state) != episode.length or len(actions) != episode.length:
            raise ValueError(
                f"{episode.path} has {len(state)} state / {len(actions)} action rows, "
                f"but episodes.jsonl declares {episode.length}."
            )
        if state.shape[-1] < _REAL_ACTION_DIM or actions.shape[-1] < _REAL_ACTION_DIM:
            raise ValueError(
                f"{episode.path} needs at least {_REAL_ACTION_DIM} state/action dims; "
                f"got state={state.shape}, actions={actions.shape}."
            )

        value = (state, actions)
        self._cache[episode_index] = value
        while len(self._cache) > self._episode_cache_size:
            self._cache.popitem(last=False)
        return value

    def _locate(self, sample_index: int) -> tuple[int, int]:
        sample_index = int(sample_index)
        if sample_index < 0:
            sample_index += len(self)
        if sample_index < 0 or sample_index >= len(self):
            raise IndexError(
                f"Full-chunk sample index {sample_index} out of range for {len(self)}."
            )
        episode_index = bisect.bisect_right(self._cumulative_valid, sample_index)
        previous_total = (
            0 if episode_index == 0 else self._cumulative_valid[episode_index - 1]
        )
        return episode_index, sample_index - previous_total

    def load_batch(self, sample_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        batch_size = len(sample_indices)
        states = np.empty((batch_size, _REAL_ACTION_DIM), dtype=np.float32)
        actions = np.empty(
            (batch_size, self.action_horizon, _REAL_ACTION_DIM),
            dtype=np.float32,
        )
        locations: dict[int, list[tuple[int, int]]] = {}
        for output_index, sample_index in enumerate(sample_indices):
            episode_index, start = self._locate(int(sample_index))
            locations.setdefault(episode_index, []).append((output_index, start))

        for episode_index, items in locations.items():
            episode_states, episode_actions = self._read_episode(episode_index)
            for output_index, start in items:
                stop = start + self.action_horizon
                # _locate only returns filtered full-chunk starts. Keep this
                # assertion next to the slice so temporal padding cannot return.
                if stop > len(episode_actions):
                    raise AssertionError(
                        f"Action chunk [{start}:{stop}] crosses episode "
                        f"{episode_index} of length {len(episode_actions)}."
                    )
                states[output_index] = episode_states[start, :_REAL_ACTION_DIM]
                actions[output_index] = episode_actions[
                    start:stop, :_REAL_ACTION_DIM
                ]
        return states, actions


def _pad_last_dim_numpy(array: np.ndarray, dim: int) -> np.ndarray:
    if array.shape[-1] > dim:
        raise ValueError(f"Cannot pad trailing dim {array.shape[-1]} down to {dim}.")
    if array.shape[-1] == dim:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[-1] = (0, dim - array.shape[-1])
    return np.pad(array, pad_width)


def _transform_batch_cpu(
    states: np.ndarray,
    actions: np.ndarray,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the exact per-item NumPy transform used by the reference."""
    padded_states = _pad_last_dim_numpy(states, action_dim).astype(
        np.float32, copy=False
    )
    padded_actions = _pad_last_dim_numpy(actions, action_dim).astype(
        np.float32, copy=False
    )
    transform = RigidBodyDeltaActions(SINGLE_ARM_ROT6D_LAYOUT)
    transformed_actions = np.empty_like(padded_actions)
    for index in range(len(padded_states)):
        item = {
            "state": padded_states[index],
            "actions": padded_actions[index],
        }
        transformed_actions[index] = transform(item)["actions"]
    return padded_states, transformed_actions


def _rot6d_to_matrix_torch(r6: torch.Tensor) -> torch.Tensor:
    """Match training.utils.rot6d.rot6d_to_matrix's dtype boundaries on CUDA."""
    r6_64 = r6.to(dtype=torch.float64)
    r1 = r6_64[..., :3]
    r2 = r6_64[..., 3:]
    n1 = torch.linalg.vector_norm(r1, dim=-1, keepdim=True)
    if bool(torch.any(n1 < 1e-8).item()):
        raise ValueError(
            "rot6d_to_matrix: |r1| underflow; cannot form orthonormal frame."
        )
    b1 = r1 / n1
    dot = torch.sum(b1 * r2, dim=-1, keepdim=True)
    r2_perp = r2 - dot * b1
    n2 = torch.linalg.vector_norm(r2_perp, dim=-1, keepdim=True)
    if bool(torch.any(n2 < 1e-8).item()):
        raise ValueError(
            "rot6d_to_matrix: |r2_perp| underflow; cannot form orthonormal frame."
        )
    b2 = r2_perp / n2
    b3 = torch.cross(b1, b2, dim=-1)
    # The NumPy helper explicitly casts its matrix result to float32.
    return torch.stack([b1, b2, b3], dim=-1).to(dtype=torch.float32)


def _pose_to_se3_torch(xyz: torch.Tensor, r6: torch.Tensor) -> torch.Tensor:
    rotation = _rot6d_to_matrix_torch(r6).to(dtype=torch.float64)
    xyz_64 = xyz.to(dtype=torch.float64)
    transform = torch.zeros(
        (*rotation.shape[:-2], 4, 4),
        device=rotation.device,
        dtype=torch.float64,
    )
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = xyz_64
    transform[..., 3, 3] = 1.0
    return transform


def _transform_batch_cuda(
    states: np.ndarray,
    actions: np.ndarray,
    action_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized high-precision equivalent of the reference action transform."""
    state_tensor = torch.as_tensor(states, device=device, dtype=torch.float32)
    action_tensor = torch.as_tensor(actions, device=device, dtype=torch.float32)
    padded_states = torch.zeros(
        (len(states), action_dim), device=device, dtype=torch.float32
    )
    padded_actions = torch.zeros(
        (len(states), actions.shape[1], action_dim),
        device=device,
        dtype=torch.float32,
    )
    padded_states[..., :_REAL_ACTION_DIM] = state_tensor[..., :_REAL_ACTION_DIM]
    padded_actions[..., :_REAL_ACTION_DIM] = action_tensor[..., :_REAL_ACTION_DIM]

    state_transform = _pose_to_se3_torch(
        padded_states[..., :3], padded_states[..., 3:9]
    )
    action_transform = _pose_to_se3_torch(
        padded_actions[..., :3], padded_actions[..., 3:9]
    )
    # The reference uses np.linalg.inv(T_state), not R.T, after the rot6d
    # matrix has been rounded to float32. Preserve that distinction here.
    state_inverse = torch.linalg.inv(state_transform)
    delta_transform = torch.matmul(state_inverse[:, None], action_transform)
    padded_actions[..., :3] = delta_transform[..., :3, 3].to(torch.float32)
    padded_actions[..., 3:9] = torch.cat(
        [delta_transform[..., :3, 0], delta_transform[..., :3, 1]], dim=-1
    ).to(torch.float32)
    return padded_states, padded_actions


@dataclasses.dataclass(frozen=True)
class _Runtime:
    device: torch.device
    rank: int
    world_size: int
    local_rank: int
    initialized_process_group: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _create_runtime(
    stats_device: str,
    deterministic: bool,
    local_rank_argument: int | None = None,
) -> _Runtime:
    stats_device = stats_device.lower()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0" if local_rank_argument is None else str(local_rank_argument),
        )
    )

    if deterministic:
        # This must be set before creating a CUDA context.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if stats_device == "auto":
        stats_device = "cuda" if torch.cuda.is_available() else "cpu"
    if stats_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"stats_device={stats_device!r} was requested, but CUDA is not available."
        )
    if stats_device.startswith("cuda"):
        if world_size > 1 or stats_device == "cuda":
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(stats_device)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif stats_device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(
            "stats_device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'; "
            f"got {stats_device!r}."
        )

    if deterministic:
        torch.use_deterministic_algorithms(True)

    initialized = False
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        initialized = True
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    return _Runtime(
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        initialized_process_group=initialized,
    )


def _normal_batch_plans(
    dataset_len: int,
    batch_size: int,
    max_frames: int | None,
) -> list[list[int]]:
    if max_frames is not None and max_frames < 0:
        raise ValueError(f"max_frames must be non-negative, got {max_frames}.")
    if max_frames is not None and max_frames < dataset_len:
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(dataset_len, generator=generator)[:max_frames].tolist()
    else:
        indices = list(range(dataset_len))
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _load_mixed_batch(
    demo_dataset: FullChunkParquetDataset,
    correction_dataset: FullChunkParquetDataset | None,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if correction_dataset is None:
        return demo_dataset.load_batch(indices)

    correction_offset = len(demo_dataset)
    demo_positions = [i for i, index in enumerate(indices) if index < correction_offset]
    correction_positions = [
        i for i, index in enumerate(indices) if index >= correction_offset
    ]
    states = np.empty((len(indices), _REAL_ACTION_DIM), dtype=np.float32)
    actions = np.empty(
        (len(indices), demo_dataset.action_horizon, _REAL_ACTION_DIM),
        dtype=np.float32,
    )
    if demo_positions:
        demo_states, demo_actions = demo_dataset.load_batch(
            [indices[position] for position in demo_positions]
        )
        states[demo_positions] = demo_states
        actions[demo_positions] = demo_actions
    if correction_positions:
        correction_states, correction_actions = correction_dataset.load_batch(
            [indices[position] - correction_offset for position in correction_positions]
        )
        states[correction_positions] = correction_states
        actions[correction_positions] = correction_actions
    return states, actions


def _all_gather_padded_batch(
    runtime: _Runtime,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if runtime.world_size == 1:
        return [states], [actions]
    gathered_states = [torch.empty_like(states) for _ in range(runtime.world_size)]
    gathered_actions = [torch.empty_like(actions) for _ in range(runtime.world_size)]
    torch.distributed.all_gather(gathered_states, states)
    torch.distributed.all_gather(gathered_actions, actions)
    return gathered_states, gathered_actions


def _compute_stats(
    *,
    demo_dataset: FullChunkParquetDataset,
    correction_dataset: FullChunkParquetDataset | None,
    batch_plans: Sequence[Sequence[int]],
    batch_size: int,
    action_dim: int,
    runtime: _Runtime,
) -> dict[str, normalize.NormStats] | None:
    stats = (
        {key: normalize.RunningStats() for key in ("state", "actions")}
        if runtime.is_main
        else None
    )
    num_rounds = math.ceil(len(batch_plans) / runtime.world_size)
    progress = (
        tqdm.tqdm(total=len(batch_plans), desc="Computing parquet stats")
        if runtime.is_main
        else None
    )

    for round_index in range(num_rounds):
        batch_index = round_index * runtime.world_size + runtime.rank
        count = 0
        if batch_index < len(batch_plans):
            indices = batch_plans[batch_index]
            count = len(indices)
            raw_states, raw_actions = _load_mixed_batch(
                demo_dataset,
                correction_dataset,
                indices,
            )
            if runtime.device.type == "cuda":
                transformed_states, transformed_actions = _transform_batch_cuda(
                    raw_states,
                    raw_actions,
                    action_dim,
                    runtime.device,
                )
            else:
                numpy_states, numpy_actions = _transform_batch_cpu(
                    raw_states,
                    raw_actions,
                    action_dim,
                )
                transformed_states = torch.from_numpy(numpy_states).to(runtime.device)
                transformed_actions = torch.from_numpy(numpy_actions).to(runtime.device)
        else:
            transformed_states = torch.empty(
                (0, action_dim), device=runtime.device, dtype=torch.float32
            )
            transformed_actions = torch.empty(
                (0, demo_dataset.action_horizon, action_dim),
                device=runtime.device,
                dtype=torch.float32,
            )

        padded_states = torch.zeros(
            (batch_size, action_dim), device=runtime.device, dtype=torch.float32
        )
        padded_actions = torch.zeros(
            (batch_size, demo_dataset.action_horizon, action_dim),
            device=runtime.device,
            dtype=torch.float32,
        )
        if count:
            padded_states[:count] = transformed_states
            padded_actions[:count] = transformed_actions

        gathered_states, gathered_actions = _all_gather_padded_batch(
            runtime,
            padded_states,
            padded_actions,
        )
        if runtime.is_main:
            assert stats is not None
            completed_this_round = 0
            # rank order is global batch order by construction.
            for source_rank in range(runtime.world_size):
                source_batch_index = round_index * runtime.world_size + source_rank
                if source_batch_index >= len(batch_plans):
                    continue
                source_count = len(batch_plans[source_batch_index])
                state_batch = (
                    gathered_states[source_rank][:source_count].cpu().numpy()
                )
                action_batch = (
                    gathered_actions[source_rank][:source_count].cpu().numpy()
                )
                stats["state"].update(state_batch)
                stats["actions"].update(action_batch)
                completed_this_round += 1
            assert progress is not None
            progress.update(completed_this_round)

    if progress is not None:
        progress.close()
    if not runtime.is_main:
        return None
    assert stats is not None
    return {key: value.get_statistics() for key, value in stats.items()}


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
    max_frames: int | None = None,
    stats_device: str = "auto",
    deterministic: bool = True,
    episode_cache_size: int = 128,
    local_rank: int | None = None,
) -> None:
    """Compute reference-compatible norm stats with optional multi-GPU speedup.

    Launch one process normally for CPU/single-GPU execution.  For distributed
    GPU execution use, for example::

        torchrun --standalone --nproc-per-node=8 \
          toolkits/lerobot/calculate_norm_stats_parquet_fast.py \
          --config-name pi05_singlefranka_tcp_rot6d \
          --repo-id dataset/my_task_franka --stats-device cuda
    """
    if config_name not in SUPPORTED_CONFIGS:
        raise ValueError(
            f"{Path(__file__).name} only supports {sorted(SUPPORTED_CONFIGS)}; "
            f"got {config_name!r}."
        )
    post_train_enabled = _as_bool(post_train)
    if post_train_enabled and not post_train_data_path:
        raise ValueError(
            "--post-train true requires --post-train-data-path to point to the "
            "collection correction dataset."
        )
    if post_train_enabled and max_frames is not None:
        raise ValueError(
            "--max-frames is not defined by the reference post-train path; omit it "
            "to preserve exact post-train sampling semantics."
        )

    demo_root = _validate_lerobot_dataset(repo_id, "repo_id")
    correction_root = None
    if post_train_data_path is not None:
        correction_root = _validate_lerobot_dataset(
            post_train_data_path, "post_train_data_path"
        )

    # torchrun versions may provide local rank as an environment variable, a
    # --local-rank CLI argument, or both. Accept both forms.
    runtime = _create_runtime(stats_device, deterministic, local_rank)
    try:
        config = get_openpi_config(
            config_name,
            model_path=model_path,
            repo_id=repo_id,
        )
        data_config = config.data.create(config.assets_dirs, config.model)
        if data_config.rlds_data_dir is not None:
            raise ValueError("The parquet fast path does not support RLDS data.")

        action_dim = int(config.model.action_dim)
        action_horizon = int(config.model.action_horizon)
        batch_size = int(config.batch_size)
        demo_dataset = FullChunkParquetDataset(
            demo_root,
            action_horizon,
            episode_cache_size=episode_cache_size,
        )
        correction_dataset = None
        sampler = None
        if post_train_enabled:
            assert correction_root is not None
            correction_config = get_openpi_config(
                config_name,
                model_path=model_path,
                repo_id=post_train_data_path,
            )
            correction_data_config = correction_config.data.create(
                correction_config.assets_dirs,
                correction_config.model,
            )
            if correction_data_config.rlds_data_dir is not None:
                raise ValueError(
                    "post_train correction norm stats currently supports parquet "
                    "torch data only."
                )
            correction_dataset = FullChunkParquetDataset(
                correction_root,
                action_horizon,
                episode_cache_size=episode_cache_size,
            )
            sampler = RatioMixedBatchSampler(
                demo_len=len(demo_dataset),
                correction_len=len(correction_dataset),
                correction_offset=len(demo_dataset),
                batch_size=batch_size,
                demo_ratio=post_train_demo_ratio,
                correction_ratio=post_train_correction_ratio,
                seed=post_train_seed,
            )
            batch_plans = list(iter(sampler))
        else:
            batch_plans = _normal_batch_plans(
                len(demo_dataset),
                batch_size,
                max_frames,
            )

        if not batch_plans:
            raise ValueError("No norm-stats batches were selected.")
        if runtime.is_main:
            print(
                "Using training-compatible parquet stats: "
                f"device={runtime.device}, world_size={runtime.world_size}, "
                f"action_horizon={action_horizon}, action_dim={action_dim}, "
                f"batches={len(batch_plans)}, demo_full_chunks={len(demo_dataset)}"
            )
            if sampler is not None:
                first_demo, first_correction = sampler.first_batch_split
                print(
                    "Computing post-train stats with requested ratio: "
                    f"{post_train_demo_ratio:g} demo : "
                    f"{post_train_correction_ratio:g} correction; "
                    f"first batch {first_demo} demo + {first_correction} correction; "
                    f"epoch samples {sampler.demo_samples} demo + "
                    f"{sampler.correction_samples} correction; "
                    f"deterministic correction shuffle seed {post_train_seed + 1}; "
                    f"correction_full_chunks={len(correction_dataset)}"
                )

        norm_stats = _compute_stats(
            demo_dataset=demo_dataset,
            correction_dataset=correction_dataset,
            batch_plans=batch_plans,
            batch_size=batch_size,
            action_dim=action_dim,
            runtime=runtime,
        )

        if runtime.is_main:
            assert norm_stats is not None
            output_path = _get_output_base_path(config_name, config, data_config)
            if output_task_name is None:
                if post_train_enabled and task_name is not None:
                    assert post_train_data_path is not None
                    output_task_name = _format_post_train_output_task_name(
                        task_name,
                        post_train_data_path,
                        post_train_demo_ratio,
                        post_train_correction_ratio,
                    )
                else:
                    output_task_name = task_name
            if output_task_name is not None:
                output_path = output_path / output_task_name
            print(f"Writing training-compatible parquet stats to: {output_path}")
            normalize.save(output_path, norm_stats)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    finally:
        if runtime.initialized_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    tyro.cli(main)
