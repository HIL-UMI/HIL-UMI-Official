#!/usr/bin/env python3
# Modified for HIL-UMI.
"""Label fold-towel base and stage1 with the base+stage1 mix advantage checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Any

# The training transform uses JAX, but PyTorch owns the inference GPUs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hil_umi_advantage.shared import paths as component_paths  # noqa: E402
from hil_umi_advantage.training.advantage_schema import AdvantageDatasetSchema  # noqa: E402
from hil_umi_advantage.training.advantage_schema import EpisodeData  # noqa: E402
from hil_umi_advantage.training.advantage_schema import EpisodePaths  # noqa: E402
from hil_umi_advantage.training.advantage_schema import detect_advantage_dataset_schema  # noqa: E402
from hil_umi_advantage.training.config_names import canonicalize_config_name  # noqa: E402
from scripts import label_towel_fold_acbc as common  # noqa: E402

LOGGER = logging.getLogger("towel_mix_acbc_labeling")

DEFAULT_BASE_SOURCE = component_paths.data_root() / "fold_towel/base/origin"
DEFAULT_STAGE1_SOURCE = component_paths.data_root() / "fold_towel/stage1/origin"
DEFAULT_OUTPUT_ROOT = component_paths.data_root() / "fold_towel/stage1/labeled"
DEFAULT_CHECKPOINT = component_paths.checkpoint_root() / "advantage/ADVANTAGE_TORCH_TOWEL_FOLD/towel_base_stage1_mix"
DEFAULT_CONFIG_NAME = "ADVANTAGE_TORCH_TOWEL_FOLD"
PROTECTED_OLD_LABELS = component_paths.data_root() / "fold_towel/base/labeled"
TASK_PROMPT = "fold the towel on the table and put it into the basket"
PREDICTION_COLUMNS = ("relative_advantage", "absolute_value", "absolute_advantage")
OUTPUT_MARKER = Path("meta/mix_acbc_source.json")
LABELING_MANIFEST = Path("meta/mix_acbc_labeling.json")


@dataclasses.dataclass(frozen=True)
class DatasetJob:
    name: str
    source: Path
    output: Path
    grouping: str
    group_count: int
    schema: AdvantageDatasetSchema
    info: dict[str, Any]
    annotations: dict[int, list[int]] | None = None
    annotation_path: Path | None = None


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    os.replace(temporary_path, path)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_path_separation(base_source: Path, stage1_source: Path, output_root: Path) -> None:
    protected = _resolved(PROTECTED_OLD_LABELS)
    paths = {
        "base source": _resolved(base_source),
        "stage1 source": _resolved(stage1_source),
        "output root": _resolved(output_root),
    }
    for label, path in paths.items():
        if path == protected or _is_within(path, protected):
            raise ValueError(f"{label} must not use the protected old-label dataset: {protected}")
    if paths["base source"] == paths["stage1 source"]:
        raise ValueError("Base and stage1 sources must be different datasets")
    for label in ("base source", "stage1 source"):
        source = paths[label]
        output = paths["output root"]
        if source == output or _is_within(output, source) or _is_within(source, output):
            raise ValueError(f"{label} and output root must not overlap: source={source}, output={output}")


def read_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    import torch

    metadata_path = checkpoint_dir / "metadata.pt"
    model_path = checkpoint_dir / "model.safetensors"
    if not metadata_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Incomplete advantage checkpoint: {checkpoint_dir}")
    return torch.load(metadata_path, map_location="cpu", weights_only=False)


def validate_mix_checkpoint_metadata(
    metadata: dict[str, Any],
    checkpoint_dir: Path,
    base_source: Path,
    stage1_source: Path,
    config_name: str,
) -> dict[str, Any]:
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint metadata does not contain a serialized training config")

    global_step = int(metadata.get("global_step", -1))
    if checkpoint_dir.name.isdigit() and global_step != int(checkpoint_dir.name):
        raise ValueError(
            f"Checkpoint step mismatch: directory={checkpoint_dir.name}, metadata global_step={global_step}"
        )
    original_config_name = str(config.get("name", ""))
    if canonicalize_config_name(original_config_name) != canonicalize_config_name(config_name):
        raise ValueError(f"Checkpoint config mismatch: expected {config_name}, got {config.get('name')}")

    actual_group_a = tuple(_resolved(Path(path)) for path in config.get("advantage_dataset_group_a", ()))
    actual_group_b = tuple(_resolved(Path(path)) for path in config.get("advantage_dataset_group_b", ()))
    expected_group_a = (_resolved(base_source),)
    expected_group_b = (_resolved(stage1_source),)
    if actual_group_a != expected_group_a or actual_group_b != expected_group_b:
        raise ValueError(
            "Checkpoint was not trained with the requested mix sources: "
            f"expected group_a={expected_group_a}, group_b={expected_group_b}; "
            f"got group_a={actual_group_a}, group_b={actual_group_b}"
        )
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": global_step,
        "config_name": canonicalize_config_name(config_name),
        "original_config_name": original_config_name,
        "experiment_name": config.get("exp_name"),
        "training_group_a": [str(path) for path in actual_group_a],
        "training_group_b": [str(path) for path in actual_group_b],
    }


def episode_paths(
    dataset_dir: Path,
    info: dict[str, Any],
    schema: AdvantageDatasetSchema,
    episode_index: int,
):
    chunk_index = episode_index // int(info["chunks_size"])
    format_values = {"episode_chunk": chunk_index, "episode_index": episode_index}
    parquet = dataset_dir / info["data_path"].format(**format_values)
    front_video = dataset_dir / info["video_path"].format(**format_values, video_key=schema.front_key)
    wrist_video = dataset_dir / info["video_path"].format(**format_values, video_key=schema.wrist_key)
    return EpisodePaths(
        episode_index,
        parquet,
        front_video,
        wrist_video,
        schema.front_key,
        schema.wrist_key,
    )


def load_episode_data(parquet_path: Path, schema: AdvantageDatasetSchema):
    table = pq.read_table(
        parquet_path,
        columns=["timestamp", "frame_index", schema.state_key, schema.action_key, "stage_progress_gt"],
    )
    return EpisodeData(
        timestamps=np.asarray(table["timestamp"].to_pylist(), dtype=np.float64),
        frame_indices=np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        states=np.asarray(table[schema.state_key].to_pylist(), dtype=np.float32),
        actions=np.asarray(table[schema.action_key].to_pylist(), dtype=np.float32),
        stage_progress_gt=np.asarray(table["stage_progress_gt"].to_pylist(), dtype=np.float32),
    )


def validate_dataset(source: Path) -> tuple[dict[str, Any], AdvantageDatasetSchema]:
    source = _resolved(source)
    info = common.read_dataset_info(source)
    schema = detect_advantage_dataset_schema(info.get("features", {}))
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes <= 0 or total_frames <= 0:
        raise ValueError(f"Dataset must contain episodes and frames: {source}")
    if "stage_progress_gt" not in info.get("features", {}):
        raise ValueError(f"Dataset is missing stage_progress_gt metadata: {source}")

    counted_frames = 0
    for episode_index in range(total_episodes):
        paths = episode_paths(source, info, schema, episode_index)
        missing = [path for path in (paths.parquet, paths.front_video, paths.wrist_video) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Episode {episode_index} is incomplete in {source}: {missing}")
        counted_frames += pq.read_metadata(paths.parquet).num_rows
    if counted_frames != total_frames:
        raise ValueError(f"Frame count mismatch for {source}: metadata={total_frames}, parquet={counted_frames}")
    return info, schema


def load_subtask_annotations(
    annotation_path: Path,
    dataset_dir: Path,
    info: dict[str, Any],
) -> tuple[dict[int, list[int]], int]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Subtask annotation file not found: {annotation_path}")
    payload = json.loads(annotation_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Subtask annotation file must contain a JSON object")

    total_episodes = int(info["total_episodes"])
    expected_indices = set(range(total_episodes))
    actual_indices = {int(key) for key in payload}
    missing = sorted(expected_indices - actual_indices)
    extra = sorted(actual_indices - expected_indices)
    if missing or extra:
        raise ValueError(f"Subtask annotations must cover every episode; missing={missing}, extra={extra}")

    annotations = {}
    boundary_count = None
    for episode_index in range(total_episodes):
        parquet_path = common.parquet_path_for_episode(dataset_dir, info, episode_index)
        num_frames = pq.read_metadata(parquet_path).num_rows
        record = payload[str(episode_index)]
        boundaries = record.get("subtask_completion_indices")
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError(f"Episode {episode_index}: expected at least one subtask boundary")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in boundaries):
            raise ValueError(f"Episode {episode_index}: subtask boundaries must be integers")
        if boundaries != sorted(set(boundaries)) or boundaries[0] <= 0 or boundaries[-1] >= num_frames:
            raise ValueError(
                f"Episode {episode_index}: boundaries must be unique, increasing, and inside (0, {num_frames})"
            )
        if boundary_count is None:
            boundary_count = len(boundaries)
        elif len(boundaries) != boundary_count:
            raise ValueError(
                "All episodes must use the same number of subtask boundaries: "
                f"expected {boundary_count}, got {len(boundaries)} for episode {episode_index}"
            )
        annotations[episode_index] = boundaries
    if boundary_count is None:
        raise ValueError("Subtask annotations are empty")
    return annotations, boundary_count + 1


def make_subtask_indices(num_frames: int, boundaries: list[int]) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not boundaries or boundaries != sorted(set(boundaries)):
        raise ValueError("boundaries must be a non-empty, strictly increasing list")
    if boundaries[0] <= 0 or boundaries[-1] >= num_frames:
        raise ValueError(f"boundaries must lie inside (0, {num_frames})")
    return np.searchsorted(np.asarray(boundaries), np.arange(num_frames), side="right").astype(np.int64)


def build_jobs(base_source: Path, stage1_source: Path, output_root: Path) -> tuple[DatasetJob, DatasetJob]:
    base_source = _resolved(base_source)
    stage1_source = _resolved(stage1_source)
    output_root = _resolved(output_root)
    base_info, base_schema = validate_dataset(base_source)
    stage1_info, stage1_schema = validate_dataset(stage1_source)

    annotation_path = base_source / "augmentation_metadata.json"
    annotations, group_count = load_subtask_annotations(annotation_path, base_source, base_info)
    return (
        DatasetJob(
            name="base",
            source=base_source,
            output=output_root / "base",
            grouping="manual_subtasks",
            group_count=group_count,
            schema=base_schema,
            info=base_info,
            annotations=annotations,
            annotation_path=annotation_path,
        ),
        DatasetJob(
            name="stage1",
            source=stage1_source,
            output=output_root / "stage1",
            grouping="whole_dataset",
            group_count=1,
            schema=stage1_schema,
            info=stage1_info,
        ),
    )


def expected_output_marker(job: DatasetJob, checkpoint_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset": str(job.source),
        "dataset_name": job.name,
        "schema": job.schema.name,
        "grouping": job.grouping,
        "checkpoint_dir": checkpoint_context["checkpoint_dir"],
        "checkpoint_step": checkpoint_context["checkpoint_step"],
        "video_storage": "materialized_copy",
    }


def validate_existing_output(job: DatasetJob, checkpoint_context: dict[str, Any]) -> None:
    if not job.output.exists():
        return
    marker_path = job.output / OUTPUT_MARKER
    if not marker_path.is_file():
        raise FileExistsError(f"Refusing to reuse unrecognized output dataset: {job.output}")
    existing = json.loads(marker_path.read_text())
    expected = expected_output_marker(job, checkpoint_context)
    if existing.get("grouping") == "four_manual_subtasks":
        existing["grouping"] = "manual_subtasks"
    if expected.get("grouping") == "four_manual_subtasks":
        expected["grouping"] = "manual_subtasks"
    if existing != expected:
        raise ValueError(f"Existing output marker does not match this run: {marker_path}")


def prepare_output_dataset(job: DatasetJob, checkpoint_context: dict[str, Any]) -> None:
    validate_existing_output(job, checkpoint_context)
    if job.output.exists():
        return

    job.output.parent.mkdir(parents=True, exist_ok=True)
    preparing = job.output.parent / f".{job.output.name}.preparing"
    if preparing.exists():
        raise FileExistsError(f"Stale preparation directory exists; inspect it before retrying: {preparing}")
    shutil.copytree(job.source, preparing, symlinks=False)
    common.atomic_write_json(preparing / OUTPUT_MARKER, expected_output_marker(job, checkpoint_context))
    preparing.rename(job.output)
    if (job.output / "videos").is_symlink():
        raise RuntimeError(f"Output videos must be materialized, not symlinked: {job.output}")


def add_subtask_indices(job: DatasetJob) -> None:
    if job.annotations is None:
        raise ValueError("Base job requires manual subtask annotations")
    for episode_index, boundaries in job.annotations.items():
        parquet_path = common.parquet_path_for_episode(job.output, job.info, episode_index)
        table = pq.read_table(parquet_path)
        stage_index = make_subtask_indices(table.num_rows, boundaries)
        table = common.replace_column(table, "stage_index", stage_index, pa.int64())
        common.atomic_write_parquet(parquet_path, table)


def add_base_stage_indices(job: DatasetJob) -> None:
    """Backward-compatible alias for the original Base+Stage1 wrapper."""
    add_subtask_indices(job)


def _prediction_worker(
    worker_index: int,
    device_name: str,
    episode_indices: list[int],
    dataset_dir: str,
    checkpoint_dir: str,
    config_name: str,
    batch_size: int,
    decode_chunk_size: int,
    future_frames: int,
    *,
    verify_training_parity: bool,
) -> None:
    import torch

    from hil_umi_advantage.training import config as config_lib
    from scripts import visualize_advantage_checkpoint as visualize

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | GPU worker {worker_index} | %(levelname)s | %(message)s",
        force=True,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available for worker device {device_name}")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset_path = Path(dataset_dir)
    info = common.read_dataset_info(dataset_path)
    schema = detect_advantage_dataset_schema(info["features"])
    train_config = config_lib.get_config(config_name)
    _, data_transform = visualize.build_training_transform(train_config)
    if verify_training_parity:
        visualize.verify_training_parity(train_config, dataset_path, data_transform)
    model = visualize.load_model(train_config, Path(checkpoint_dir), device)

    effective_batch_size = batch_size
    for sequence_index, episode_index in enumerate(episode_indices, start=1):
        parquet_path = common.parquet_path_for_episode(dataset_path, info, episode_index)
        existing_columns = set(pq.read_schema(parquet_path).names)
        if set(PREDICTION_COLUMNS).issubset(existing_columns):
            LOGGER.info("Skipping already predicted %s episode %06d", schema.name, episode_index)
            continue

        paths = episode_paths(dataset_path, info, schema, episode_index)
        episode_data = load_episode_data(paths.parquet, schema)
        visualize.validate_episode(paths, episode_data, info)
        LOGGER.info(
            "Predicting %s episode %06d (%d/%d, %d frames)",
            schema.name,
            episode_index,
            sequence_index,
            len(episode_indices),
            episode_data.num_frames,
        )
        absolute_values, effective_batch_size = visualize.infer_episode(
            model=model,
            device=device,
            train_config=train_config,
            data_transform=data_transform,
            paths=paths,
            episode_data=episode_data,
            ground_truth=np.zeros(episode_data.num_frames, dtype=np.float32),
            batch_size=effective_batch_size,
            decode_chunk_size=decode_chunk_size,
        )
        relative_advantage, effective_batch_size = common._infer_relative_episode(  # noqa: SLF001
            model=model,
            device=device,
            train_config=train_config,
            data_transform=data_transform,
            paths=paths,
            episode_data=episode_data,
            batch_size=effective_batch_size,
            decode_chunk_size=decode_chunk_size,
            future_frames=future_frames,
        )
        absolute_advantage = common.normalize_future_difference(absolute_values, future_frames)
        table = pq.read_table(parquet_path)
        table = common.replace_column(table, "relative_advantage", relative_advantage, pa.float32())
        table = common.replace_column(table, "absolute_value", absolute_values.astype(np.float32), pa.float32())
        table = common.replace_column(table, "absolute_advantage", absolute_advantage, pa.float32())
        common.atomic_write_parquet(parquet_path, table)


def ensure_devices_available(devices: list[str]) -> None:
    import torch

    parsed = [torch.device(device) for device in devices]
    cuda_devices = [device for device in parsed if device.type == "cuda"]
    if cuda_devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested, but PyTorch cannot access an NVIDIA driver")
    available_count = torch.cuda.device_count()
    for device in cuda_devices:
        index = device.index if device.index is not None else 0
        if index >= available_count:
            raise RuntimeError(f"Requested {device}, but only {available_count} CUDA devices are visible")


def run_prediction_workers(
    job: DatasetJob,
    checkpoint_dir: Path,
    config_name: str,
    devices: list[str],
    batch_size: int,
    decode_chunk_size: int,
    future_frames: int,
    *,
    verify_training_parity: bool,
) -> None:
    episodes = list(range(int(job.info["total_episodes"])))
    shards = [episodes[index :: len(devices)] for index in range(len(devices))]
    if len(devices) == 1:
        _prediction_worker(
            0,
            devices[0],
            shards[0],
            str(job.output),
            str(checkpoint_dir),
            config_name,
            batch_size,
            decode_chunk_size,
            future_frames,
            verify_training_parity=verify_training_parity,
        )
        return

    context = multiprocessing.get_context("spawn")
    processes = []
    for worker_index, (device, episode_indices) in enumerate(zip(devices, shards, strict=True)):
        process = context.Process(
            target=_prediction_worker,
            args=(
                worker_index,
                device,
                episode_indices,
                str(job.output),
                str(checkpoint_dir),
                config_name,
                batch_size,
                decode_chunk_size,
                future_frames,
            ),
            kwargs={"verify_training_parity": verify_training_parity and worker_index == 0},
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [(index, process.exitcode) for index, process in enumerate(processes) if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"Prediction workers failed for {job.name}: {failed}")


def group_ids_for_table(table: pa.Table, grouping: str) -> np.ndarray:
    if grouping in {"four_manual_subtasks", "manual_subtasks"}:
        if "stage_index" not in table.column_names:
            raise ValueError("Base output is missing stage_index")
        group_ids = np.asarray(table["stage_index"].to_pylist(), dtype=np.int64)
        if np.any(group_ids < 0):
            raise ValueError("stage_index must be non-negative")
        return group_ids
    if grouping == "whole_dataset":
        return np.zeros(table.num_rows, dtype=np.int64)
    raise ValueError(f"Unknown labeling grouping: {grouping}")


def compute_group_thresholds(job: DatasetJob, top_percent: float) -> dict[int, float]:
    if not 0.0 < top_percent < 100.0:
        raise ValueError("top_percent must be between 0 and 100")
    values_by_group: dict[int, list[np.ndarray]] = {group: [] for group in range(job.group_count)}
    for episode_index in range(int(job.info["total_episodes"])):
        table = pq.read_table(common.parquet_path_for_episode(job.output, job.info, episode_index))
        group_ids = group_ids_for_table(table, job.grouping)
        if np.any(group_ids >= job.group_count):
            raise ValueError(f"{job.name} contains stage_index outside [0, {job.group_count})")
        advantages = np.asarray(table["absolute_advantage"].to_pylist(), dtype=np.float32)
        if not np.isfinite(advantages).all():
            raise ValueError(f"{job.name} episode {episode_index} contains non-finite absolute_advantage")
        for group, arrays in values_by_group.items():
            arrays.append(advantages[group_ids == group])

    thresholds = {}
    for group, arrays in values_by_group.items():
        values = np.concatenate(arrays)
        if len(values) == 0:
            raise ValueError(f"{job.name} labeling group {group} has no frames")
        thresholds[group] = float(np.percentile(values, 100.0 - top_percent))
    return thresholds


def apply_labels(job: DatasetJob, thresholds: dict[int, float]) -> dict[int, dict[str, float | int]]:
    counts = {group: {"total": 0, "positive": 0} for group in thresholds}
    for episode_index in range(int(job.info["total_episodes"])):
        parquet_path = common.parquet_path_for_episode(job.output, job.info, episode_index)
        table = pq.read_table(parquet_path)
        group_ids = group_ids_for_table(table, job.grouping)
        if np.any(group_ids >= job.group_count):
            raise ValueError(f"{job.name} contains stage_index outside [0, {job.group_count})")
        advantages = np.asarray(table["absolute_advantage"].to_pylist(), dtype=np.float32)
        task_index = np.zeros(table.num_rows, dtype=np.int64)
        for group, threshold in thresholds.items():
            group_mask = group_ids == group
            positive = group_mask & (advantages >= threshold)
            task_index[positive] = 1
            counts[group]["total"] += int(np.sum(group_mask))
            counts[group]["positive"] += int(np.sum(positive))
        table = common.replace_column(table, "task_index", task_index, pa.int64())
        common.atomic_write_parquet(parquet_path, table)

    return {
        group: {
            **group_counts,
            "threshold": thresholds[group],
            "positive_fraction": group_counts["positive"] / group_counts["total"],
        }
        for group, group_counts in counts.items()
    }


def update_metadata(job: DatasetJob) -> None:
    info = common.read_dataset_info(job.output)
    feature_types: dict[str, tuple[str, pa.DataType]] = {
        "relative_advantage": ("float32", pa.float32()),
        "absolute_value": ("float32", pa.float32()),
        "absolute_advantage": ("float32", pa.float32()),
        "task_index": ("int64", pa.int64()),
    }
    if job.grouping in {"four_manual_subtasks", "manual_subtasks"}:
        feature_types = {"stage_index": ("int64", pa.int64()), **feature_types}
    for name, (dtype, _) in feature_types.items():
        info.setdefault("features", {})[name] = {"dtype": dtype, "shape": [1], "names": [name]}
    info["total_tasks"] = 2
    common.atomic_write_json(job.output / "meta/info.json", info)

    episode_values: dict[int, dict[str, np.ndarray]] = {}
    all_values: dict[str, list[np.ndarray]] = {name: [] for name in feature_types}
    for episode_index in range(int(info["total_episodes"])):
        table = pq.read_table(
            common.parquet_path_for_episode(job.output, info, episode_index),
            columns=list(feature_types),
        )
        episode_values[episode_index] = {}
        for name in feature_types:
            values = np.asarray(table[name].to_pylist())
            episode_values[episode_index][name] = values
            all_values[name].append(values)

    episode_stats_path = job.output / "meta/episodes_stats.jsonl"
    if episode_stats_path.is_file():
        records = [json.loads(line) for line in episode_stats_path.read_text().splitlines() if line.strip()]
        for record in records:
            episode_index = int(record["episode_index"])
            for name, values in episode_values[episode_index].items():
                record.setdefault("stats", {})[name] = common.compute_scalar_stats(values)
        _atomic_write_jsonl(episode_stats_path, records)

    stats_path = job.output / "meta/stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    for name, arrays in all_values.items():
        stats[name] = common.compute_scalar_stats(np.concatenate(arrays))
    common.atomic_write_json(stats_path, stats)

    tasks = [
        {"task_index": 0, "task": f"{TASK_PROMPT}, Advantage: negative"},
        {"task_index": 1, "task": f"{TASK_PROMPT}, Advantage: positive"},
    ]
    _atomic_write_jsonl(job.output / "meta/tasks.jsonl", tasks)
    episodes_path = job.output / "meta/episodes.jsonl"
    if episodes_path.is_file():
        records = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
        task_strings = [record["task"] for record in tasks]
        for record in records:
            record["tasks"] = task_strings
        _atomic_write_jsonl(episodes_path, records)


def label_job(
    job: DatasetJob,
    checkpoint_dir: Path,
    checkpoint_context: dict[str, Any],
    config_name: str,
    devices: list[str],
    batch_size: int,
    decode_chunk_size: int,
    future_frames: int,
    top_percent: float,
    *,
    verify_training_parity: bool,
) -> dict[str, Any]:
    prepare_output_dataset(job, checkpoint_context)
    completed_path = job.output / LABELING_MANIFEST
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text())
        completed_grouping = completed.get("grouping")
        if completed_grouping == "four_manual_subtasks":
            completed_grouping = "manual_subtasks"
        expected_values = {
            "checkpoint_dir": checkpoint_context["checkpoint_dir"],
            "checkpoint_step": checkpoint_context["checkpoint_step"],
            "source_dataset": str(job.source),
            "future_frames": future_frames,
            "top_percent": top_percent,
            "grouping": job.grouping,
            "advantage_source": "absolute_advantage",
        }
        actual_values = {**completed, "grouping": completed_grouping}
        mismatches = {
            key: {"expected": value, "actual": actual_values.get(key)}
            for key, value in expected_values.items()
            if actual_values.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Completed output does not match the requested labeling parameters: {mismatches}")
        LOGGER.info("Output already completed; leaving it unchanged: %s", job.output)
        return completed

    if job.grouping in {"four_manual_subtasks", "manual_subtasks"}:
        add_subtask_indices(job)

    prediction_manifest = {
        **checkpoint_context,
        "source_dataset": str(job.source),
        "output_dataset": str(job.output),
        "schema": job.schema.name,
        "future_frames": future_frames,
        "devices": devices,
        "advantage_source": "absolute_advantage",
    }
    prediction_manifest_path = job.output / "meta/advantage_prediction.json"
    if prediction_manifest_path.is_file():
        existing = json.loads(prediction_manifest_path.read_text())
        comparable = ("checkpoint_dir", "checkpoint_step", "source_dataset", "future_frames")
        if any(existing.get(key) != prediction_manifest.get(key) for key in comparable):
            raise ValueError(f"Existing prediction manifest does not match this run: {prediction_manifest_path}")
    common.atomic_write_json(prediction_manifest_path, prediction_manifest)

    run_prediction_workers(
        job,
        checkpoint_dir,
        config_name,
        devices,
        batch_size,
        decode_chunk_size,
        future_frames,
        verify_training_parity=verify_training_parity,
    )
    thresholds = compute_group_thresholds(job, top_percent)
    group_summary = apply_labels(job, thresholds)
    update_metadata(job)

    manifest = {
        **prediction_manifest,
        "grouping": job.grouping,
        "group_count": job.group_count,
        "top_percent": top_percent,
        "threshold_semantics": (f"positive iff absolute_advantage >= per-group {100.0 - top_percent:g}th percentile"),
        "group_summary": group_summary,
        "total_episodes": int(job.info["total_episodes"]),
        "total_frames": int(job.info["total_frames"]),
        "video_storage": "materialized_copy",
    }
    if job.annotation_path is not None:
        manifest["annotations"] = str(job.annotation_path)
        manifest["annotations_sha256"] = common.sha256_file(job.annotation_path)
    common.atomic_write_json(completed_path, manifest)
    LOGGER.info("Completed %s labels: %s", job.name, json.dumps(group_summary, sort_keys=True))
    return manifest


def write_root_summary(output_root: Path, manifests: dict[str, dict[str, Any]]) -> None:
    common.atomic_write_json(output_root / "mix_acbc_labeling.json", {"datasets": manifests})
    status = (
        "# Mix advantage labels\n\n"
        "This directory contains labels produced by the base+stage1 mix advantage checkpoint.\n\n"
        "- `base/`: base data, top 30% labeled independently within each of four manual subtasks.\n"
        "- `stage1/`: stage1 data, top 30% labeled across the whole dataset.\n"
        "- The older `../../base/labeled` dataset is unrelated and was not used.\n"
    )
    temporary_path = output_root / ".STATUS.md.tmp"
    temporary_path.write_text(status)
    os.replace(temporary_path, output_root / "STATUS.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, default=DEFAULT_BASE_SOURCE)
    parser.add_argument("--stage1-source", type=Path, default=DEFAULT_STAGE1_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--devices", type=common.parse_devices, default=common.parse_devices("cuda:0,cuda:1"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-chunk-size", type=int, default=64)
    parser.add_argument("--future-frames", type=int, default=50)
    parser.add_argument("--top-percent", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--verify-training-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    if args.batch_size <= 0 or args.decode_chunk_size <= 0 or args.future_frames <= 0:
        raise ValueError("batch_size, decode_chunk_size, and future_frames must be positive")
    if args.decode_chunk_size < args.batch_size:
        raise ValueError("decode_chunk_size must be greater than or equal to batch_size")
    if not 0.0 < args.top_percent < 100.0:
        raise ValueError("top_percent must be between 0 and 100")

    validate_path_separation(args.base_source, args.stage1_source, args.output_root)
    checkpoint_dir = common.resolve_latest_complete_checkpoint(args.checkpoint)
    checkpoint_metadata = read_checkpoint_metadata(checkpoint_dir)
    checkpoint_context = validate_mix_checkpoint_metadata(
        checkpoint_metadata,
        checkpoint_dir,
        args.base_source,
        args.stage1_source,
        args.config_name,
    )
    jobs = build_jobs(args.base_source, args.stage1_source, args.output_root)
    for job in jobs:
        validate_existing_output(job, checkpoint_context)
        LOGGER.info(
            "Validated %s: schema=%s episodes=%s frames=%s output=%s",
            job.name,
            job.schema.name,
            job.info["total_episodes"],
            job.info["total_frames"],
            job.output,
        )
    LOGGER.info("Validated mix checkpoint: %s", json.dumps(checkpoint_context, sort_keys=True))
    if args.validate_only:
        LOGGER.info("Validation-only mode complete; no output was written")
        return

    ensure_devices_available(args.devices)
    manifests = {}
    for job in jobs:
        manifests[job.name] = label_job(
            job,
            checkpoint_dir,
            checkpoint_context,
            args.config_name,
            args.devices,
            args.batch_size,
            args.decode_chunk_size,
            args.future_frames,
            args.top_percent,
            verify_training_parity=args.verify_training_parity and job.name == "base",
        )
    write_root_summary(_resolved(args.output_root), manifests)
    LOGGER.info("Mix ACBC labeling complete: %s", _resolved(args.output_root))


if __name__ == "__main__":
    main()
