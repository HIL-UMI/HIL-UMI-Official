#!/usr/bin/env python3
# Modified for HIL-UMI.
"""Build a four-stage towel-fold ACBC dataset from manual boundaries.

This script never modifies the source dataset. It creates a derived LeRobot
dataset, adds an integer ``stage_index`` from manually confirmed boundaries,
runs the two-view ADVANTAGE advantage estimator, and marks the top 30 percent of
``absolute_advantage`` within each stage across all episodes as positive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # ``python scripts/label_towel_fold_acbc.py`` puts only ``scripts/`` on
    # sys.path.  Multiprocessing ``spawn`` re-imports this file in each worker,
    # so add the repository root explicitly before importing sibling scripts.
    sys.path.insert(0, str(PROJECT_ROOT))

from hil_umi_advantage.shared import paths as component_paths  # noqa: E402
from hil_umi_advantage.training.advantage_schema import canonicalize_masked_state  # noqa: E402

LOGGER = logging.getLogger("towel_acbc_labeling")
PREDICTION_COLUMNS = ("relative_advantage", "absolute_value", "absolute_advantage")
DERIVED_COLUMNS = ("stage_index", *PREDICTION_COLUMNS, "task_index")
DEFAULT_DATASET = component_paths.data_root() / "fold_towel/base/origin"
DEFAULT_CHECKPOINT_ROOT = component_paths.checkpoint_root() / "advantage/ADVANTAGE_TORCH_TOWEL_FOLD/towel_base_v1"
DEFAULT_OUTPUT = component_paths.data_root() / "fold_towel/base/labeled"
DEFAULT_CONFIG_NAME = "ADVANTAGE_TORCH_TOWEL_FOLD"
TASK_PROMPT = "fold the towel on the table and put it into the basket"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, path)


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary_path)
    os.replace(temporary_path, path)


def replace_column(table: pa.Table, name: str, values: np.ndarray, arrow_type: pa.DataType) -> pa.Table:
    if name in table.column_names:
        table = table.drop_columns([name])
    return table.append_column(name, pa.array(values, type=arrow_type))


def read_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    return json.loads(info_path.read_text())


def parquet_path_for_episode(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_index = episode_index // int(info["chunks_size"])
    return dataset_dir / info["data_path"].format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
    )


def load_and_validate_annotations(
    annotation_path: Path,
    dataset_dir: Path,
    info: dict[str, Any],
) -> dict[int, list[int]]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Manual annotation file not found: {annotation_path}")
    payload = json.loads(annotation_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Annotation file must contain a JSON object")

    total_episodes = int(info["total_episodes"])
    expected_indices = set(range(total_episodes))
    actual_indices = {int(key) for key in payload}
    missing = sorted(expected_indices - actual_indices)
    extra = sorted(actual_indices - expected_indices)
    if missing or extra:
        raise ValueError(f"Annotations must cover every episode; missing={missing}, extra={extra}")

    annotations = {}
    for episode_index in range(total_episodes):
        parquet_path = parquet_path_for_episode(dataset_dir, info, episode_index)
        num_frames = pq.read_metadata(parquet_path).num_rows
        record = payload[str(episode_index)]
        boundaries = record.get("subtask_completion_indices")
        if not isinstance(boundaries, list) or len(boundaries) != 3:
            raise ValueError(f"Episode {episode_index}: expected exactly three boundaries")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in boundaries):
            raise ValueError(f"Episode {episode_index}: boundaries must be integers")
        if not 0 < boundaries[0] < boundaries[1] < boundaries[2] < num_frames:
            raise ValueError(f"Episode {episode_index}: boundaries must satisfy 0 < b1 < b2 < b3 < {num_frames}")
        annotations[episode_index] = boundaries
    return annotations


def make_stage_indices(num_frames: int, boundaries: list[int]) -> np.ndarray:
    b1, b2, b3 = boundaries
    if not 0 < b1 < b2 < b3 < num_frames:
        raise ValueError(f"Invalid boundaries for {num_frames} frames: {boundaries}")
    stage_index = np.empty(num_frames, dtype=np.int64)
    stage_index[:b1] = 0
    stage_index[b1:b2] = 1
    stage_index[b2:b3] = 2
    stage_index[b3:] = 3
    return stage_index


def resolve_latest_complete_checkpoint(checkpoint_root: Path) -> Path:
    checkpoint_root = checkpoint_root.expanduser().resolve()
    if (checkpoint_root / "model.safetensors").is_file() and (checkpoint_root / "metadata.pt").is_file():
        checkpoint_dir = checkpoint_root
    else:
        candidates = [
            child
            for child in checkpoint_root.iterdir()
            if child.is_dir()
            and child.name.isdigit()
            and (child / "model.safetensors").is_file()
            and (child / "metadata.pt").is_file()
        ]
        if not candidates:
            raise FileNotFoundError(f"No complete checkpoint found under {checkpoint_root}")
        checkpoint_dir = max(candidates, key=lambda path: int(path.name))

    import torch

    metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
    global_step = int(metadata.get("global_step", -1))
    if checkpoint_dir.name.isdigit() and global_step != int(checkpoint_dir.name):
        raise ValueError(
            f"Checkpoint metadata mismatch: directory={checkpoint_dir.name}, metadata global_step={global_step}"
        )
    return checkpoint_dir


def prepare_derived_dataset(
    source_dir: Path,
    output_dir: Path,
    annotation_path: Path,
    annotations: dict[int, list[int]],
    info: dict[str, Any],
) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    marker_path = output_dir / "meta/acbc_source.json"
    if output_dir.exists():
        if not marker_path.is_file():
            raise FileExistsError(f"Refusing to reuse unrecognized output directory: {output_dir}")
        marker = json.loads(marker_path.read_text())
        if Path(marker["source_dataset"]).resolve() != source_dir:
            raise ValueError(f"Existing output was derived from a different source: {marker['source_dataset']}")
    else:
        output_dir.mkdir(parents=True)
        shutil.copytree(source_dir / "data", output_dir / "data")
        shutil.copytree(source_dir / "meta", output_dir / "meta")
        os.symlink(source_dir / "videos", output_dir / "videos", target_is_directory=True)

    shutil.copy2(annotation_path, output_dir / "augmentation_metadata.json")
    marker = {
        "source_dataset": str(source_dir),
        "annotation_file": str(annotation_path.resolve()),
        "stage_definition": ["[0,b1)", "[b1,b2)", "[b2,b3)", "[b3,N)"],
        "stage_progress_gt_policy": "preserved_from_source_and_not_used_for_acbc_grouping",
    }
    atomic_write_json(marker_path, marker)

    for episode_index, boundaries in annotations.items():
        parquet_path = parquet_path_for_episode(output_dir, info, episode_index)
        table = pq.read_table(parquet_path)
        stage_index = make_stage_indices(table.num_rows, boundaries)
        table = replace_column(table, "stage_index", stage_index, pa.int64())
        atomic_write_parquet(parquet_path, table)


def normalize_future_difference(values: np.ndarray, future_frames: int) -> np.ndarray:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1D array")
    if future_frames <= 0:
        raise ValueError("future_frames must be positive")
    output = np.zeros_like(values, dtype=np.float32)
    last_index = len(values) - 1
    for frame_index in range(len(values)):
        future_index = min(frame_index + future_frames, last_index)
        distance = future_index - frame_index
        if distance == 0:
            output[frame_index] = 0.0
        else:
            output[frame_index] = (values[future_index] - values[frame_index]) / distance * future_frames
    return np.clip(output, -1.0, 1.0)


def normalize_direct_advantage(raw_values: np.ndarray, future_frames: int) -> np.ndarray:
    if raw_values.ndim != 1 or len(raw_values) == 0:
        raise ValueError("raw_values must be a non-empty 1D array")
    output = raw_values.astype(np.float32, copy=True)
    last_index = len(output) - 1
    for frame_index in range(len(output)):
        future_index = min(frame_index + future_frames, last_index)
        distance = future_index - frame_index
        if distance == 0:
            output[frame_index] = 0.0
        elif distance != future_frames:
            output[frame_index] = output[frame_index] / distance * future_frames
    return np.clip(output, -1.0, 1.0)


def _infer_relative_episode(
    *,
    model,
    device,
    train_config,
    data_transform,
    paths,
    episode_data,
    batch_size: int,
    decode_chunk_size: int,
    future_frames: int,
) -> tuple[np.ndarray, int]:
    from lerobot.common.datasets.video_utils import decode_video_frames
    import torch

    from scripts import visualize_advantage_checkpoint as visualize

    raw_predictions = np.full(episode_data.num_frames, np.nan, dtype=np.float32)
    effective_batch_size = batch_size
    contract_checked = False
    future_indices = np.minimum(np.arange(episode_data.num_frames) + future_frames, episode_data.num_frames - 1)

    for chunk_start in range(0, episode_data.num_frames, decode_chunk_size):
        chunk_end = min(chunk_start + decode_chunk_size, episode_data.num_frames)
        base_indices = np.arange(chunk_start, chunk_end)
        chunk_future_indices = future_indices[chunk_start:chunk_end]
        base_timestamps = episode_data.timestamps[base_indices].tolist()
        future_timestamps = episode_data.timestamps[chunk_future_indices].tolist()
        base_front = decode_video_frames(paths.front_video, base_timestamps, tolerance_s=1e-4, backend="pyav")
        base_wrist = decode_video_frames(paths.wrist_video, base_timestamps, tolerance_s=1e-4, backend="pyav")
        future_front = decode_video_frames(paths.front_video, future_timestamps, tolerance_s=1e-4, backend="pyav")
        future_wrist = decode_video_frames(paths.wrist_video, future_timestamps, tolerance_s=1e-4, backend="pyav")

        local_start = 0
        while local_start < chunk_end - chunk_start:
            current_batch_size = min(effective_batch_size, chunk_end - chunk_start - local_start)
            global_start = chunk_start + local_start
            items = []
            for offset in range(current_batch_size):
                frame_index = global_start + offset
                future_index = int(future_indices[frame_index])
                local_index = local_start + offset
                items.append(
                    data_transform(
                        visualize.build_raw_pair(
                            current_front=future_front[local_index],
                            current_wrist=future_wrist[local_index],
                            reference_front=base_front[local_index],
                            reference_wrist=base_wrist[local_index],
                            state=canonicalize_masked_state(
                                episode_data.states[future_index],
                                train_config.model.action_dim,
                            ),
                            action_chunk=visualize.make_action_chunk(
                                episode_data.actions,
                                frame_index=future_index,
                                action_horizon=train_config.model.action_horizon,
                            ),
                            episode_length=episode_data.num_frames,
                            frame_index=int(episode_data.frame_indices[future_index]),
                            episode_index=paths.episode_index,
                            progress=0.0,
                        )
                    )
                )
            observation = visualize.collate_like_training(items)
            if not contract_checked:
                visualize.assert_model_input_contract(observation)
                contract_checked = True
            observation = visualize.move_observation(observation, device)
            try:
                with torch.inference_mode():
                    values = model.sample_values(device, observation)[:, 0]
                raw_predictions[global_start : global_start + current_batch_size] = (
                    values.float().cpu().numpy().astype(np.float32)
                )
                local_start += current_batch_size
            except torch.cuda.OutOfMemoryError:
                del observation
                torch.cuda.empty_cache()
                if effective_batch_size == 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                LOGGER.warning("CUDA OOM during relative inference; retrying with batch_size=%d", effective_batch_size)

    if not np.isfinite(raw_predictions).all():
        raise RuntimeError("Relative advantage inference produced non-finite values")
    return normalize_direct_advantage(raw_predictions, future_frames), effective_batch_size


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
    checkpoint_path = Path(checkpoint_dir)
    info = read_dataset_info(dataset_path)
    train_config = config_lib.get_config(config_name)
    _, data_transform = visualize.build_training_transform(train_config)
    if verify_training_parity:
        visualize.verify_training_parity(train_config, Path(train_config.data.repo_id), data_transform)
    model = visualize.load_model(train_config, checkpoint_path, device)

    for sequence_index, episode_index in enumerate(episode_indices, start=1):
        parquet_path = parquet_path_for_episode(dataset_path, info, episode_index)
        existing_columns = set(pq.read_schema(parquet_path).names)
        if set(PREDICTION_COLUMNS).issubset(existing_columns):
            LOGGER.info("Skipping already predicted episode %06d", episode_index)
            continue
        paths = visualize.get_episode_paths(dataset_path, episode_index, info)
        episode_data = visualize.load_episode_data(paths.parquet)
        visualize.validate_episode(paths, episode_data, info)
        LOGGER.info(
            "Predicting episode %06d (%d/%d, %d frames)",
            episode_index,
            sequence_index,
            len(episode_indices),
            episode_data.num_frames,
        )
        absolute_values, absolute_batch_size = visualize.infer_episode(
            model=model,
            device=device,
            train_config=train_config,
            data_transform=data_transform,
            paths=paths,
            episode_data=episode_data,
            ground_truth=np.zeros(episode_data.num_frames, dtype=np.float32),
            batch_size=batch_size,
            decode_chunk_size=decode_chunk_size,
        )
        relative_advantage, relative_batch_size = _infer_relative_episode(
            model=model,
            device=device,
            train_config=train_config,
            data_transform=data_transform,
            paths=paths,
            episode_data=episode_data,
            batch_size=min(batch_size, absolute_batch_size),
            decode_chunk_size=decode_chunk_size,
            future_frames=future_frames,
        )
        absolute_advantage = normalize_future_difference(absolute_values, future_frames)
        table = pq.read_table(parquet_path)
        table = replace_column(table, "relative_advantage", relative_advantage, pa.float32())
        table = replace_column(table, "absolute_value", absolute_values.astype(np.float32), pa.float32())
        table = replace_column(table, "absolute_advantage", absolute_advantage, pa.float32())
        atomic_write_parquet(parquet_path, table)
        LOGGER.info(
            "Completed episode %06d (absolute_batch=%d, relative_batch=%d)",
            episode_index,
            absolute_batch_size,
            relative_batch_size,
        )


def run_prediction_workers(
    *,
    dataset_dir: Path,
    checkpoint_dir: Path,
    config_name: str,
    devices: list[str],
    batch_size: int,
    decode_chunk_size: int,
    future_frames: int,
    verify_training_parity: bool,
) -> None:
    info = read_dataset_info(dataset_dir)
    episodes = list(range(int(info["total_episodes"])))
    shards = [episodes[index :: len(devices)] for index in range(len(devices))]
    if len(devices) == 1:
        _prediction_worker(
            0,
            devices[0],
            shards[0],
            str(dataset_dir),
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
                str(dataset_dir),
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
        raise RuntimeError(f"Prediction workers failed: {failed}")


def compute_stage_thresholds(dataset_dir: Path, info: dict[str, Any], top_percent: float) -> dict[int, float]:
    if not 0.0 < top_percent < 100.0:
        raise ValueError("top_percent must be between 0 and 100")
    values_by_stage: dict[int, list[np.ndarray]] = {stage: [] for stage in range(4)}
    for episode_index in range(int(info["total_episodes"])):
        table = pq.read_table(
            parquet_path_for_episode(dataset_dir, info, episode_index),
            columns=["stage_index", "absolute_advantage"],
        )
        stages = np.asarray(table["stage_index"].to_pylist(), dtype=np.int64)
        advantages = np.asarray(table["absolute_advantage"].to_pylist(), dtype=np.float32)
        if not np.isfinite(advantages).all():
            raise ValueError(f"Episode {episode_index} contains non-finite absolute_advantage")
        for stage in range(4):
            values_by_stage[stage].append(advantages[stages == stage])

    thresholds = {}
    for stage in range(4):
        values = np.concatenate(values_by_stage[stage])
        if len(values) == 0:
            raise ValueError(f"Stage {stage} has no frames")
        thresholds[stage] = float(np.percentile(values, 100.0 - top_percent))
    return thresholds


def apply_acbc_labels(
    dataset_dir: Path,
    info: dict[str, Any],
    thresholds: dict[int, float],
) -> dict[int, dict[str, float | int]]:
    counts = {stage: {"total": 0, "positive": 0} for stage in range(4)}
    for episode_index in range(int(info["total_episodes"])):
        parquet_path = parquet_path_for_episode(dataset_dir, info, episode_index)
        table = pq.read_table(parquet_path)
        stages = np.asarray(table["stage_index"].to_pylist(), dtype=np.int64)
        advantages = np.asarray(table["absolute_advantage"].to_pylist(), dtype=np.float32)
        task_index = np.zeros(table.num_rows, dtype=np.int64)
        for stage in range(4):
            mask = stages == stage
            positive = mask & (advantages >= thresholds[stage])
            task_index[positive] = 1
            counts[stage]["total"] += int(np.sum(mask))
            counts[stage]["positive"] += int(np.sum(positive))
        table = replace_column(table, "task_index", task_index, pa.int64())
        atomic_write_parquet(parquet_path, table)

    summary = {}
    for stage, stage_counts in counts.items():
        summary[stage] = {
            **stage_counts,
            "threshold": thresholds[stage],
            "positive_fraction": stage_counts["positive"] / stage_counts["total"],
        }
    return summary


def compute_scalar_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Statistics require a non-empty 1D array")
    if not np.isfinite(values).all():
        raise ValueError("Statistics input contains non-finite values")
    return {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [len(values)],
        "q01": [float(np.quantile(values, 0.01))],
        "q10": [float(np.quantile(values, 0.10))],
        "q50": [float(np.quantile(values, 0.50))],
        "q90": [float(np.quantile(values, 0.90))],
        "q99": [float(np.quantile(values, 0.99))],
    }


def update_lerobot_metadata(dataset_dir: Path, info: dict[str, Any]) -> None:
    feature_types = {
        "stage_index": ("int64", pa.int64()),
        "relative_advantage": ("float32", pa.float32()),
        "absolute_value": ("float32", pa.float32()),
        "absolute_advantage": ("float32", pa.float32()),
        "task_index": ("int64", pa.int64()),
    }
    for name, (dtype, _) in feature_types.items():
        info.setdefault("features", {})[name] = {"dtype": dtype, "shape": [1], "names": [name]}
    info["total_tasks"] = 2
    atomic_write_json(dataset_dir / "meta/info.json", info)

    episode_values: dict[int, dict[str, np.ndarray]] = {}
    all_values: dict[str, list[np.ndarray]] = {name: [] for name in feature_types}
    for episode_index in range(int(info["total_episodes"])):
        table = pq.read_table(
            parquet_path_for_episode(dataset_dir, info, episode_index),
            columns=list(feature_types),
        )
        episode_values[episode_index] = {}
        for name in feature_types:
            values = np.asarray(table[name].to_pylist())
            episode_values[episode_index][name] = values
            all_values[name].append(values)

    episodes_stats_path = dataset_dir / "meta/episodes_stats.jsonl"
    if episodes_stats_path.is_file():
        records = [json.loads(line) for line in episodes_stats_path.read_text().splitlines() if line.strip()]
        for record in records:
            episode_index = int(record["episode_index"])
            if episode_index not in episode_values:
                continue
            stats = record.setdefault("stats", {})
            for name, values in episode_values[episode_index].items():
                stats[name] = compute_scalar_stats(values)
        temporary_path = episodes_stats_path.with_name(f".{episodes_stats_path.name}.tmp")
        temporary_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        os.replace(temporary_path, episodes_stats_path)

    stats_path = dataset_dir / "meta/stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    for name, arrays in all_values.items():
        stats[name] = compute_scalar_stats(np.concatenate(arrays))
    atomic_write_json(stats_path, stats)

    tasks = [
        {"task_index": 0, "task": f"{TASK_PROMPT}, Advantage: negative"},
        {"task_index": 1, "task": f"{TASK_PROMPT}, Advantage: positive"},
    ]
    tasks_path = dataset_dir / "meta/tasks.jsonl"
    temporary_path = tasks_path.with_name(f".{tasks_path.name}.tmp")
    temporary_path.write_text("".join(json.dumps(record) + "\n" for record in tasks))
    os.replace(temporary_path, tasks_path)

    episodes_path = dataset_dir / "meta/episodes.jsonl"
    if episodes_path.is_file():
        records = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
        task_strings = [record["task"] for record in tasks]
        for record in records:
            record["tasks"] = task_strings
        temporary_path = episodes_path.with_name(f".{episodes_path.name}.tmp")
        temporary_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        os.replace(temporary_path, episodes_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("At least one device is required")
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--annotations", type=Path, help="Default: <dataset>/augmentation_metadata.json")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("cuda:0,cuda:1"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-chunk-size", type=int, default=64)
    parser.add_argument("--future-frames", type=int, default=50)
    parser.add_argument("--top-percent", type=float, default=30.0)
    parser.add_argument(
        "--verify-training-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    source_dir = args.dataset.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    annotation_path = (args.annotations or source_dir / "augmentation_metadata.json").expanduser().resolve()
    info = read_dataset_info(source_dir)
    annotations = load_and_validate_annotations(annotation_path, source_dir, info)
    checkpoint_dir = resolve_latest_complete_checkpoint(args.checkpoint_root)
    LOGGER.info("Resolved checkpoint: %s", checkpoint_dir)

    prepare_derived_dataset(source_dir, output_dir, annotation_path, annotations, info)
    prediction_manifest_path = output_dir / "meta/advantage_prediction.json"
    prediction_manifest = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": int(checkpoint_dir.name) if checkpoint_dir.name.isdigit() else None,
        "future_frames": args.future_frames,
        "source_dataset": str(source_dir),
        "annotations": str(annotation_path),
        "annotations_sha256": sha256_file(annotation_path),
        "devices": args.devices,
    }
    if prediction_manifest_path.is_file():
        existing = json.loads(prediction_manifest_path.read_text())
        comparable_keys = ("config_name", "checkpoint_dir", "future_frames", "annotations_sha256")
        if any(existing.get(key) != prediction_manifest[key] for key in comparable_keys):
            raise ValueError(
                "Existing prediction manifest does not match this run. Use a new output directory to avoid mixing labels."
            )
    atomic_write_json(prediction_manifest_path, prediction_manifest)

    run_prediction_workers(
        dataset_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        config_name=args.config_name,
        devices=args.devices,
        batch_size=args.batch_size,
        decode_chunk_size=args.decode_chunk_size,
        future_frames=args.future_frames,
        verify_training_parity=args.verify_training_parity,
    )
    output_info = read_dataset_info(output_dir)
    thresholds = compute_stage_thresholds(output_dir, output_info, args.top_percent)
    stage_summary = apply_acbc_labels(output_dir, output_info, thresholds)
    update_lerobot_metadata(output_dir, output_info)

    provenance = {
        **prediction_manifest,
        "output_dataset": str(output_dir),
        "stage_source": "manual integer boundaries from augmentation_metadata.json",
        "stage_progress_gt": "preserved from source; not used for grouping",
        "top_percent": args.top_percent,
        "threshold_semantics": "positive iff absolute_advantage >= per-stage percentile threshold",
        "stage_summary": stage_summary,
    }
    atomic_write_json(output_dir / "meta/acbc_labeling.json", provenance)
    LOGGER.info("ACBC dataset complete: %s", output_dir)
    LOGGER.info("Stage summary: %s", json.dumps(stage_summary, sort_keys=True))


if __name__ == "__main__":
    main()
