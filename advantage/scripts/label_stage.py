#!/usr/bin/env python3
# Modified for HIL-UMI.
"""Automatically label stage_progress_gt for a collected task stage.

For every episode with ``N`` frames in ``data/<task>/<stage>/origin``, the
label grows linearly from 0 to ``N / mean_base_episode_length``. The base mean
is computed from every episode in ``data/<task>/base/origin``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAME = "stage_progress_gt"


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    length: int
    parquet_path: Path


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    info: dict[str, Any]
    episodes: tuple[EpisodeSpec, ...]


def parse_path_component(value: str) -> str:
    """Accept one directory component and reject absolute/traversal paths."""
    if not value or value in {".", ".."} or Path(value).name != value:
        raise argparse.ArgumentTypeError(f"Expected one directory name, got {value!r}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(record)
    return records


def _positive_int(value: Any, *, field: str, context: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}: {field} must be a positive integer, got {value!r}")
    return value


def parquet_path_for_episode(
    dataset_dir: Path,
    data_path_template: str,
    chunks_size: int,
    episode_index: int,
) -> Path:
    chunk_index = episode_index // chunks_size
    try:
        relative_path = Path(
            data_path_template.format(
                episode_chunk=chunk_index,
                chunk_index=chunk_index,
                episode_index=episode_index,
            )
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unsupported data_path template: {data_path_template!r}") from error
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"data_path must stay within the dataset directory: {data_path_template!r}")
    return dataset_dir / relative_path


def load_and_validate_dataset(dataset_dir: Path) -> DatasetSpec:
    """Load LeRobot metadata and verify every declared episode parquet."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    info = read_json(dataset_dir / "meta/info.json")
    episodes_records = read_jsonl(dataset_dir / "meta/episodes.jsonl")
    total_episodes = _positive_int(info.get("total_episodes"), field="total_episodes", context=dataset_dir)
    chunks_size = _positive_int(info.get("chunks_size"), field="chunks_size", context=dataset_dir)
    data_path_template = info.get("data_path")
    if not isinstance(data_path_template, str) or not data_path_template:
        raise ValueError(f"{dataset_dir}: info.json data_path must be a non-empty string")
    if len(episodes_records) != total_episodes:
        raise ValueError(f"{dataset_dir}: expected {total_episodes} episode records, found {len(episodes_records)}")

    records_by_index: dict[int, dict[str, Any]] = {}
    for record in episodes_records:
        episode_index = record.get("episode_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(f"{dataset_dir}: invalid episode_index {episode_index!r}")
        if episode_index in records_by_index:
            raise ValueError(f"{dataset_dir}: duplicate episode_index {episode_index}")
        records_by_index[episode_index] = record

    expected_indices = set(range(total_episodes))
    actual_indices = set(records_by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise ValueError(f"{dataset_dir}: episode indices must be contiguous; missing={missing}, extra={extra}")

    episodes = []
    for episode_index in range(total_episodes):
        record = records_by_index[episode_index]
        declared_length = _positive_int(
            record.get("length"),
            field=f"episode {episode_index} length",
            context=dataset_dir,
        )
        parquet_path = parquet_path_for_episode(
            dataset_dir,
            data_path_template,
            chunks_size,
            episode_index,
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
        parquet_length = pq.read_metadata(parquet_path).num_rows
        if parquet_length != declared_length:
            raise ValueError(
                f"Episode {episode_index} length mismatch: episodes.jsonl={declared_length}, parquet={parquet_length}"
            )
        episodes.append(EpisodeSpec(episode_index, declared_length, parquet_path))

    total_frames = _positive_int(info.get("total_frames"), field="total_frames", context=dataset_dir)
    episode_frames = sum(episode.length for episode in episodes)
    if total_frames != episode_frames:
        raise ValueError(f"{dataset_dir}: info.json total_frames={total_frames}, episode total={episode_frames}")
    return DatasetSpec(dataset_dir, info, tuple(episodes))


def compute_stage_progress_gt(num_frames: int, mean_base_episode_length: float) -> np.ndarray:
    """Return float32 labels spanning [0, num_frames / base mean]."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if not np.isfinite(mean_base_episode_length) or mean_base_episode_length <= 0:
        raise ValueError(f"mean_base_episode_length must be positive, got {mean_base_episode_length}")
    endpoint = num_frames / mean_base_episode_length
    if num_frames == 1:
        return np.zeros(1, dtype=np.float32)
    return np.linspace(0.0, endpoint, num=num_frames, dtype=np.float32)


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


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name in table.column_names:
        table = table.drop_columns([name])
    return table.append_column(name, pa.array(values, type=pa.float32()))


def atomic_write_text(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content)
    os.replace(temporary_path, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary_path)
    os.replace(temporary_path, path)


def prepare_metadata_updates(
    dataset: DatasetSpec,
    labels_by_episode: dict[int, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build all metadata payloads before any dataset file is modified."""
    info = copy.deepcopy(dataset.info)
    features = info.setdefault("features", {})
    if not isinstance(features, dict):
        raise ValueError(f"{dataset.root}: info.json features must be an object")
    features[FEATURE_NAME] = {
        "dtype": "float32",
        "shape": [1],
        "names": [FEATURE_NAME],
    }

    stats_path = dataset.root / "meta/stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    all_labels = np.concatenate([labels_by_episode[episode.episode_index] for episode in dataset.episodes])
    stats[FEATURE_NAME] = compute_scalar_stats(all_labels)

    episodes_stats_path = dataset.root / "meta/episodes_stats.jsonl"
    if episodes_stats_path.exists():
        episodes_stats = read_jsonl(episodes_stats_path)
        records_by_index = {}
        for record in episodes_stats:
            episode_index = record.get("episode_index")
            if isinstance(episode_index, bool) or not isinstance(episode_index, int):
                raise ValueError(f"{episodes_stats_path}: invalid episode_index {episode_index!r}")
            if episode_index in records_by_index:
                raise ValueError(f"{episodes_stats_path}: duplicate episode_index {episode_index}")
            records_by_index[episode_index] = record
        expected_indices = {episode.episode_index for episode in dataset.episodes}
        if set(records_by_index) != expected_indices:
            missing = sorted(expected_indices - set(records_by_index))
            extra = sorted(set(records_by_index) - expected_indices)
            raise ValueError(f"{episodes_stats_path}: episode mismatch; missing={missing}, extra={extra}")
        episodes_stats = [records_by_index[episode.episode_index] for episode in dataset.episodes]
    else:
        episodes_stats = [{"episode_index": episode.episode_index, "stats": {}} for episode in dataset.episodes]

    for record in episodes_stats:
        episode_index = int(record["episode_index"])
        record_stats = record.setdefault("stats", {})
        if not isinstance(record_stats, dict):
            raise ValueError(f"{episodes_stats_path}: stats for episode {episode_index} must be an object")
        record_stats[FEATURE_NAME] = compute_scalar_stats(labels_by_episode[episode_index])
    return info, stats, episodes_stats


def label_stage_dataset(base_dir: Path, stage_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Validate, label, and synchronize metadata for one stage dataset."""
    base = load_and_validate_dataset(base_dir)
    stage = load_and_validate_dataset(stage_dir)
    mean_base_episode_length = sum(episode.length for episode in base.episodes) / len(base.episodes)
    labels_by_episode = {
        episode.episode_index: compute_stage_progress_gt(episode.length, mean_base_episode_length)
        for episode in stage.episodes
    }
    info, stats, episodes_stats = prepare_metadata_updates(stage, labels_by_episode)

    print(f"Base dataset: {base.root}")
    print(f"Stage dataset: {stage.root}")
    print(f"Mean base episode length: {mean_base_episode_length:.10f}")
    for episode in stage.episodes:
        endpoint = episode.length / mean_base_episode_length
        print(f"Episode {episode.episode_index:06d}: frames={episode.length}, endpoint={endpoint:.10f}")

    if dry_run:
        print("Dry-run complete; no files were modified.")
    else:
        for episode in stage.episodes:
            table = pq.read_table(episode.parquet_path)
            if table.num_rows != episode.length:
                raise RuntimeError(f"Episode changed after preflight: {episode.parquet_path}")
            table = replace_column(table, FEATURE_NAME, labels_by_episode[episode.episode_index])
            atomic_write_parquet(episode.parquet_path, table)

        meta_dir = stage.root / "meta"
        atomic_write_json(meta_dir / "info.json", info)
        atomic_write_json(meta_dir / "stats.json", stats)
        atomic_write_text(
            meta_dir / "episodes_stats.jsonl",
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in episodes_stats),
        )
        print(f"Updated {len(stage.episodes)} episodes ({sum(episode.length for episode in stage.episodes)} frames).")

    endpoints = [episode.length / mean_base_episode_length for episode in stage.episodes]
    return {
        "base_mean_episode_length": mean_base_episode_length,
        "total_episodes": len(stage.episodes),
        "total_frames": sum(episode.length for episode in stage.episodes),
        "minimum_endpoint": min(endpoints),
        "maximum_endpoint": max(endpoints),
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", type=parse_path_component, required=True)
    parser.add_argument("--stage-name", type=parse_path_component, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("ADVANTAGE_DATA_ROOT", PROJECT_ROOT / "data"))
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    task_root = data_root / args.task_name
    base_dir = task_root / "base/origin"
    stage_dir = task_root / args.stage_name / "origin"
    if base_dir == stage_dir:
        raise ValueError("stage-name must not resolve to the base dataset")
    label_stage_dataset(base_dir, stage_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
