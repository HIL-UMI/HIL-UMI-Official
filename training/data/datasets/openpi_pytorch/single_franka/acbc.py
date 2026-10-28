# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""ACBC prompt conditioning and dataset validation for single-Franka SFT."""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ADVANTAGE_PROMPT = re.compile(r"^(?P<task>.+), Advantage: (?P<label>[^\r\n]+)$")
_REQUIRED_FEATURES = {"state", "actions", "image", "extra_view_image", "task_index"}


class AcbcDatasetError(ValueError):
    """Raised when a LeRobot dataset cannot preserve advantage ACBC semantics."""


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask:
    """Extract a prompt from the current LeRobot dataset task.

    This intentionally mirrors advantage's ``PromptFromLeRobotTask`` transform:
    ``task_index`` is authoritative and any pre-existing ``prompt`` value is
    overwritten with the exact string from ``meta/tasks.jsonl``.

    Attributes:
        tasks: Mapping from integer LeRobot task indices to prompt strings.
    """

    tasks: Mapping[int, str]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return ``data`` with its task-derived prompt attached."""
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(
                f"task_index={task_index} not found in task mapping: {self.tasks}"
            )

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class AcbcDatasetSummary:
    """Validated ACBC dataset metadata and observed label counts."""

    dataset_root: Path
    total_episodes: int
    total_frames: int
    video_files: int
    task_counts: dict[int, int]
    tasks: dict[int, str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise AcbcDatasetError(
            f"Failed to read JSON metadata {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AcbcDatasetError(
            f"Expected a JSON object in {path}, got {type(value).__name__}."
        )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AcbcDatasetError(
                        f"Expected a JSON object at {path}:{line_number}, "
                        f"got {type(value).__name__}."
                    )
                records.append(value)
    except AcbcDatasetError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise AcbcDatasetError(
            f"Failed to read JSONL metadata {path}: {error}"
        ) from error
    return records


def _require_inside_repo(dataset_root: Path, repo_root: Path) -> None:
    resolved_dataset = dataset_root.resolve()
    resolved_repo = repo_root.resolve()
    try:
        resolved_dataset.relative_to(resolved_repo)
    except ValueError as error:
        raise AcbcDatasetError(
            f"ACBC dataset must be stored inside training: {resolved_dataset} is outside {resolved_repo}."
        ) from error


def _reject_symlinks(dataset_root: Path) -> None:
    if dataset_root.is_symlink():
        raise AcbcDatasetError(
            f"ACBC dataset root must not be a symlink: {dataset_root}"
        )
    for dir_path, dir_names, file_names in os.walk(dataset_root, followlinks=False):
        parent = Path(dir_path)
        for name in [*dir_names, *file_names]:
            path = parent / name
            if path.is_symlink():
                raise AcbcDatasetError(
                    f"ACBC datasets must be self-contained; symlink found: {path} -> {os.readlink(path)}"
                )


def _load_tasks(tasks_path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for record in _read_jsonl(tasks_path):
        if "task_index" not in record or "task" not in record:
            raise AcbcDatasetError(
                f"Every record in {tasks_path} must contain task_index and task."
            )
        task_index = int(record["task_index"])
        prompt = record["task"]
        if not isinstance(prompt, str) or not prompt:
            raise AcbcDatasetError(
                f"Task {task_index} in {tasks_path} must have a non-empty string prompt."
            )
        if task_index in tasks:
            raise AcbcDatasetError(
                f"Duplicate task_index={task_index} in {tasks_path}."
            )
        tasks[task_index] = prompt
    if not tasks:
        raise AcbcDatasetError(f"No task mappings found in {tasks_path}.")
    return tasks


def _validate_advantage_prompts(
    tasks: Mapping[int, str], used_indices: set[int]
) -> None:
    prompts_by_task: dict[str, set[str]] = defaultdict(set)
    for task_index in sorted(used_indices):
        prompt = tasks[task_index]
        match = _ADVANTAGE_PROMPT.fullmatch(prompt)
        if match is None:
            raise AcbcDatasetError(
                f"task_index={task_index} is not an exact ACBC prompt of the form "
                f"'<task>, Advantage: <bin>': {prompt!r}"
            )
        prompts_by_task[match.group("task")].add(match.group("label"))

    for task, labels in prompts_by_task.items():
        if len(labels) < 2:
            raise AcbcDatasetError(
                f"ACBC task {task!r} has only one observed advantage bin: {sorted(labels)}."
            )


def _validate_videos(
    dataset_root: Path,
    info: Mapping[str, Any],
    episode_indices: set[int],
) -> int:
    features = info["features"]
    video_keys = [
        key for key, value in features.items() if value.get("dtype") == "video"
    ]
    for required_video_key in ("image", "extra_view_image"):
        if required_video_key not in video_keys:
            raise AcbcDatasetError(
                f"ACBC π0.5 dataset is missing video feature {required_video_key!r}."
            )

    video_template = info.get("video_path")
    chunks_size = int(info.get("chunks_size", 1000))
    if not isinstance(video_template, str) or not video_template:
        raise AcbcDatasetError(
            "meta/info.json must define a relative video_path template."
        )
    for episode_index in episode_indices:
        for video_key in video_keys:
            relative_path = Path(
                video_template.format(
                    episode_chunk=episode_index // chunks_size,
                    episode_index=episode_index,
                    video_key=video_key,
                )
            )
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise AcbcDatasetError(
                    f"Unsafe video path in metadata: {relative_path}"
                )
            video_path = dataset_root / relative_path
            if not video_path.is_file():
                raise AcbcDatasetError(
                    f"Missing video for episode {episode_index}, feature {video_key!r}: {video_path}"
                )

    return sum(1 for path in (dataset_root / "videos").rglob("*") if path.is_file())


def validate_acbc_dataset(
    dataset_root: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> AcbcDatasetSummary:
    """Validate a self-contained, advantage-labeled LeRobot dataset.

    Args:
        dataset_root: Local LeRobot v2.1 dataset directory.
        repo_root: When supplied, require the dataset to live inside this
            training checkout.

    Returns:
        Counts and exact task mappings observed during validation.

    Raises:
        AcbcDatasetError: If the dataset cannot provide advantage-compatible ACBC
            prompt conditioning or is not self-contained.
    """
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise AcbcDatasetError(f"ACBC dataset directory does not exist: {root}")
    if repo_root is not None:
        _require_inside_repo(root, Path(repo_root).expanduser())
    _reject_symlinks(root)

    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    for metadata_path in (info_path, tasks_path, episodes_path):
        if not metadata_path.is_file():
            raise AcbcDatasetError(
                f"Required LeRobot metadata is missing: {metadata_path}"
            )

    info = _read_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise AcbcDatasetError(f"meta/info.json has no features mapping: {info_path}")
    missing_features = sorted(_REQUIRED_FEATURES - features.keys())
    if missing_features:
        raise AcbcDatasetError(
            f"ACBC π0.5 dataset is missing required features: {missing_features}"
        )
    for vector_key in ("state", "actions"):
        if features[vector_key].get("shape") != [10]:
            raise AcbcDatasetError(
                f"Feature {vector_key!r} must have shape [10], got {features[vector_key].get('shape')!r}."
            )

    tasks = _load_tasks(tasks_path)
    expected_tasks = int(info.get("total_tasks", len(tasks)))
    if expected_tasks != len(tasks):
        raise AcbcDatasetError(
            f"meta/info.json reports {expected_tasks} tasks but {tasks_path} contains {len(tasks)}."
        )

    episode_records = _read_jsonl(episodes_path)
    try:
        episode_indices = {int(record["episode_index"]) for record in episode_records}
    except (KeyError, TypeError, ValueError) as error:
        raise AcbcDatasetError(
            f"Every record in {episodes_path} must contain an integer episode_index."
        ) from error
    if len(episode_indices) != len(episode_records):
        raise AcbcDatasetError(
            f"Duplicate episode_index values found in {episodes_path}."
        )

    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise AcbcDatasetError(f"No parquet episodes found under {root / 'data'}.")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise AcbcDatasetError(
            "pyarrow is required to validate ACBC task_index values. Install the training embodied environment."
        ) from error

    task_counts: Counter[int] = Counter()
    total_frames = 0
    parquet_episode_indices: set[int] = set()
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        schema_names = set(parquet_file.schema_arrow.names)
        missing_columns = sorted(
            {"state", "actions", "episode_index", "task_index"} - schema_names
        )
        if missing_columns:
            raise AcbcDatasetError(
                f"Parquet file {parquet_path} is missing required columns: {missing_columns}"
            )
        table = parquet_file.read(columns=["episode_index", "task_index"])
        task_column = table.column("task_index")
        if not pa.types.is_integer(task_column.type):
            raise AcbcDatasetError(
                f"task_index in {parquet_path} must be integer, got {task_column.type}."
            )
        if task_column.null_count:
            raise AcbcDatasetError(
                f"task_index in {parquet_path} contains null values."
            )
        episode_column = table.column("episode_index")
        if not pa.types.is_integer(episode_column.type) or episode_column.null_count:
            raise AcbcDatasetError(
                f"episode_index in {parquet_path} must contain non-null integers."
            )
        task_counts.update(int(value) for value in task_column.to_pylist())
        parquet_episode_indices.update(
            int(value) for value in episode_column.unique().to_pylist()
        )
        total_frames += table.num_rows

    unknown_indices = sorted(set(task_counts) - tasks.keys())
    if unknown_indices:
        raise AcbcDatasetError(
            f"Dataset uses task_index values missing from meta/tasks.jsonl: {unknown_indices}"
        )
    if parquet_episode_indices != episode_indices:
        raise AcbcDatasetError(
            "Episode indices differ between parquet data and meta/episodes.jsonl: "
            f"parquet={sorted(parquet_episode_indices)}, metadata={sorted(episode_indices)}"
        )
    _validate_advantage_prompts(tasks, set(task_counts))

    expected_frames = int(info.get("total_frames", total_frames))
    expected_episodes = int(info.get("total_episodes", len(episode_indices)))
    if expected_frames != total_frames:
        raise AcbcDatasetError(
            f"meta/info.json reports {expected_frames} frames but parquet data contains {total_frames}."
        )
    if expected_episodes != len(episode_indices):
        raise AcbcDatasetError(
            f"meta/info.json reports {expected_episodes} episodes but metadata contains {len(episode_indices)}."
        )

    video_files = _validate_videos(root, info, episode_indices)
    return AcbcDatasetSummary(
        dataset_root=root.resolve(),
        total_episodes=len(episode_indices),
        total_frames=total_frames,
        video_files=video_files,
        task_counts=dict(sorted(task_counts.items())),
        tasks=dict(sorted(tasks.items())),
    )
