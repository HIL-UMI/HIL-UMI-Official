#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.
"""Convert raw dual-hand UMI data into single-Franka TCP rot6d data.

The source dataset is left untouched. The output follows the LeRobot v2.1
contract consumed by ``pi05_singlefranka_tcp_rot6d``:

* ``state``: left ``[xyz, rot6d, gripper_position]`` (10-D)
* ``actions``: left ``[xyz, rot6d, gripper_command]`` (10-D)
* ``image``: left-hand UMI camera
* ``extra_view_image``: fixed front camera

Example:

.. code-block:: bash

   python toolkits/lerobot/convert_umi_left_to_single_franka.py \
       --src dataset/stack_cubes_umi_1 \
       --dst dataset/stack_cubes_umi_1_franka \
       --prompt "stack the cubes"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

STATE_SOURCE_KEY = "observation.state"
ACTION_SOURCE_KEY = "action"
STATE_SOURCE_DIM = 47
ACTION_SOURCE_DIM = 30
OUTPUT_DIM = 10
STATE_SOURCE_INDICES = (9, 10, 11, 12, 13, 14, 15, 16, 17, 33)
ACTION_SOURCE_INDICES = (9, 10, 11, 12, 13, 14, 15, 16, 17, 21)
WRIST_VIDEO_SOURCE_KEY = "observation.images.cam_left"
FRONT_VIDEO_SOURCE_KEY = "observation.images.cam_front"
OUTPUT_VIDEO_KEYS = ("image", "extra_view_image-0", "extra_view_image")
QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))
_ADVANTAGE_PROMPT = re.compile(r"^(?P<task>.+), Advantage: (?P<label>[^\r\n]+)$")
_SOURCE_METADATA_FILES = ("info.json", "tasks.jsonl", "episodes.jsonl")
_ACBC_PROVENANCE_FILES = (
    "advantage_prediction.json",
    "mix_acbc_labeling.json",
    "mix_acbc_source.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty rows from a UTF-8 JSONL file."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write formatted UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_metadata_sha256(source_root: Path) -> str:
    """Return a stable digest for metadata that defines dataset membership."""
    digest = hashlib.sha256()
    for name in _SOURCE_METADATA_FILES:
        path = source_root / "meta" / name
        if not path.is_file():
            raise FileNotFoundError(f"missing source metadata: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_acbc_tasks(source_root: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Load and validate exact ACBC prompts from a raw UMI dataset."""
    rows = _read_jsonl(source_root / "meta" / "tasks.jsonl")
    tasks: dict[int, str] = {}
    labels_by_task: dict[str, set[str]] = {}
    for row in rows:
        if "task_index" not in row or "task" not in row:
            raise ValueError("ACBC tasks.jsonl rows require task_index and task")
        task_index = int(row["task_index"])
        prompt = row["task"]
        if task_index in tasks:
            raise ValueError(f"duplicate ACBC task_index={task_index}")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"ACBC task_index={task_index} has an invalid prompt")
        match = _ADVANTAGE_PROMPT.fullmatch(prompt)
        if match is None:
            raise ValueError(
                "ACBC prompts must have the exact form "
                f"'<task>, Advantage: <label>', got {prompt!r}"
            )
        tasks[task_index] = prompt
        labels_by_task.setdefault(match.group("task"), set()).add(match.group("label"))
    if not tasks:
        raise ValueError("ACBC tasks.jsonl has no task mappings")
    for task, labels in labels_by_task.items():
        if len(labels) < 2:
            raise ValueError(
                f"ACBC task {task!r} has only one advantage label: {sorted(labels)}"
            )
    return rows, tasks


def validate_raw_acbc_source(source_root: Path) -> None:
    """Validate a raw dual-hand UMI dataset before ACBC-preserving conversion."""
    source_root = source_root.resolve()
    source_meta = source_root / "meta"
    source_data = source_root / "data"
    source_videos = source_root / "videos"
    for required in (source_meta, source_data, source_videos):
        if not required.is_dir():
            raise FileNotFoundError(f"missing source directory: {required}")
    for dir_path, dir_names, file_names in os.walk(source_root, followlinks=False):
        parent = Path(dir_path)
        for name in [*dir_names, *file_names]:
            path = parent / name
            if path.is_symlink():
                raise ValueError(f"raw ACBC source contains a symlink: {path}")

    source_info = _read_json(source_meta / "info.json")
    features = source_info.get("features", {})
    expected_features = {
        STATE_SOURCE_KEY: [STATE_SOURCE_DIM],
        ACTION_SOURCE_KEY: [ACTION_SOURCE_DIM],
    }
    for key, shape in expected_features.items():
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(
                f"raw ACBC feature {key!r} must have shape {shape}, "
                f"got {features.get(key, {}).get('shape')!r}"
            )
    for key in (WRIST_VIDEO_SOURCE_KEY, FRONT_VIDEO_SOURCE_KEY):
        if features.get(key, {}).get("dtype") != "video":
            raise ValueError(f"raw ACBC source is missing video feature {key!r}")

    _, tasks = _load_acbc_tasks(source_root)
    episode_rows = _read_jsonl(source_meta / "episodes.jsonl")
    episode_indices = [int(row["episode_index"]) for row in episode_rows]
    if episode_indices != list(range(len(episode_rows))):
        raise ValueError("source episode indices must be contiguous from zero")

    task_counts: dict[int, int] = {}
    total_frames = 0
    used_episode_indices: set[int] = set()
    for episode_index in episode_indices:
        chunk = episode_index // int(source_info.get("chunks_size", 1000))
        parquet_path = (
            source_data / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing source parquet: {parquet_path}")
        table = pq.read_table(parquet_path, columns=["episode_index", "task_index"])
        parquet_episode_indices = {
            int(value) for value in table["episode_index"].unique().to_pylist()
        }
        used_episode_indices.update(parquet_episode_indices)
        for task_index in table["task_index"].to_pylist():
            task_index = int(task_index)
            task_counts[task_index] = task_counts.get(task_index, 0) + 1
        total_frames += table.num_rows

    unknown_task_indices = sorted(task_counts.keys() - tasks.keys())
    if unknown_task_indices:
        raise ValueError(
            "raw UMI parquet data uses task indices missing from tasks.jsonl: "
            f"{unknown_task_indices}"
        )
    if used_episode_indices != set(episode_indices):
        raise ValueError(
            "raw UMI parquet episode indices do not match episodes.jsonl: "
            f"{sorted(used_episode_indices)} != {episode_indices}"
        )

    labels_by_task: dict[str, set[str]] = {}
    for task_index in task_counts:
        match = _ADVANTAGE_PROMPT.fullmatch(tasks[task_index])
        assert match is not None
        labels_by_task.setdefault(match.group("task"), set()).add(match.group("label"))
    for task, labels in labels_by_task.items():
        if len(labels) < 2:
            raise ValueError(
                f"raw ACBC task {task!r} has only one observed advantage label: "
                f"{sorted(labels)}"
            )

    expected_frames = int(source_info.get("total_frames", total_frames))
    expected_episodes = int(source_info.get("total_episodes", len(episode_indices)))
    if total_frames != expected_frames:
        raise ValueError(
            f"raw ACBC metadata reports {expected_frames} frames, found {total_frames}"
        )
    if len(episode_indices) != expected_episodes:
        raise ValueError(
            "raw ACBC metadata reports "
            f"{expected_episodes} episodes, found {len(episode_indices)}"
        )

    video_template = source_info.get("video_path")
    if not isinstance(video_template, str) or not video_template:
        raise ValueError("raw ACBC meta/info.json has no video_path template")
    chunks_size = int(source_info.get("chunks_size", 1000))
    for episode_index in episode_indices:
        for video_key in (WRIST_VIDEO_SOURCE_KEY, FRONT_VIDEO_SOURCE_KEY):
            relative_video_path = Path(
                video_template.format(
                    episode_chunk=episode_index // chunks_size,
                    episode_index=episode_index,
                    video_key=video_key,
                )
            )
            if relative_video_path.is_absolute() or ".." in relative_video_path.parts:
                raise ValueError(f"unsafe raw ACBC video path: {relative_video_path}")
            video_path = source_root / relative_video_path
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"episode {episode_index} is missing raw video {video_key!r}: "
                    f"{video_path}"
                )


def _validate_reusable_acbc_cache(
    source_root: Path,
    destination_root: Path,
) -> None:
    """Reject a converted cache that does not match its current raw source."""
    mapping_path = destination_root / "meta" / "umi_left_franka_mapping.json"
    if not mapping_path.is_file():
        raise ValueError(
            f"existing conversion cache has no provenance mapping: {mapping_path}"
        )
    mapping = _read_json(mapping_path)
    if not mapping.get("preserve_acbc_tasks", False):
        raise ValueError(
            f"existing conversion cache was not created in ACBC mode: {destination_root}"
        )
    cached_source = Path(str(mapping.get("source_dataset", ""))).expanduser()
    if cached_source.resolve() != source_root.resolve():
        raise ValueError(
            "existing conversion cache belongs to a different source: "
            f"{cached_source} != {source_root}"
        )
    expected_digest = _source_metadata_sha256(source_root)
    if mapping.get("source_metadata_sha256") != expected_digest:
        raise ValueError(
            "existing conversion cache is stale because source metadata changed: "
            f"{destination_root}. Move it aside and rerun the conversion."
        )


def _fixed_size_list_to_numpy(
    column: pa.ChunkedArray, expected_dim: int, name: str
) -> np.ndarray:
    """Convert a fixed-size-list Arrow column into a float32 matrix."""
    list_size = getattr(column.type, "list_size", None)
    if list_size != expected_dim:
        raise ValueError(
            f"{name} must be fixed_size_list[{expected_dim}], got {column.type}"
        )
    combined = column.combine_chunks()
    flat = np.asarray(combined.values.to_numpy(zero_copy_only=False))
    return flat.reshape(-1, expected_dim).astype(np.float32, copy=False)


def _fixed_size_list(values: np.ndarray) -> pa.FixedSizeListArray:
    """Create ``FixedSizeList<float32>`` from a two-dimensional matrix."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {values.shape}")
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _numeric_stats(values: np.ndarray) -> dict[str, list[Any]]:
    """Calculate LeRobot v2.1 numeric statistics."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"statistics require a non-empty matrix, got {values.shape}")
    values64 = values.astype(np.float64)
    stats: dict[str, list[Any]] = {
        "min": values64.min(axis=0).tolist(),
        "max": values64.max(axis=0).tolist(),
        "mean": values64.mean(axis=0).tolist(),
        "std": values64.std(axis=0).tolist(),
        "count": [int(values64.shape[0])],
    }
    for name, quantile in QUANTILES:
        stats[name] = np.quantile(values64, quantile, axis=0).tolist()
    return stats


def _validate_rot6d(values: np.ndarray, label: str) -> None:
    """Reject non-finite or degenerate rot6d poses."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains NaN or Inf")
    first = values[:, 3:6].astype(np.float64)
    second = values[:, 6:9].astype(np.float64)
    first_norm = np.linalg.norm(first, axis=1)
    first_unit = first / first_norm[:, None]
    second_perp = second - np.sum(first_unit * second, axis=1)[:, None] * first_unit
    second_perp_norm = np.linalg.norm(second_perp, axis=1)
    if first_norm.min() < 1e-6 or second_perp_norm.min() < 1e-6:
        raise ValueError(
            f"{label} contains degenerate rot6d values: "
            f"min first norm={first_norm.min():.3e}, "
            f"min second perpendicular norm={second_perp_norm.min():.3e}"
        )


def _rewrite_episode(
    source_path: Path,
    destination_path: Path,
) -> tuple[dict[str, np.ndarray], int]:
    """Rewrite one raw UMI parquet episode."""
    table = pq.read_table(source_path)
    required = {
        STATE_SOURCE_KEY,
        ACTION_SOURCE_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"{source_path}: missing columns {sorted(missing)}")

    source_state = _fixed_size_list_to_numpy(
        table[STATE_SOURCE_KEY], STATE_SOURCE_DIM, STATE_SOURCE_KEY
    )
    source_actions = _fixed_size_list_to_numpy(
        table[ACTION_SOURCE_KEY], ACTION_SOURCE_DIM, ACTION_SOURCE_KEY
    )
    state = source_state[:, STATE_SOURCE_INDICES].copy()
    actions = source_actions[:, ACTION_SOURCE_INDICES].copy()
    _validate_rot6d(state, f"{source_path}: state")
    _validate_rot6d(actions, f"{source_path}: actions")
    if np.any((state[:, -1] < 0.0) | (state[:, -1] > 1.0)):
        raise ValueError(f"{source_path}: state gripper values fall outside [0, 1]")
    if np.any((actions[:, -1] < 0.0) | (actions[:, -1] > 1.0)):
        raise ValueError(f"{source_path}: action gripper values fall outside [0, 1]")

    arrays: list[pa.Array] = [_fixed_size_list(state), _fixed_size_list(actions)]
    names = ["state", "actions"]
    numeric: dict[str, np.ndarray] = {"state": state, "actions": actions}
    for name, arrow_type in (
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ):
        values = np.asarray(table[name].to_numpy(zero_copy_only=False))
        arrays.append(pa.array(values, type=arrow_type))
        names.append(name)
        numeric[name] = values

    output = pa.Table.from_arrays(arrays, names=names)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output, destination_path)
    return numeric, table.num_rows


def _copy_or_link(source: Path, destination: Path, *, link: bool = False) -> None:
    """Copy one file, optionally using a hard link when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _copy_videos(
    source_root: Path,
    destination_root: Path,
    episode_indices: list[int],
) -> None:
    """Copy left wrist/front videos and create the compatibility front alias."""
    source_video_root = source_root / "videos"
    destination_video_root = destination_root / "videos"
    for episode_index in tqdm.tqdm(episode_indices, desc="copy videos"):
        chunk = episode_index // 1000
        filename = f"episode_{episode_index:06d}.mp4"
        source_chunk = source_video_root / f"chunk-{chunk:03d}"
        destination_chunk = destination_video_root / f"chunk-{chunk:03d}"
        wrist_source = source_chunk / WRIST_VIDEO_SOURCE_KEY / filename
        front_source = source_chunk / FRONT_VIDEO_SOURCE_KEY / filename
        if not wrist_source.is_file() or not front_source.is_file():
            raise FileNotFoundError(
                f"episode {episode_index}: missing wrist/front video "
                f"({wrist_source}, {front_source})"
            )
        _copy_or_link(wrist_source, destination_chunk / "image" / filename)
        front_destination = destination_chunk / "extra_view_image" / filename
        _copy_or_link(front_source, front_destination)
        _copy_or_link(
            front_destination,
            destination_chunk / "extra_view_image-0" / filename,
            link=True,
        )


def _video_feature(source_feature: dict[str, Any]) -> dict[str, Any]:
    """Create output video metadata with the actual H.264 codec."""
    feature = json.loads(json.dumps(source_feature))
    feature["info"]["video.codec"] = "h264"
    return feature


def _build_info(
    source_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_tasks: int = 1,
) -> dict[str, Any]:
    """Build the single-Franka ``meta/info.json`` object."""
    features = source_info["features"]
    wrist_feature = _video_feature(features[WRIST_VIDEO_SOURCE_KEY])
    front_feature = _video_feature(features[FRONT_VIDEO_SOURCE_KEY])
    return {
        "codebase_version": "v2.1",
        "robot_type": "franka_umi_left_tcp_rot6d",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": 3,
        "total_chunks": (total_episodes + 999) // 1000,
        "chunks_size": 1000,
        "data_files_size_in_mb": source_info.get("data_files_size_in_mb", 100),
        "video_files_size_in_mb": source_info.get("video_files_size_in_mb", 200),
        "fps": int(source_info["fps"]),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "state": {
                "dtype": "float32",
                "shape": [OUTPUT_DIM],
                "names": [
                    "x",
                    "y",
                    "z",
                    "rot6d_0",
                    "rot6d_1",
                    "rot6d_2",
                    "rot6d_3",
                    "rot6d_4",
                    "rot6d_5",
                    "gripper_pos",
                ],
            },
            "actions": {
                "dtype": "float32",
                "shape": [OUTPUT_DIM],
                "names": [
                    "x",
                    "y",
                    "z",
                    "rot6d_0",
                    "rot6d_1",
                    "rot6d_2",
                    "rot6d_3",
                    "rot6d_4",
                    "rot6d_5",
                    "gripper_cmd_pos_pct",
                ],
            },
            "image": wrist_feature,
            "extra_view_image-0": front_feature,
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": ["timestamp"],
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["frame_index"],
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["episode_index"],
            },
            "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": ["task_index"],
            },
            "extra_view_image": front_feature,
        },
    }


def _video_stats(
    source_stats: dict[str, Any], source_key: str, context: str
) -> dict[str, Any]:
    """Fetch copied-video statistics from source metadata."""
    try:
        return source_stats[source_key]
    except KeyError as exc:
        raise ValueError(f"{context}: missing video stats for {source_key}") from exc


def _build_episode_stats(
    source_root: Path,
    episode_numeric: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Build output ``episodes_stats.jsonl`` rows."""
    source_rows = {
        row["episode_index"]: row
        for row in _read_jsonl(source_root / "meta" / "episodes_stats.jsonl")
    }
    output: list[dict[str, Any]] = []
    for episode_index in sorted(episode_numeric):
        if episode_index not in source_rows:
            raise ValueError(f"missing source episode stats for {episode_index}")
        source_stats = source_rows[episode_index]["stats"]
        stats = {
            key: _numeric_stats(value)
            for key, value in episode_numeric[episode_index].items()
        }
        stats["image"] = _video_stats(
            source_stats, WRIST_VIDEO_SOURCE_KEY, f"episode {episode_index}"
        )
        front_stats = _video_stats(
            source_stats, FRONT_VIDEO_SOURCE_KEY, f"episode {episode_index}"
        )
        stats["extra_view_image-0"] = front_stats
        stats["extra_view_image"] = front_stats
        output.append({"episode_index": episode_index, "stats": stats})
    return output


def _build_global_stats(
    source_root: Path,
    global_numeric: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    """Build output ``meta/stats.json``."""
    source_stats = _read_json(source_root / "meta" / "stats.json")
    stats = {
        key: _numeric_stats(np.concatenate(values, axis=0))
        for key, values in global_numeric.items()
    }
    stats["image"] = _video_stats(source_stats, WRIST_VIDEO_SOURCE_KEY, "global")
    front_stats = _video_stats(source_stats, FRONT_VIDEO_SOURCE_KEY, "global")
    stats["extra_view_image-0"] = front_stats
    stats["extra_view_image"] = front_stats
    return stats


def convert(
    source_root: Path,
    destination_root: Path,
    prompt: str,
    *,
    preserve_acbc_tasks: bool = False,
    reuse_existing: bool = False,
) -> None:
    """Convert one raw UMI dataset into the single-Franka training format."""
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if source_root == destination_root:
        raise ValueError("source and destination must be different directories")
    if destination_root.exists():
        if reuse_existing and preserve_acbc_tasks:
            _validate_reusable_acbc_cache(source_root, destination_root)
            print(f"Reusing current ACBC conversion cache: {destination_root}")
            return
        raise FileExistsError(f"destination already exists: {destination_root}")
    source_meta = source_root / "meta"
    source_data = source_root / "data"
    source_videos = source_root / "videos"
    for required in (source_meta, source_data, source_videos):
        if not required.is_dir():
            raise FileNotFoundError(f"missing source directory: {required}")

    source_info = _read_json(source_meta / "info.json")
    if int(source_info.get("fps", -1)) != 30:
        raise ValueError(f"expected 30 FPS source data, got {source_info.get('fps')}")
    episode_rows = _read_jsonl(source_meta / "episodes.jsonl")
    source_metadata_sha256 = _source_metadata_sha256(source_root)
    if preserve_acbc_tasks:
        validate_raw_acbc_source(source_root)
        output_task_rows, acbc_tasks = _load_acbc_tasks(source_root)
    else:
        output_task_rows = [{"task_index": 0, "task": prompt}]
        acbc_tasks = None
    episode_indices = [int(row["episode_index"]) for row in episode_rows]
    if episode_indices != list(range(len(episode_rows))):
        raise ValueError("source episode indices must be contiguous from zero")

    temporary_root = destination_root.with_name(destination_root.name + ".tmp")
    if temporary_root.exists():
        raise FileExistsError(
            f"temporary destination already exists: {temporary_root}; "
            "remove it after checking that no conversion is active"
        )

    episode_numeric: dict[int, dict[str, np.ndarray]] = {}
    global_numeric: dict[str, list[np.ndarray]] = {}
    total_frames = 0
    for episode_index in tqdm.tqdm(episode_indices, desc="convert parquet"):
        chunk = episode_index // 1000
        filename = f"episode_{episode_index:06d}.parquet"
        source_path = source_data / f"chunk-{chunk:03d}" / filename
        destination_path = temporary_root / "data" / f"chunk-{chunk:03d}" / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source parquet: {source_path}")
        numeric, length = _rewrite_episode(source_path, destination_path)
        if length != int(episode_rows[episode_index]["length"]):
            raise ValueError(
                f"episode {episode_index}: parquet length {length} does not match "
                f"metadata length {episode_rows[episode_index]['length']}"
            )
        episode_numeric[episode_index] = numeric
        for key, value in numeric.items():
            global_numeric.setdefault(key, []).append(value)
        total_frames += length

    if acbc_tasks is not None:
        used_task_indices = {
            int(task_index)
            for values in global_numeric["task_index"]
            for task_index in np.asarray(values).reshape(-1)
        }
        unknown_task_indices = sorted(used_task_indices - acbc_tasks.keys())
        if unknown_task_indices:
            raise ValueError(
                "raw UMI parquet data uses task indices missing from "
                f"tasks.jsonl: {unknown_task_indices}"
            )

    _copy_videos(source_root, temporary_root, episode_indices)
    info = _build_info(
        source_info,
        len(episode_indices),
        total_frames,
        total_tasks=len(output_task_rows),
    )
    _write_json(temporary_root / "meta" / "info.json", info)
    _write_jsonl(
        temporary_root / "meta" / "tasks.jsonl",
        output_task_rows,
    )
    _write_jsonl(
        temporary_root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": row["episode_index"],
                "tasks": (row.get("tasks", []) if preserve_acbc_tasks else [prompt]),
                "length": row["length"],
            }
            for row in episode_rows
        ],
    )
    _write_jsonl(
        temporary_root / "meta" / "episodes_stats.jsonl",
        _build_episode_stats(source_root, episode_numeric),
    )
    _write_json(
        temporary_root / "meta" / "stats.json",
        _build_global_stats(source_root, global_numeric),
    )
    if preserve_acbc_tasks:
        for name in _ACBC_PROVENANCE_FILES:
            source_path = source_meta / name
            if source_path.is_file():
                _copy_or_link(source_path, temporary_root / "meta" / name)
    _write_json(
        temporary_root / "meta" / "umi_left_franka_mapping.json",
        {
            "source_dataset": str(source_root),
            "prompt": None if preserve_acbc_tasks else prompt,
            "preserve_acbc_tasks": preserve_acbc_tasks,
            "source_metadata_sha256": source_metadata_sha256,
            "state_source_indices": list(STATE_SOURCE_INDICES),
            "action_source_indices": list(ACTION_SOURCE_INDICES),
            "state_layout": ("[xyz(3), rot6d(6), gripper_pos(1)] from UMI left hand"),
            "action_layout": (
                "[xyz(3), rot6d(6), gripper_cmd_pos_pct(1)] from UMI left hand"
            ),
            "video_mapping": {
                "image": WRIST_VIDEO_SOURCE_KEY,
                "extra_view_image-0": FRONT_VIDEO_SOURCE_KEY,
                "extra_view_image": FRONT_VIDEO_SOURCE_KEY,
            },
            "deployment_note": (
                "Training converts absolute poses to body-frame SE(3) deltas. "
                "Before hardware rollout, verify the UMI policy-tool axes against "
                "the Franka TCP and configure franka_tcp_to_policy_tool."
            ),
            "metadata_note": (
                "Parquet schema metadata is intentionally cleared; "
                "meta/info.json is authoritative."
            ),
        },
    )

    temporary_root.rename(destination_root)
    print(
        f"Converted {len(episode_indices)} episodes / {total_frames} frames:\n"
        f"  source:      {source_root}\n"
        f"  destination: {destination_root}"
    )


def main() -> int:
    """Parse command-line arguments and run conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Raw UMI root.")
    parser.add_argument("--dst", type=Path, help="New single-Franka output root.")
    parser.add_argument(
        "--prompt",
        default="stack the cubes",
        help=(
            "Task prompt written to tasks/episodes metadata. Ignored when "
            "--preserve-acbc-tasks is set."
        ),
    )
    parser.add_argument(
        "--preserve-acbc-tasks",
        action="store_true",
        help=(
            "Preserve and validate the source task_index, tasks.jsonl, and "
            "per-episode Advantage prompts."
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse an existing ACBC conversion only when its source path and "
            "metadata fingerprint still match."
        ),
    )
    parser.add_argument(
        "--validate-acbc-source-only",
        action="store_true",
        help="Validate a raw ACBC UMI source without writing converted data.",
    )
    args = parser.parse_args()
    try:
        if args.validate_acbc_source_only:
            validate_raw_acbc_source(args.src)
            print(f"Raw ACBC source validation: PASS ({args.src.resolve()})")
            return 0
        if args.dst is None:
            parser.error("--dst is required unless --validate-acbc-source-only is set")
        convert(
            args.src,
            args.dst,
            args.prompt,
            preserve_acbc_tasks=args.preserve_acbc_tasks,
            reuse_existing=args.reuse_existing,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
