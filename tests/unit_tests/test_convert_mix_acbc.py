# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Tests for ACBC-preserving raw UMI conversion used by Mix ACBC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from toolkits.lerobot.convert_umi_left_to_single_franka import (
    convert,
    validate_raw_acbc_source,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _make_raw_acbc_dataset(root: Path) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    video_feature = {
        "dtype": "video",
        "shape": [2, 2, 3],
        "info": {"video.codec": "h264"},
    }
    _write_json(
        meta_dir / "info.json",
        {
            "codebase_version": "v2.1",
            "chunks_size": 1000,
            "fps": 30,
            "total_episodes": 1,
            "total_frames": 2,
            "total_tasks": 2,
            "data_path": (
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "features": {
                "observation.state": {"dtype": "float32", "shape": [47]},
                "action": {"dtype": "float32", "shape": [30]},
                "observation.images.cam_left": video_feature,
                "observation.images.cam_front": video_feature,
                "episode_index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        },
    )
    prompts = [
        "fold the towel, Advantage: negative",
        "fold the towel, Advantage: positive",
    ]
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {"task_index": 0, "task": prompts[0]},
            {"task_index": 1, "task": prompts[1]},
        ],
    )
    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [{"episode_index": 0, "length": 2, "tasks": prompts}],
    )

    states = np.zeros((2, 47), dtype=np.float32)
    actions = np.zeros((2, 30), dtype=np.float32)
    rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    states[:, 12:18] = rot6d
    states[:, 33] = 0.5
    actions[:, 12:18] = rot6d
    actions[:, 21] = 0.5
    table = pa.table(
        {
            "observation.state": pa.array(
                states.tolist(), type=pa.list_(pa.float32(), 47)
            ),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 30)),
            "timestamp": pa.array([0.0, 1 / 30], type=pa.float32()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "index": pa.array([0, 1], type=pa.int64()),
            "task_index": pa.array([0, 1], type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")

    video_stats = {"min": [0], "max": [255]}
    _write_jsonl(
        meta_dir / "episodes_stats.jsonl",
        [
            {
                "episode_index": 0,
                "stats": {
                    "observation.images.cam_left": video_stats,
                    "observation.images.cam_front": video_stats,
                },
            }
        ],
    )
    _write_json(
        meta_dir / "stats.json",
        {
            "observation.images.cam_left": video_stats,
            "observation.images.cam_front": video_stats,
        },
    )
    _write_json(meta_dir / "mix_acbc_labeling.json", {"source": "unit-test"})
    for video_key in (
        "observation.images.cam_left",
        "observation.images.cam_front",
    ):
        video_dir = root / "videos" / "chunk-000" / video_key
        video_dir.mkdir(parents=True)
        (video_dir / "episode_000000.mp4").write_bytes(b"test-video")
    return root


def test_acbc_conversion_preserves_task_prompts_and_reuses_current_cache(tmp_path):
    source = _make_raw_acbc_dataset(tmp_path / "raw")
    destination = tmp_path / "raw_single_franka_acbc"

    validate_raw_acbc_source(source)
    convert(
        source,
        destination,
        "ignored static prompt",
        preserve_acbc_tasks=True,
    )

    info = json.loads((destination / "meta" / "info.json").read_text())
    tasks = (destination / "meta" / "tasks.jsonl").read_text()
    episodes = (destination / "meta" / "episodes.jsonl").read_text()
    mapping = json.loads(
        (destination / "meta" / "umi_left_franka_mapping.json").read_text()
    )
    table = pytest.importorskip("pyarrow.parquet").read_table(
        destination / "data" / "chunk-000" / "episode_000000.parquet"
    )

    assert info["total_tasks"] == 2
    assert info["features"]["state"]["shape"] == [10]
    assert info["features"]["actions"]["shape"] == [10]
    assert "Advantage: negative" in tasks
    assert "Advantage: positive" in tasks
    assert "Advantage: negative" in episodes
    assert "Advantage: positive" in episodes
    assert table["task_index"].to_pylist() == [0, 1]
    assert mapping["preserve_acbc_tasks"] is True
    assert len(mapping["source_metadata_sha256"]) == 64
    assert (destination / "meta" / "mix_acbc_labeling.json").is_file()

    convert(
        source,
        destination,
        "ignored static prompt",
        preserve_acbc_tasks=True,
        reuse_existing=True,
    )


def test_acbc_conversion_rejects_stale_cache(tmp_path):
    source = _make_raw_acbc_dataset(tmp_path / "raw")
    destination = tmp_path / "raw_single_franka_acbc"
    convert(source, destination, "ignored", preserve_acbc_tasks=True)
    tasks_path = source / "meta" / "tasks.jsonl"
    tasks_path.write_text(tasks_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cache is stale"):
        convert(
            source,
            destination,
            "ignored",
            preserve_acbc_tasks=True,
            reuse_existing=True,
        )


def test_raw_acbc_validation_requires_both_observed_labels(tmp_path):
    source = _make_raw_acbc_dataset(tmp_path / "raw")
    parquet_path = source / "data" / "chunk-000" / "episode_000000.parquet"
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    table = pq.read_table(parquet_path).set_column(
        6,
        "task_index",
        pa.array([0, 0], type=pa.int64()),
    )
    pq.write_table(table, parquet_path)

    with pytest.raises(ValueError, match="only one observed advantage label"):
        validate_raw_acbc_source(source)
