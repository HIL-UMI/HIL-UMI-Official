# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for advantage-compatible ACBC prompt conditioning and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from training.data.datasets.openpi_pytorch.single_franka.acbc import (
    AcbcDatasetError,
    PromptFromLeRobotTask,
    validate_acbc_dataset,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def _make_acbc_dataset(
    root: Path,
    *,
    prompts: tuple[str, str] = (
        "fold the towel, Advantage: negative",
        "fold the towel, Advantage: positive",
    ),
) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "chunks_size": 1000,
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 2,
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "state": {"dtype": "float32", "shape": [10]},
            "actions": {"dtype": "float32", "shape": [10]},
            "image": {"dtype": "video", "shape": [2, 2, 3]},
            "extra_view_image": {"dtype": "video", "shape": [2, 2, 3]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        meta_dir / "episodes.jsonl", [{"episode_index": 0, "tasks": list(prompts)}]
    )
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {"task_index": 0, "task": prompts[0]},
            {"task_index": 1, "task": prompts[1]},
        ],
    )

    table = pa.table(
        {
            "state": pa.array([[0.0] * 10, [1.0] * 10]),
            "actions": pa.array([[0.0] * 10, [1.0] * 10]),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "task_index": pa.array([0, 1], type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    for video_key in ("image", "extra_view_image"):
        video_dir = root / "videos" / "chunk-000" / video_key
        video_dir.mkdir(parents=True)
        (video_dir / "episode_000000.mp4").write_bytes(b"test-video")
    return root


def test_prompt_from_task_overwrites_static_prompt_exactly():
    transform = PromptFromLeRobotTask(
        {
            0: "fold the towel, Advantage: negative",
            1: "fold the towel, Advantage: positive",
        }
    )

    result = transform({"task_index": 1, "prompt": "static prompt"})

    assert result["prompt"] == "fold the towel, Advantage: positive"


def test_prompt_from_task_rejects_missing_or_unknown_index():
    transform = PromptFromLeRobotTask({1: "task, Advantage: positive"})

    with pytest.raises(ValueError, match="Cannot extract prompt"):
        transform({})
    with pytest.raises(ValueError, match="task_index=2 not found"):
        transform({"task_index": 2})


def test_validator_accepts_binary_acbc_dataset(tmp_path):
    dataset_root = _make_acbc_dataset(tmp_path / "dataset")

    summary = validate_acbc_dataset(dataset_root, repo_root=tmp_path)

    assert summary.total_episodes == 1
    assert summary.total_frames == 2
    assert summary.video_files == 2
    assert summary.task_counts == {0: 1, 1: 1}


def test_validator_accepts_numeric_advantage_slices(tmp_path):
    dataset_root = _make_acbc_dataset(
        tmp_path / "dataset",
        prompts=("fold the towel, Advantage: 0", "fold the towel, Advantage: 1"),
    )

    summary = validate_acbc_dataset(dataset_root, repo_root=tmp_path)

    assert summary.tasks[0].endswith("Advantage: 0")
    assert summary.tasks[1].endswith("Advantage: 1")


def test_validator_rejects_single_observed_advantage_bin(tmp_path):
    dataset_root = _make_acbc_dataset(tmp_path / "dataset")
    parquet_path = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(parquet_path).set_column(
        3,
        "task_index",
        pytest.importorskip("pyarrow").array([0, 0]),
    )
    pq.write_table(table, parquet_path)

    with pytest.raises(AcbcDatasetError, match="only one observed advantage bin"):
        validate_acbc_dataset(dataset_root, repo_root=tmp_path)


def test_validator_rejects_symlinked_dataset_content(tmp_path):
    dataset_root = _make_acbc_dataset(tmp_path / "dataset")
    target = dataset_root / "videos" / "chunk-000" / "image" / "episode_000000.mp4"
    link = dataset_root / "videos" / "linked-video.mp4"
    os.symlink(target, link)

    with pytest.raises(AcbcDatasetError, match="symlink found"):
        validate_acbc_dataset(dataset_root, repo_root=tmp_path)
