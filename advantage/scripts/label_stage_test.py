import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import label_stage


def _write_dataset(
    dataset_dir: Path,
    episode_lengths: list[int],
    *,
    include_existing_labels: bool = False,
) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    info = {
        "total_episodes": len(episode_lengths),
        "total_frames": sum(episode_lengths),
        "chunks_size": 2,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
            "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    (meta_dir / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": episode_index, "length": length, "tasks": ["test"]}) + "\n"
            for episode_index, length in enumerate(episode_lengths)
        )
    )
    (meta_dir / "stats.json").write_text(json.dumps({"task_index": {"count": [sum(episode_lengths)]}}))
    (meta_dir / "episodes_stats.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": episode_index, "stats": {"task_index": {"count": [length]}}}) + "\n"
            for episode_index, length in enumerate(episode_lengths)
        )
    )

    for episode_index, length in enumerate(episode_lengths):
        columns = {
            "frame_index": pa.array(range(length), type=pa.int64()),
            "task_index": pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
        }
        if include_existing_labels:
            columns["stage_progress_gt"] = pa.array(np.full(length, -1.0), type=pa.float32())
        table = pa.table(columns)
        chunk_index = episode_index // 2
        parquet_path = dataset_dir / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_compute_stage_progress_gt_uses_scaled_endpoint() -> None:
    np.testing.assert_allclose(
        label_stage.compute_stage_progress_gt(5, 10.0),
        [0.0, 0.125, 0.25, 0.375, 0.5],
    )
    np.testing.assert_array_equal(label_stage.compute_stage_progress_gt(1, 10.0), [0.0])
    with pytest.raises(ValueError, match="num_frames must be positive"):
        label_stage.compute_stage_progress_gt(0, 10.0)


def test_label_stage_replaces_labels_and_updates_all_metadata(tmp_path: Path) -> None:
    base_dir = tmp_path / "data/test_task/base/origin"
    stage_dir = tmp_path / "data/test_task/stage1/origin"
    _write_dataset(base_dir, [4, 6])
    _write_dataset(stage_dir, [2, 5, 10], include_existing_labels=True)

    summary = label_stage.label_stage_dataset(base_dir, stage_dir)

    assert summary == {
        "base_mean_episode_length": 5.0,
        "total_episodes": 3,
        "total_frames": 17,
        "minimum_endpoint": 0.4,
        "maximum_endpoint": 2.0,
        "dry_run": False,
    }
    expected_endpoints = [0.4, 1.0, 2.0]
    for episode_index, endpoint in enumerate(expected_endpoints):
        chunk_index = episode_index // 2
        parquet_path = stage_dir / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet_path)
        assert table.column_names.count("stage_progress_gt") == 1
        values = table["stage_progress_gt"].to_numpy()
        assert values.dtype == np.float32
        assert values[0] == 0.0
        assert values[-1] == pytest.approx(endpoint)

    info = json.loads((stage_dir / "meta/info.json").read_text())
    assert info["features"]["stage_progress_gt"] == {
        "dtype": "float32",
        "names": ["stage_progress_gt"],
        "shape": [1],
    }
    stats = json.loads((stage_dir / "meta/stats.json").read_text())
    assert stats["task_index"]["count"] == [17]
    assert stats["stage_progress_gt"]["count"] == [17]
    assert stats["stage_progress_gt"]["max"] == pytest.approx([2.0])
    episode_stats = _read_jsonl(stage_dir / "meta/episodes_stats.jsonl")
    assert [record["stats"]["stage_progress_gt"]["count"] for record in episode_stats] == [[2], [5], [10]]
    assert episode_stats[-1]["stats"]["stage_progress_gt"]["max"] == pytest.approx([2.0])


def test_dry_run_does_not_modify_dataset(tmp_path: Path) -> None:
    base_dir = tmp_path / "data/test_task/base/origin"
    stage_dir = tmp_path / "data/test_task/stage1/origin"
    _write_dataset(base_dir, [4, 6])
    _write_dataset(stage_dir, [3])
    tracked_paths = sorted(path for path in stage_dir.rglob("*") if path.is_file())
    contents_before = {path: path.read_bytes() for path in tracked_paths}

    summary = label_stage.label_stage_dataset(base_dir, stage_dir, dry_run=True)

    assert summary["dry_run"] is True
    assert {path: path.read_bytes() for path in tracked_paths} == contents_before
    parquet_path = stage_dir / "data/chunk-000/episode_000000.parquet"
    assert "stage_progress_gt" not in pq.read_schema(parquet_path).names


def test_length_mismatch_is_rejected_before_writing(tmp_path: Path) -> None:
    base_dir = tmp_path / "data/test_task/base/origin"
    stage_dir = tmp_path / "data/test_task/stage1/origin"
    _write_dataset(base_dir, [4, 6])
    _write_dataset(stage_dir, [3])
    episodes_path = stage_dir / "meta/episodes.jsonl"
    episodes_path.write_text(json.dumps({"episode_index": 0, "length": 4, "tasks": ["test"]}) + "\n")
    parquet_path = stage_dir / "data/chunk-000/episode_000000.parquet"
    parquet_before = parquet_path.read_bytes()

    with pytest.raises(ValueError, match="length mismatch"):
        label_stage.label_stage_dataset(base_dir, stage_dir)

    assert parquet_path.read_bytes() == parquet_before
    assert "stage_progress_gt" not in pq.read_schema(parquet_path).names


@pytest.mark.parametrize("value", ["", ".", "..", "nested/task", "/absolute"])
def test_parse_path_component_rejects_unsafe_names(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Expected one directory name"):
        label_stage.parse_path_component(value)
