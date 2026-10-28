import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import label_towel_fold_acbc as labeling


def _write_dataset(dataset_dir: Path, stage_advantages: list[list[list[float]]]) -> dict:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    (dataset_dir / "videos").mkdir()
    info = {
        "total_episodes": len(stage_advantages),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "stage_progress_gt": {"dtype": "float32", "shape": [1], "names": ["stage_progress_gt"]},
            "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    (meta_dir / "stats.json").write_text("{}")
    (meta_dir / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": ["fold towel"]}) + "\n"
            for index in range(len(stage_advantages))
        )
    )
    episode_stats = []
    for episode_index, stages in enumerate(stage_advantages):
        advantages = np.concatenate([np.asarray(values, dtype=np.float32) for values in stages])
        stage_index = np.concatenate(
            [np.full(len(values), stage, dtype=np.int64) for stage, values in enumerate(stages)]
        )
        num_frames = len(advantages)
        table = pa.table(
            {
                "frame_index": pa.array(range(num_frames), type=pa.int64()),
                "stage_progress_gt": pa.array(np.linspace(0, 1, num_frames), type=pa.float32()),
                "stage_index": pa.array(stage_index, type=pa.int64()),
                "relative_advantage": pa.array(advantages, type=pa.float32()),
                "absolute_value": pa.array(np.cumsum(advantages), type=pa.float32()),
                "absolute_advantage": pa.array(advantages, type=pa.float32()),
                "task_index": pa.array(np.zeros(num_frames, dtype=np.int64), type=pa.int64()),
            }
        )
        parquet_path = dataset_dir / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        episode_stats.append({"episode_index": episode_index, "stats": {}})
    (meta_dir / "episodes_stats.jsonl").write_text("".join(json.dumps(record) + "\n" for record in episode_stats))
    return info


def test_make_stage_indices_uses_half_open_boundaries():
    result = labeling.make_stage_indices(10, [2, 5, 8])
    np.testing.assert_array_equal(result, [0, 0, 1, 1, 1, 2, 2, 2, 3, 3])


def test_normalize_future_differences_and_direct_advantage():
    values = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    np.testing.assert_allclose(labeling.normalize_future_difference(values, 2), [0.2, 0.2, 0.2, 0.0])
    raw = np.asarray([0.2, 0.2, 0.1, 0.5], dtype=np.float32)
    np.testing.assert_allclose(labeling.normalize_direct_advantage(raw, 2), [0.2, 0.2, 0.2, 0.0])


def test_annotations_require_all_episodes_and_three_boundaries(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    info = _write_dataset(dataset_dir, [[[0], [0], [0], [0]], [[0], [0], [0], [0]]])
    annotations = {
        "0": {"subtask_completion_indices": [1, 2, 3], "segments": []},
        "1": {"subtask_completion_indices": [1, 2, 3], "segments": []},
    }
    annotation_path = dataset_dir / "augmentation_metadata.json"
    annotation_path.write_text(json.dumps(annotations))

    assert labeling.load_and_validate_annotations(annotation_path, dataset_dir, info) == {
        0: [1, 2, 3],
        1: [1, 2, 3],
    }

    annotation_path.write_text(json.dumps({"0": annotations["0"]}))
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        labeling.load_and_validate_annotations(annotation_path, dataset_dir, info)


def test_stage_thresholds_and_labels_are_computed_across_episodes(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    info = _write_dataset(
        dataset_dir,
        [
            [[0, 1], [10, 11], [20, 21], [30, 31]],
            [[2, 3], [12, 13], [22, 23], [32, 33]],
        ],
    )

    thresholds = labeling.compute_stage_thresholds(dataset_dir, info, top_percent=25.0)
    assert thresholds == {0: 2.25, 1: 12.25, 2: 22.25, 3: 32.25}
    summary = labeling.apply_acbc_labels(dataset_dir, info, thresholds)
    assert all(stage["total"] == 4 for stage in summary.values())
    assert all(stage["positive"] == 1 for stage in summary.values())

    episode_zero = pq.read_table(dataset_dir / "data/chunk-000/episode_000000.parquet")
    episode_one = pq.read_table(dataset_dir / "data/chunk-000/episode_000001.parquet")
    assert episode_zero["task_index"].to_pylist() == [0] * 8
    assert episode_one["task_index"].to_pylist() == [0, 1, 0, 1, 0, 1, 0, 1]


def test_prepare_derived_dataset_preserves_source_and_adds_manual_stage_index(tmp_path: Path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "derived"
    info = _write_dataset(source_dir, [[[0, 1], [2, 3], [4, 5], [6, 7]]])
    annotation_path = source_dir / "augmentation_metadata.json"
    annotation_path.write_text(json.dumps({"0": {"subtask_completion_indices": [1, 3, 7], "segments": []}}))
    source_parquet = source_dir / "data/chunk-000/episode_000000.parquet"
    source_stages_before = pq.read_table(source_parquet)["stage_index"].to_pylist()

    labeling.prepare_derived_dataset(
        source_dir,
        output_dir,
        annotation_path,
        annotations={0: [1, 3, 7]},
        info=info,
    )

    derived_stages = pq.read_table(output_dir / "data/chunk-000/episode_000000.parquet")["stage_index"].to_pylist()
    assert derived_stages == [0, 1, 1, 2, 2, 2, 2, 3]
    assert pq.read_table(source_parquet)["stage_index"].to_pylist() == source_stages_before
    assert (output_dir / "videos").is_symlink()
    assert json.loads((output_dir / "augmentation_metadata.json").read_text())["0"]["subtask_completion_indices"] == [
        1,
        3,
        7,
    ]


def test_update_metadata_registers_stage_and_advantage_fields(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    info = _write_dataset(dataset_dir, [[[0, 1], [2, 3], [4, 5], [6, 7]]])

    labeling.update_lerobot_metadata(dataset_dir, info)

    updated_info = json.loads((dataset_dir / "meta/info.json").read_text())
    assert updated_info["total_tasks"] == 2
    assert updated_info["features"]["stage_index"]["dtype"] == "int64"
    assert updated_info["features"]["absolute_advantage"]["dtype"] == "float32"
    tasks = [json.loads(line) for line in (dataset_dir / "meta/tasks.jsonl").read_text().splitlines()]
    assert tasks[1]["task"].endswith("Advantage: positive")
    episode = json.loads((dataset_dir / "meta/episodes.jsonl").read_text())
    assert episode["tasks"] == [tasks[0]["task"], tasks[1]["task"]]
