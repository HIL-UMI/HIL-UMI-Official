# Modified for HIL-UMI.
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hil_umi_advantage.training.advantage_schema import BASE_UMI_SCHEMA
from hil_umi_advantage.training.advantage_schema import STAGE_UMI_SCHEMA
from scripts import label_towel_mix_acbc as labeling


def write_test_dataset(
    dataset_dir: Path,
    schema,
    episode_advantages: list[list[float]],
    *,
    with_annotations: bool,
) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": len(episode_advantages),
        "total_frames": sum(len(values) for values in episode_advantages),
        "total_tasks": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            schema.front_key: {"dtype": "video"},
            schema.wrist_key: {"dtype": "video"},
            schema.state_key: {"dtype": "float32"},
            schema.action_key: {"dtype": "float32"},
            "stage_progress_gt": {"dtype": "float32"},
            "task_index": {"dtype": "int64"},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    (meta_dir / "stats.json").write_text("{}")
    (meta_dir / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "fold towel"}) + "\n")
    episodes = []
    episode_stats = []
    annotations = {}
    global_index = 0
    for episode_index, advantage_values in enumerate(episode_advantages):
        advantages = np.asarray(advantage_values, dtype=np.float32)
        num_frames = len(advantages)
        table = pa.table(
            {
                schema.state_key: pa.array(np.zeros((num_frames, 10), dtype=np.float32).tolist()),
                schema.action_key: pa.array(np.zeros((num_frames, 10), dtype=np.float32).tolist()),
                "timestamp": pa.array(np.arange(num_frames, dtype=np.float32) / 30),
                "frame_index": pa.array(np.arange(num_frames), type=pa.int64()),
                "episode_index": pa.array(np.full(num_frames, episode_index), type=pa.int64()),
                "index": pa.array(np.arange(global_index, global_index + num_frames), type=pa.int64()),
                "task_index": pa.array(np.zeros(num_frames), type=pa.int64()),
                "stage_progress_gt": pa.array(np.linspace(0, 1, num_frames), type=pa.float32()),
                "absolute_advantage": pa.array(advantages, type=pa.float32()),
                "absolute_value": pa.array(np.cumsum(advantages), type=pa.float32()),
                "relative_advantage": pa.array(advantages, type=pa.float32()),
            }
        )
        parquet_path = dataset_dir / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        for video_key in schema.video_keys:
            video_path = dataset_dir / f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(f"video-{episode_index}-{video_key}".encode())
        episodes.append({"episode_index": episode_index, "length": num_frames, "tasks": ["fold towel"]})
        episode_stats.append({"episode_index": episode_index, "stats": {}})
        if with_annotations:
            if num_frames < 8:
                raise ValueError("Synthetic annotated episodes require at least eight frames")
            annotations[str(episode_index)] = {
                "segments": [],
                "subtask_completion_indices": [2, 4, 6],
            }
        global_index += num_frames
    (meta_dir / "episodes.jsonl").write_text("".join(json.dumps(record) + "\n" for record in episodes))
    (meta_dir / "episodes_stats.jsonl").write_text("".join(json.dumps(record) + "\n" for record in episode_stats))
    if with_annotations:
        (dataset_dir / "augmentation_metadata.json").write_text(json.dumps(annotations))


def _job(
    name: str,
    dataset_dir: Path,
    schema,
    grouping: str,
    *,
    group_count: int = 1,
    annotations=None,
) -> labeling.DatasetJob:
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    return labeling.DatasetJob(
        name=name,
        source=dataset_dir,
        output=dataset_dir,
        grouping=grouping,
        group_count=group_count,
        schema=schema,
        info=info,
        annotations=annotations,
    )


def test_checkpoint_metadata_requires_the_exact_mix_sources(tmp_path: Path):
    base = (tmp_path / "base").resolve()
    stage1 = (tmp_path / "stage1").resolve()
    checkpoint = tmp_path / "10000"
    metadata = {
        "global_step": 10000,
        "config": {
            "name": labeling.DEFAULT_CONFIG_NAME,
            "exp_name": "towel_base_stage1_mix",
            "advantage_dataset_group_a": (str(base),),
            "advantage_dataset_group_b": (str(stage1),),
        },
    }

    context = labeling.validate_mix_checkpoint_metadata(
        metadata,
        checkpoint,
        base,
        stage1,
        labeling.DEFAULT_CONFIG_NAME,
    )
    assert context["checkpoint_step"] == 10000
    assert context["training_group_a"] == [str(base)]

    metadata["config"]["advantage_dataset_group_b"] = (str(tmp_path / "wrong"),)
    with pytest.raises(ValueError, match="was not trained with the requested mix sources"):
        labeling.validate_mix_checkpoint_metadata(
            metadata,
            checkpoint,
            base,
            stage1,
            labeling.DEFAULT_CONFIG_NAME,
        )


def test_build_jobs_detects_both_schemas_and_four_base_subtasks(tmp_path: Path):
    base = tmp_path / "base"
    stage1 = tmp_path / "stage1"
    write_test_dataset(base, BASE_UMI_SCHEMA, [list(range(8))], with_annotations=True)
    write_test_dataset(stage1, STAGE_UMI_SCHEMA, [list(range(5))], with_annotations=False)

    base_job, stage1_job = labeling.build_jobs(base, stage1, tmp_path / "output")

    assert base_job.schema is BASE_UMI_SCHEMA
    assert base_job.grouping == "manual_subtasks"
    assert base_job.group_count == 4
    assert base_job.annotations == {0: [2, 4, 6]}
    assert stage1_job.schema is STAGE_UMI_SCHEMA
    assert stage1_job.grouping == "whole_dataset"
    stage_episode = labeling.load_episode_data(
        stage1 / "data/chunk-000/episode_000000.parquet",
        STAGE_UMI_SCHEMA,
    )
    assert stage_episode.states.shape == (5, 10)
    assert stage_episode.actions.shape == (5, 10)


def test_output_copy_materializes_videos_and_preserves_source(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "labeled" / "base"
    write_test_dataset(source, BASE_UMI_SCHEMA, [list(range(8))], with_annotations=True)
    info = json.loads((source / "meta/info.json").read_text())
    annotations = {0: [2, 4, 6]}
    job = labeling.DatasetJob(
        name="base",
        source=source.resolve(),
        output=output.resolve(),
        grouping="manual_subtasks",
        group_count=4,
        schema=BASE_UMI_SCHEMA,
        info=info,
        annotations=annotations,
    )
    context = {"checkpoint_dir": "/checkpoint/10000", "checkpoint_step": 10000}

    labeling.prepare_output_dataset(job, context)
    labeling.add_base_stage_indices(job)

    output_video = output / "videos/chunk-000/extra_view_image/episode_000000.mp4"
    assert output_video.is_file()
    assert not output_video.is_symlink()
    output_table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert output_table["stage_index"].to_pylist() == [0, 0, 1, 1, 2, 2, 3, 3]
    source_table = pq.read_table(source / "data/chunk-000/episode_000000.parquet")
    assert "stage_index" not in source_table.column_names


def test_existing_legacy_four_subtask_marker_is_compatible(tmp_path: Path):
    output = tmp_path / "output"
    marker_path = output / labeling.OUTPUT_MARKER
    marker_path.parent.mkdir(parents=True)
    info = {"total_episodes": 1, "total_frames": 8}
    job = labeling.DatasetJob(
        name="base",
        source=(tmp_path / "source").resolve(),
        output=output.resolve(),
        grouping="manual_subtasks",
        group_count=4,
        schema=BASE_UMI_SCHEMA,
        info=info,
    )
    context = {"checkpoint_dir": "/checkpoint/10000", "checkpoint_step": 10000}
    marker = labeling.expected_output_marker(job, context)
    marker["grouping"] = "four_manual_subtasks"
    marker_path.write_text(json.dumps(marker))

    labeling.validate_existing_output(job, context)


def test_completed_output_rejects_changed_top_percent(tmp_path: Path):
    dataset = tmp_path / "dataset"
    write_test_dataset(dataset, STAGE_UMI_SCHEMA, [list(range(8))], with_annotations=False)
    job = _job("stage1", dataset, STAGE_UMI_SCHEMA, "whole_dataset")
    context = {
        "checkpoint_dir": "/checkpoint/10000",
        "checkpoint_step": 10000,
        "config_name": labeling.DEFAULT_CONFIG_NAME,
    }
    labeling.common.atomic_write_json(dataset / labeling.OUTPUT_MARKER, labeling.expected_output_marker(job, context))
    labeling.common.atomic_write_json(
        dataset / labeling.LABELING_MANIFEST,
        {
            **context,
            "source_dataset": str(dataset),
            "future_frames": 50,
            "top_percent": 30.0,
            "grouping": "whole_dataset",
            "advantage_source": "absolute_advantage",
        },
    )

    with pytest.raises(ValueError, match="does not match the requested labeling parameters"):
        labeling.label_job(
            job,
            Path("/checkpoint/10000"),
            context,
            labeling.DEFAULT_CONFIG_NAME,
            ["cpu"],
            1,
            1,
            50,
            20.0,
            verify_training_parity=False,
        )


def test_base_thresholds_are_per_subtask_and_stage1_is_global(tmp_path: Path):
    base = tmp_path / "base"
    base_values = [
        *range(10),
        *range(10, 20),
        *range(20, 30),
        *range(30, 40),
    ]
    write_test_dataset(base, BASE_UMI_SCHEMA, [base_values], with_annotations=False)
    base_table_path = base / "data/chunk-000/episode_000000.parquet"
    base_table = pq.read_table(base_table_path)
    stage_index = np.repeat(np.arange(4), 10)
    pq.write_table(base_table.append_column("stage_index", pa.array(stage_index, type=pa.int64())), base_table_path)
    base_job = _job("base", base, BASE_UMI_SCHEMA, "manual_subtasks", group_count=4)

    base_thresholds = labeling.compute_group_thresholds(base_job, top_percent=30.0)
    assert base_thresholds == pytest.approx({0: 6.3, 1: 16.3, 2: 26.3, 3: 36.3})
    base_summary = labeling.apply_labels(base_job, base_thresholds)
    assert [base_summary[group]["positive"] for group in range(4)] == [3, 3, 3, 3]

    stage1 = tmp_path / "stage1"
    write_test_dataset(stage1, STAGE_UMI_SCHEMA, [list(range(5)), list(range(5, 10))], with_annotations=False)
    stage1_job = _job("stage1", stage1, STAGE_UMI_SCHEMA, "whole_dataset")
    stage1_thresholds = labeling.compute_group_thresholds(stage1_job, top_percent=30.0)
    assert stage1_thresholds == pytest.approx({0: 6.3})
    stage1_summary = labeling.apply_labels(stage1_job, stage1_thresholds)
    assert stage1_summary[0]["total"] == 10
    assert stage1_summary[0]["positive"] == 3


def test_percentile_ties_are_all_positive(tmp_path: Path):
    stage1 = tmp_path / "stage1"
    write_test_dataset(stage1, STAGE_UMI_SCHEMA, [[0] * 8 + [1] * 2], with_annotations=False)
    job = _job("stage1", stage1, STAGE_UMI_SCHEMA, "whole_dataset")

    thresholds = labeling.compute_group_thresholds(job, top_percent=30.0)
    summary = labeling.apply_labels(job, thresholds)

    assert thresholds == {0: 0.0}
    assert summary[0]["positive"] == 10


def test_protected_old_base_labels_are_rejected(tmp_path: Path, monkeypatch):
    protected = tmp_path / "base" / "labeled"
    monkeypatch.setattr(labeling, "PROTECTED_OLD_LABELS", protected)

    with pytest.raises(ValueError, match="protected old-label dataset"):
        labeling.validate_path_separation(tmp_path / "base-origin", tmp_path / "stage1", protected / "child")


def test_mix_checkpoint_accepts_historical_name_without_mutation(tmp_path):
    base = tmp_path / "base"
    stage = tmp_path / "stage"
    metadata = {
        "global_step": 5,
        "config": {
            "name": "ADVANTAGE_TORCH_KAI0_TOWEL_FOLD",
            "advantage_dataset_group_a": [str(base)],
            "advantage_dataset_group_b": [str(stage)],
        },
    }
    original = copy.deepcopy(metadata)
    context = labeling.validate_mix_checkpoint_metadata(
        metadata, tmp_path / "5", base, stage, "ADVANTAGE_TORCH_TOWEL_FOLD"
    )
    assert context["config_name"] == "ADVANTAGE_TORCH_TOWEL_FOLD"
    assert context["original_config_name"] == "ADVANTAGE_TORCH_KAI0_TOWEL_FOLD"
    assert metadata == original
