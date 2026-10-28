# Modified for HIL-UMI.
import copy
import json
from pathlib import Path

import pytest

from hil_umi_advantage.training.advantage_schema import BASE_UMI_SCHEMA
from hil_umi_advantage.training.advantage_schema import STAGE_UMI_SCHEMA
from scripts import label_advantage_datasets as labeling
from scripts.label_towel_mix_acbc_test import write_test_dataset


def test_dataset_spec_resolves_stage_origin_and_rejects_duplicates(tmp_path: Path):
    stage_root = tmp_path / "stage1"
    write_test_dataset(stage_root / "origin", STAGE_UMI_SCHEMA, [list(range(5))], with_annotations=False)

    specs = labeling.normalize_dataset_specs([labeling.parse_named_path(f"stage1={stage_root}")])

    assert specs == [labeling.DatasetSpec("stage1", (stage_root / "origin").resolve())]
    with pytest.raises(ValueError, match="output names must be unique"):
        labeling.normalize_dataset_specs(
            [
                labeling.parse_named_path(f"stage1={stage_root}"),
                labeling.parse_named_path(f"stage1={stage_root / 'origin'}"),
            ]
        )


def test_generic_checkpoint_accepts_targets_outside_training_groups(tmp_path: Path):
    checkpoint = tmp_path / "7500"
    metadata = {
        "global_step": 7500,
        "config": {
            "name": "ADVANTAGE_TORCH_TOWEL_FOLD",
            "exp_name": "base_stage1_stage2_mix",
            "advantage_estimator": True,
            "advantage_dataset_group_a": ("/training/base", "/training/stage2"),
            "advantage_dataset_group_b": ("/training/stage1",),
        },
    }

    context = labeling.validate_advantage_checkpoint_metadata(metadata, checkpoint, requested_config_name=None)

    assert context["checkpoint_step"] == 7500
    assert context["config_name"] == "ADVANTAGE_TORCH_TOWEL_FOLD"
    assert context["training_groups"]["group_a"] == ["/training/base", "/training/stage2"]
    metadata["config"]["advantage_estimator"] = False
    with pytest.raises(ValueError, match="not an advantage estimator"):
        labeling.validate_advantage_checkpoint_metadata(metadata, checkpoint, requested_config_name=None)


def test_jobs_auto_detect_dynamic_subtasks_and_whole_dataset_override(tmp_path: Path):
    base = tmp_path / "base"
    stage1 = tmp_path / "stage1"
    write_test_dataset(base, BASE_UMI_SCHEMA, [list(range(8))], with_annotations=False)
    write_test_dataset(stage1, STAGE_UMI_SCHEMA, [list(range(8))], with_annotations=False)
    (base / "augmentation_metadata.json").write_text(
        json.dumps({"0": {"segments": [], "subtask_completion_indices": [2, 5]}})
    )
    (stage1 / "augmentation_metadata.json").write_text(
        json.dumps({"0": {"segments": [], "subtask_completion_indices": [2, 4, 6]}})
    )
    specs = [
        labeling.DatasetSpec("base", base.resolve()),
        labeling.DatasetSpec("stage1", stage1.resolve()),
    ]

    jobs = labeling.build_dataset_jobs(
        specs,
        (tmp_path / "output").resolve(),
        whole_dataset_names={"stage1"},
        annotation_paths={},
    )

    assert jobs[0].grouping == "manual_subtasks"
    assert jobs[0].group_count == 3
    assert jobs[0].annotations == {0: [2, 5]}
    assert jobs[1].grouping == "whole_dataset"
    assert jobs[1].group_count == 1


def test_explicit_annotations_must_reference_a_known_dataset(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown dataset name"):
        labeling.normalize_named_paths(
            [("stage2", tmp_path / "annotations.json")],
            valid_names={"base", "stage1"},
            option="--subtask-annotations",
        )


def test_cli_exposes_batch_size_with_larger_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "label_advantage_datasets.py",
            "--checkpoint",
            "/checkpoint/10000",
            "--output-root",
            "/output",
            "--dataset",
            "base=/data/base",
        ],
    )

    args = labeling.parse_args()

    assert args.batch_size == 32
    assert args.decode_chunk_size >= args.batch_size


def test_checkpoint_metadata_accepts_historical_config_without_mutation(tmp_path):
    metadata = {
        "global_step": 5,
        "config": {"name": "ADVANTAGE_TORCH_KAI0_TOWEL_FOLD", "advantage_estimator": True},
    }
    original = copy.deepcopy(metadata)
    context = labeling.validate_advantage_checkpoint_metadata(metadata, tmp_path / "5", "ADVANTAGE_TORCH_TOWEL_FOLD")
    assert context["config_name"] == "ADVANTAGE_TORCH_TOWEL_FOLD"
    assert context["original_config_name"] == "ADVANTAGE_TORCH_KAI0_TOWEL_FOLD"
    assert metadata == original
    with pytest.raises(ValueError, match="config mismatch"):
        labeling.validate_advantage_checkpoint_metadata(metadata, tmp_path / "5", "ADVANTAGE_TORCH_UMI")
