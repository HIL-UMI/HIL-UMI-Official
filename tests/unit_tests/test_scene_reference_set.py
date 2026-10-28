# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for named real-world scene-reference presets."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from training.envs.realworld.common.scene_alignment import (
    ReferenceSceneAlignmentWebUI,
)
from training.envs.realworld.common.scene_reference_set import (
    SceneReferenceSet,
    resolve_scene_reference_preset,
)


def _write_manifest(preset_dir, *, num_references=2):
    preset_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "name": preset_dir.name,
        "view_name": "wrist_2",
        "resolution": [12, 8],
        "num_references": num_references,
        "references_dir": "references",
        "sources": [
            {
                "epoch_start": 0,
                "count": num_references,
                "video_dir": "recordings",
                "video_suffix": "wrist_2",
            }
        ],
    }
    (preset_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return preset_dir


def _write_saved_references(preset, *, num_references=2):
    references_dir = preset / "references"
    references_dir.mkdir()
    jpeg = ReferenceSceneAlignmentWebUI._encode_jpeg(
        np.zeros((8, 12, 3), dtype=np.uint8)
    )
    for epoch_index in range(num_references):
        stem = references_dir / f"{epoch_index}_wrist_2_reference"
        stem.with_suffix(".jpg").write_bytes(jpeg)
        stem.with_suffix(".json").write_text(
            json.dumps({"epoch": epoch_index, "frame_index": epoch_index + 4}),
            encoding="utf-8",
        )


def test_scene_reference_set_resolves_source_videos(tmp_path):
    preset_dir = _write_manifest(tmp_path / "stack_cubes")

    preset = SceneReferenceSet.load(preset_dir)

    assert preset.name == "stack_cubes"
    assert preset.source_video(0) == (
        preset_dir / "recordings" / "0_wrist_2.mp4",
        0,
    )
    assert preset.source_video(1) == (
        preset_dir / "recordings" / "1_wrist_2.mp4",
        1,
    )
    assert (
        resolve_scene_reference_preset("stack_cubes", root_dir=tmp_path) == preset_dir
    )


def test_scene_reference_set_requires_every_saved_position(tmp_path):
    preset_dir = _write_manifest(tmp_path / "stack_cubes")
    preset = SceneReferenceSet.load(preset_dir)
    preset.references_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"missing epochs: \[0, 1\]"):
        preset.validate_saved_references()


def test_named_reference_inference_does_not_open_source_videos(tmp_path, monkeypatch):
    preset_dir = _write_manifest(tmp_path / "stack_cubes")
    _write_saved_references(preset_dir)
    ui = ReferenceSceneAlignmentWebUI(reference_set_dir=str(preset_dir))
    inspect_video = Mock(side_effect=AssertionError("source video must not be opened"))
    monkeypatch.setattr(ui, "_inspect_video", inspect_video)
    monkeypatch.setattr(ui, "_ensure_server", Mock())

    reference_path = ui.start_epoch(1)
    status = ui._status_snapshot()

    assert reference_path == preset_dir / "references/1_wrist_2_reference.jpg"
    assert not inspect_video.called
    assert status["preset_name"] == "stack_cubes"
    assert status["video_path"] is None
    assert status["selected_frame_index"] == 5
    assert not status["frame_selection_enabled"]
    with pytest.raises(ValueError, match="not accepting frame selections"):
        ui.preview_video_frame(1, 0)


def test_offline_calibration_can_switch_positions(tmp_path, monkeypatch):
    source_video = tmp_path / "recordings/0_wrist_2.mp4"
    source_video.parent.mkdir()
    source_video.touch()
    cache_dir = tmp_path / "references"
    ui = ReferenceSceneAlignmentWebUI(
        str(source_video.parent),
        cache_dir=str(cache_dir),
        video_paths={10: source_video},
        source_epoch_indices={10: 0},
        calibration_epochs=[10],
    )
    monkeypatch.setattr(ui, "_inspect_video", lambda _path: (20, 10.0))
    monkeypatch.setattr(
        ui,
        "_read_video_frame",
        lambda _path, frame_index: np.full((8, 12, 3), frame_index, dtype=np.uint8),
    )
    monkeypatch.setattr(ui, "_ensure_server", Mock())

    result = ui.select_calibration_epoch(10)
    ui.save_reference_frame(10, 7)
    metadata = json.loads(
        (cache_dir / "10_wrist_2_reference.json").read_text(encoding="utf-8")
    )

    assert result["epoch"] == 10
    assert ui._status_snapshot()["calibration_mode"]
    assert metadata["source_epoch_index"] == 0
    assert b"select-epoch" in ui._render_html()
