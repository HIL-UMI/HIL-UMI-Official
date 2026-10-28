# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the browser-based real-world scene-alignment UI."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from training.envs.realworld.common.scene_alignment import (
    ReferenceSceneAlignmentWebUI,
)
from training.envs.realworld.common.wrappers.keyboard_eval_control_wrapper import (
    KeyboardEvalControlWrapper,
)


def test_scene_alignment_selects_epoch_and_publishes_live_frame(tmp_path, monkeypatch):
    ui = ReferenceSceneAlignmentWebUI(str(tmp_path), default_reference_opacity=0.42)
    reference_frame = np.full((8, 12, 3), 32, dtype=np.uint8)
    (tmp_path / "3_wrist_2.mp4").touch()
    monkeypatch.setattr(ui, "_inspect_video", lambda _path: (20, 10.0))
    monkeypatch.setattr(ui, "_read_video_frame", lambda _path, _index: reference_frame)
    monkeypatch.setattr(ui, "_ensure_server", Mock())

    reference_path = ui.start_epoch(3)
    ui.update_live_frame(np.full_like(reference_frame, 128))

    assert reference_path == tmp_path / "3_wrist_2.mp4"
    assert ui._snapshot("reference") is not None
    assert ui._snapshot("live") is not None
    assert ui._status_snapshot()["epoch"] == 3
    assert ui._status_snapshot()["frame_count"] == 20
    assert not ui._status_snapshot()["reference_saved"]
    assert b'value="0.420"' in ui._render_html()
    assert b'id="frameSlider"' in ui._render_html()


def test_scene_alignment_rejects_different_live_resolution(tmp_path, monkeypatch):
    ui = ReferenceSceneAlignmentWebUI(str(tmp_path))
    (tmp_path / "0_wrist_2.mp4").touch()
    monkeypatch.setattr(ui, "_inspect_video", lambda _path: (20, 10.0))
    monkeypatch.setattr(
        ui,
        "_read_video_frame",
        lambda _path, _index: np.zeros((8, 12, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(ui, "_ensure_server", Mock())
    ui.start_epoch(0)

    with pytest.raises(ValueError, match="identical resolution"):
        ui.update_live_frame(np.zeros((4, 6, 3), dtype=np.uint8))


def test_scene_alignment_saves_and_reuses_reference_per_epoch(tmp_path, monkeypatch):
    video_path = tmp_path / "2_wrist_2.mp4"
    video_path.touch()
    ui = ReferenceSceneAlignmentWebUI(str(tmp_path))
    monkeypatch.setattr(ui, "_inspect_video", lambda _path: (30, 10.0))
    monkeypatch.setattr(
        ui,
        "_read_video_frame",
        lambda _path, frame_index: np.full((8, 12, 3), frame_index, dtype=np.uint8),
    )
    monkeypatch.setattr(ui, "_ensure_server", Mock())
    ui.start_epoch(2)

    result = ui.save_reference_frame(2, 17)

    cached_image = tmp_path / "scene_alignment_references" / "2_wrist_2_reference.jpg"
    cached_metadata = cached_image.with_suffix(".json")
    assert result["frame_index"] == 17
    assert cached_image.is_file()
    assert cached_metadata.is_file()
    assert ui._status_snapshot()["reference_saved"]
    assert ui._status_snapshot()["selected_frame_index"] == 17

    reused_ui = ReferenceSceneAlignmentWebUI(str(tmp_path))
    monkeypatch.setattr(reused_ui, "_inspect_video", lambda _path: (30, 10.0))
    read_frame = Mock(side_effect=AssertionError("saved image should be reused"))
    monkeypatch.setattr(reused_ui, "_read_video_frame", read_frame)
    monkeypatch.setattr(reused_ui, "_ensure_server", Mock())

    reused_ui.start_epoch(2)

    assert not read_frame.called
    assert reused_ui._status_snapshot()["reference_saved"]
    assert reused_ui._status_snapshot()["selected_frame_index"] == 17


def test_scene_alignment_offsets_source_videos_and_uses_shared_cache(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "second_run"
    source_dir.mkdir()
    (source_dir / "0_wrist_2.mp4").touch()
    shared_cache = tmp_path / "all_references"
    ui = ReferenceSceneAlignmentWebUI(
        str(source_dir),
        epoch_offset=10,
        cache_dir=str(shared_cache),
    )
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    monkeypatch.setattr(ui, "_inspect_video", lambda _path: (30, 10.0))
    monkeypatch.setattr(ui, "_read_video_frame", lambda _path, _index: frame)
    monkeypatch.setattr(ui, "_ensure_server", Mock())

    reference_path = ui.start_epoch(10)
    ui.save_reference_frame(10, 12)

    assert reference_path == source_dir / "0_wrist_2.mp4"
    metadata_path = shared_cache / "10_wrist_2_reference.json"
    metadata = metadata_path.read_text(encoding="utf-8")
    assert '"epoch": 10' in metadata
    assert '"source_epoch_index": 0' in metadata
    assert (shared_cache / "10_wrist_2_reference.jpg").is_file()

    with pytest.raises(ValueError, match="precedes configured"):
        ui.start_epoch(9)


def test_scene_alignment_frame_selection_stops_after_rollout(tmp_path, monkeypatch):
    (tmp_path / "0_wrist_2.mp4").touch()
    ui = ReferenceSceneAlignmentWebUI(str(tmp_path))
    monkeypatch.setattr(ui, "_inspect_video", lambda _path: (5, 10.0))
    monkeypatch.setattr(
        ui,
        "_read_video_frame",
        lambda _path, frame_index: np.full((8, 12, 3), frame_index, dtype=np.uint8),
    )
    monkeypatch.setattr(ui, "_ensure_server", Mock())
    ui.start_epoch(0)
    assert ui.preview_video_frame(0, 4)

    ui.mark_rollout_started()

    with pytest.raises(ValueError, match="not accepting frame selections"):
        ui.preview_video_frame(0, 4)


def test_keyboard_reset_returns_fresh_observation_after_alignment(monkeypatch):
    listener = Mock()
    listener.pop_pressed_keys.side_effect = [[], ["a"]]
    monkeypatch.setattr(
        "training.envs.realworld.common.wrappers.keyboard_eval_control_wrapper."
        "KeyboardListener",
        lambda: listener,
    )
    monkeypatch.setattr(
        "training.envs.realworld.common.wrappers.keyboard_eval_control_wrapper.time.sleep",
        lambda _seconds: None,
    )

    class AlignmentEnv(gym.Env):
        scene_alignment_epoch_offset = 10

        def __init__(self):
            super().__init__()
            self.calls = []

        def reset(self, *, seed=None, options=None):
            self.calls.append("reset")
            return {"frame_id": "stale"}, {"reset": True}

        def begin_scene_alignment(self, epoch_index):
            self.calls.append(("begin", epoch_index))
            return True

        def refresh_scene_alignment(self):
            self.calls.append("refresh")

        def capture_scene_alignment_observation(self):
            self.calls.append("capture")
            return {"frame_id": "fresh"}

        def finish_scene_alignment(self):
            self.calls.append("finish")

    base_env = AlignmentEnv()
    env = KeyboardEvalControlWrapper(base_env)

    observation, info = env.reset()

    assert observation == {"frame_id": "fresh"}
    assert info == {"reset": True}
    assert env._last_obs == observation
    assert env._running
    assert env._epoch_index == 11
    assert base_env.calls == [
        "reset",
        ("begin", 10),
        "refresh",
        "capture",
        "finish",
    ]
