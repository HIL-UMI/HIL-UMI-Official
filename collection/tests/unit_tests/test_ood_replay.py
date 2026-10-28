# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ood.replay as replay_module
from ood.replay import EpisodeReplayRecorder, ReplayCatalog


class _FakeEncoder:
    def __init__(self, path: Path, **_kwargs) -> None:
        self.path = path
        self.frames = 0

    def write(self, _image: np.ndarray) -> None:
        self.frames += 1

    def close(self) -> None:
        self.path.write_bytes(b"fake-mp4")


def test_saved_episode_archive_is_atomic_and_catalogued(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_module, "_RawVideoEncoder", _FakeEncoder)
    monkeypatch.setattr(replay_module, "_probe_nvenc", lambda _binary: False)
    run_root = tmp_path / "run_20260806_010203Z"
    recorder = EpisodeReplayRecorder(queue_size=4)
    recorder.start_episode(
        {
            "state": "recording",
            "run_root": str(run_root),
            "episode": 1,
            "timestamp": 100.0,
        }
    )
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    snapshot = recorder.record_frame(
        {"front": image, "wrist_left": image},
        {"front": 11, "wrist_left": 22},
        100.1,
    )
    assert snapshot.playback_frame_index == 0
    assert snapshot.run_id == run_root.name
    recorder.append_event(
        {
            "prediction_id": "p0",
            "frame_index": 0,
            "run_id": run_root.name,
            "episode_index": 0,
            "energy_score": 2.5,
            "chunk_ood": True,
        }
    )

    manifest = recorder.finish_saved(
        {"state": "waiting", "saved_frames": 1, "timestamp": 101.0}
    )

    assert manifest is not None
    episode_dir = run_root / "ood_replay" / "episode_000000"
    assert episode_dir.is_dir()
    assert not list((run_root / "ood_replay").glob("*.partial-*"))
    assert (
        json.loads((episode_dir / "events.json").read_text())[0]["prediction_id"]
        == "p0"
    )
    catalog = ReplayCatalog([tmp_path])
    assert catalog.list_episodes()[0]["run_id"] == run_root.name
    assert catalog.resolve_episode(run_root.name, 0) == episode_dir.resolve()


def test_discarded_episode_is_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_module, "_RawVideoEncoder", _FakeEncoder)
    run_root = tmp_path / "run_20260806_020304Z"
    recorder = EpisodeReplayRecorder()
    recorder.start_episode({"run_root": str(run_root), "episode": 2})
    recorder.discard()

    assert not (run_root / "ood_replay" / "episode_000001").exists()


def test_advantage_replay_uses_scalar_schema_and_strict_event_ood(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(replay_module, "_RawVideoEncoder", _FakeEncoder)
    monkeypatch.setattr(replay_module, "_probe_nvenc", lambda _binary: False)
    run_root = tmp_path / "run_20260820_010203Z"
    recorder = EpisodeReplayRecorder(
        detector_mode="advantage", score_threshold=0.0, queue_size=4
    )
    recorder.start_episode({"run_root": str(run_root), "episode": 1})
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    recorder.record_frame(
        {"front": image, "wrist_left": image},
        {"front": 51, "wrist_left": 52},
        100.0,
    )
    for frame_index, score, ood in ((0, -0.4, True), (1, 0.2, False)):
        recorder.append_event(
            {
                "frame_index": frame_index,
                "reference_frame_index": max(0, frame_index - 50),
                "run_id": run_root.name,
                "episode_index": 0,
                "advantage_score": score,
                "ood": ood,
            }
        )

    manifest = recorder.finish_saved({"timestamp": 101.0})

    assert manifest is not None
    assert manifest["schema_version"] == 2
    assert manifest["detector_mode"] == "advantage"
    assert manifest["score_name"] == "advantage_score"
    assert manifest["score_threshold"] == 0.0
    assert manifest["score_min"] == -0.4
    assert manifest["score_max"] == 0.2
    assert manifest["ood_ratio"] == 0.5
    assert "energy_mean" not in manifest


def test_advantage_replay_prepends_idle_frames_and_events(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_module, "_RawVideoEncoder", _FakeEncoder)
    monkeypatch.setattr(replay_module, "_probe_nvenc", lambda _binary: False)
    run_root = tmp_path / "run_20260822_010203Z"
    recorder = EpisodeReplayRecorder(
        fps=2.0,
        detector_mode="advantage",
        score_threshold=0.02,
        pretrigger_s=1.5,
        queue_size=4,
    )
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    for timestamp, frame_id in ((1.0, 10), (1.5, 11), (2.0, 12)):
        recorder.record_frame(
            {"front": image, "wrist_left": image},
            {"front": frame_id, "wrist_left": frame_id + 100},
            timestamp,
        )
    recorder.append_event(
        {
            "type": "advantage_event",
            "step": 7,
            "timestamp": 2.0,
            "reference_timestamp": 1.0,
            "source_frame_ids": {"front": 12, "wrist_left": 112},
            "reference_source_frame_ids": {"front": 10, "wrist_left": 110},
            "prediction_id": "live:-1:-1:7",
            "advantage_score": -0.1,
            "ood": True,
        }
    )

    recorder.start_episode(
        {
            "run_root": str(run_root),
            "episode": 1,
            "timestamp": 2.1,
        }
    )
    active = recorder.record_frame(
        {"front": image, "wrist_left": image},
        {"front": 13, "wrist_left": 113},
        2.5,
    )
    assert active.playback_frame_index == 3
    assert recorder.resolve_frame_index({"front": 12, "wrist_left": 112}, 2.0) == 2
    recorder.append_event(
        {
            "type": "advantage_event",
            "step": 8,
            "timestamp": 2.5,
            "reference_timestamp": 2.0,
            "source_frame_ids": {"front": 13, "wrist_left": 113},
            "reference_source_frame_ids": {"front": 12, "wrist_left": 112},
            "prediction_id": f"{run_root.name}:0:3:8",
            "advantage_score": 0.1,
            "ood": False,
        }
    )

    manifest = recorder.finish_saved({"timestamp": 3.0})

    assert manifest is not None
    assert manifest["schema_version"] == 3
    assert manifest["pretrigger_frame_count"] == 3
    assert manifest["pretrigger_event_count"] == 1
    assert manifest["recording_start_frame"] == 3
    assert manifest["recording_frame_count"] == 1
    assert manifest["pretrigger_duration_s"] == 1.5
    events = json.loads(
        (run_root / "ood_replay" / "episode_000000" / "events.json").read_text()
    )
    assert [(event["frame_index"], event["phase"]) for event in events] == [
        (2, "pretrigger"),
        (3, "recording"),
    ]
    assert events[0]["reference_frame_index"] == 0
    assert events[1]["reference_frame_index"] == 2
    assert events[1]["reference_phase"] == "pretrigger"


def test_camera_discontinuity_resets_pretrigger_context(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_module, "_RawVideoEncoder", _FakeEncoder)
    monkeypatch.setattr(replay_module, "_probe_nvenc", lambda _binary: False)
    run_root = tmp_path / "run_20260822_020304Z"
    recorder = EpisodeReplayRecorder(
        fps=2.0,
        detector_mode="advantage",
        pretrigger_s=1.0,
    )
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    recorder.record_frame(
        {"front": image, "wrist_left": image},
        {"front": 1, "wrist_left": 101},
        1.0,
    )
    recorder.reset_pretrigger("camera_frame_id_rollback")
    recorder.start_episode({"run_root": str(run_root), "episode": 1})
    recorder.record_frame(
        {"front": image, "wrist_left": image},
        {"front": 2, "wrist_left": 102},
        2.0,
    )

    manifest = recorder.finish_saved({"timestamp": 3.0})

    assert manifest is not None
    assert manifest["pretrigger_frame_count"] == 0
    assert manifest["recording_start_frame"] == 0
