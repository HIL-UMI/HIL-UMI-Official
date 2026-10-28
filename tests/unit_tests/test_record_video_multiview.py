# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future
from pathlib import Path

import numpy as np

from training.envs.wrappers.record_video import RecordVideo


class _DummyEnv:
    seed = 7
    metadata = {"render_fps": 30}


def test_record_video_writes_primary_and_named_additional_view(
    tmp_path: Path, monkeypatch
):
    recorder = RecordVideo(
        _DummyEnv(),
        {
            "video_base_dir": str(tmp_path),
            "info_on_video": False,
            "additional_image_fields": {"wrist_2": "extra_view_images"},
        },
    )
    primary = np.full((1, 4, 5, 3), 11, dtype=np.uint8)
    wrist_2 = np.full((1, 4, 5, 3), 22, dtype=np.uint8)
    recorder.add_new_frames({"main_images": primary, "extra_view_images": wrist_2})

    saved: dict[str, list[np.ndarray]] = {}

    def fake_submit(frames: list[np.ndarray], mp4_path: str) -> Future:
        saved[Path(mp4_path).name] = frames
        future = Future()
        future.set_result(None)
        return future

    monkeypatch.setattr(recorder, "_submit_save", fake_submit)
    recorder.flush_video()
    recorder._executor.shutdown(wait=True)

    assert set(saved) == {"0.mp4", "0_wrist_2.mp4"}
    assert len(saved["0.mp4"]) == len(saved["0_wrist_2.mp4"]) == 1
    np.testing.assert_array_equal(saved["0.mp4"][0], primary[0])
    np.testing.assert_array_equal(saved["0_wrist_2.mp4"][0], wrist_2[0])
    assert recorder.video_cnt == 1


def test_missing_additional_view_keeps_legacy_primary_video(
    tmp_path: Path, monkeypatch
):
    recorder = RecordVideo(
        _DummyEnv(),
        {
            "video_base_dir": str(tmp_path),
            "info_on_video": False,
            "additional_image_fields": {"wrist_2": "extra_view_images"},
        },
    )
    recorder.add_new_frames({"main_images": np.zeros((1, 3, 3, 3), dtype=np.uint8)})

    saved_paths = []

    def fake_submit(frames: list[np.ndarray], mp4_path: str) -> Future:
        saved_paths.append(Path(mp4_path).name)
        future = Future()
        future.set_result(None)
        return future

    monkeypatch.setattr(recorder, "_submit_save", fake_submit)
    recorder.flush_video()
    recorder._executor.shutdown(wait=True)

    assert saved_paths == ["0.mp4"]
