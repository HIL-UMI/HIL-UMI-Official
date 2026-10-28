# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Franka camera failure recovery."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np

from training.envs.realworld.common.camera import BaseCamera, CameraInfo
from training.envs.realworld.common.camera.realsense_camera import RealSenseCamera
from training.envs.realworld.franka.franka_env import FrankaEnv, FrankaRobotConfig


class _FakeCamera:
    """Minimal camera double that records lifecycle operations."""

    def __init__(self, name, events, *, frame=None, fail=False):
        self._camera_info = SimpleNamespace(name=name, crop_region=None)
        self._events = events
        self._frame = frame
        self._fail = fail

    def open(self):
        self._events.append(("open", self._camera_info.name))

    def close(self):
        self._events.append(("close", self._camera_info.name))

    def get_frame(self):
        self._events.append(("get", self._camera_info.name))
        if self._fail:
            raise queue.Empty
        return self._frame


class _ClosingCamera(BaseCamera):
    """Camera double that models a pipeline stopped during a blocking read."""

    def __init__(self):
        super().__init__(CameraInfo(name="closing", serial_number="serial"))
        self.close_device_calls = 0

    def _read_frame(self):
        self._frame_capturing_start = False
        raise RuntimeError("wait_for_frames cannot be called before start()")

    def _close_device(self):
        self.close_device_calls += 1


def test_camera_timeout_closes_full_set_before_reopening(monkeypatch):
    events = []
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    initial_cameras = [
        _FakeCamera("wrist_1", events, fail=True),
        _FakeCamera("wrist_2", events, frame=frame),
    ]
    replacement_cameras = [
        _FakeCamera("wrist_1", events, frame=frame),
        _FakeCamera("wrist_2", events, frame=frame),
    ]

    env = object.__new__(FrankaEnv)
    env.config = FrankaRobotConfig(is_dummy=False)
    env._cameras = initial_cameras
    env._camera_infos = [
        CameraInfo(name="wrist_1", serial_number="serial-1"),
        CameraInfo(name="wrist_2", serial_number="serial-2"),
    ]
    env.observation_space = gym.spaces.Dict(
        {
            "frames": gym.spaces.Dict(
                {
                    "wrist_1": gym.spaces.Box(
                        0, 255, shape=(48, 64, 3), dtype=np.uint8
                    ),
                    "wrist_2": gym.spaces.Box(
                        0, 255, shape=(48, 64, 3), dtype=np.uint8
                    ),
                }
            )
        }
    )
    env.camera_player = Mock()
    env._logger = Mock()

    def create_replacement_camera(_camera_info):
        assert ("close", "wrist_1") in events
        assert ("close", "wrist_2") in events
        return replacement_cameras.pop(0)

    monkeypatch.setattr(
        "training.envs.realworld.franka.franka_env.create_camera",
        create_replacement_camera,
    )
    monkeypatch.setattr(
        "training.envs.realworld.franka.franka_env.time.sleep",
        lambda seconds: events.append(("sleep", seconds)),
    )

    frames = env._get_camera_frames()

    assert list(frames) == ["wrist_1", "wrist_2"]
    assert events[:4] == [
        ("get", "wrist_1"),
        ("close", "wrist_1"),
        ("close", "wrist_2"),
        ("sleep", 5),
    ]


def test_realsense_close_is_idempotent():
    camera = object.__new__(RealSenseCamera)
    camera._pipeline = Mock()
    camera._config = Mock()
    camera._pipeline_lock = threading.Lock()
    camera._pipeline_started = True

    camera._close_device()
    camera._close_device()

    camera._pipeline.stop.assert_called_once_with()
    camera._config.disable_all_streams.assert_called_once_with()
    assert not camera._pipeline_started


def test_capture_thread_does_not_report_stop_as_read_failure(monkeypatch):
    camera = _ClosingCamera()
    camera._frame_capturing_start = True
    logger = Mock()
    monkeypatch.setattr(
        "training.envs.realworld.common.camera.base_camera._logger",
        logger,
    )
    monkeypatch.setattr(
        "training.envs.realworld.common.camera.base_camera.time.sleep",
        lambda _seconds: None,
    )

    camera._capture_frames()

    logger.error.assert_not_called()
    assert camera.close_device_calls == 0
