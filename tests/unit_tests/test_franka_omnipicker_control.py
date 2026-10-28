# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dispatching OmniPicker actions through the Franka stack."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from training.envs.realworld.franka.end_effectors import EndEffectorType
from training.envs.realworld.franka.franka_controller import FrankaController
from training.envs.realworld.franka.franka_env import FrankaEnv
from training.envs.realworld.franka.franka_robot_state import FrankaRobotState
from training.envs.realworld.franka.tasks.single_franka_tcp_env import (
    SingleFrankaTCPEnv,
)


class _WaitResult:
    def __init__(self, result):
        self._result = result

    def wait(self):
        return [self._result]


class _ContinuousGripper:
    def __init__(self):
        self.actions = []
        self.positions = []
        self.cleanup_calls = 0
        self._is_open = False

    def command_rlinf_action(self, action):
        self.actions.append(action)
        position_pct = (float(action) + 1.0) / 2.0
        self.positions.append(position_pct)
        was_open = self._is_open
        self._is_open = position_pct >= 0.5
        return was_open != self._is_open

    @property
    def is_open(self):
        return self._is_open

    def cleanup(self):
        self.cleanup_calls += 1


class _BinaryGripper:
    def __init__(self):
        self.open_calls = 0
        self.close_calls = 0

    def open(self):
        self.open_calls += 1

    def close(self):
        self.close_calls += 1


def _make_controller(end_effector_type, gripper, *, gripper_open):
    controller = object.__new__(FrankaController)
    if hasattr(gripper, "_is_open"):
        gripper._is_open = gripper_open
    controller._end_effector_type = end_effector_type
    controller._gripper = gripper
    controller._end_effector = None
    controller._state = FrankaRobotState(gripper_open=gripper_open)
    controller.log_debug = lambda _message: None
    return controller


def test_controller_maps_each_omnipicker_action_and_repeats_targets():
    gripper = _ContinuousGripper()
    controller = _make_controller(
        EndEffectorType.OMNIPICKER_GRIPPER,
        gripper,
        gripper_open=True,
    )

    for value in (-1.0, 0.0, 0.0, 1.0):
        controller.command_end_effector(np.array([value]))

    assert gripper.actions == pytest.approx([-1.0, 0.0, 0.0, 1.0])
    assert gripper.positions == pytest.approx([0.0, 0.5, 0.5, 1.0])


def test_controller_keeps_legacy_grippers_transition_only():
    gripper = _BinaryGripper()
    controller = _make_controller(
        EndEffectorType.ROBOTIQ_GRIPPER,
        gripper,
        gripper_open=True,
    )

    results = [
        controller.command_end_effector(np.array([-1.0])),
        controller.command_end_effector(np.array([-1.0])),
        controller.command_end_effector(np.array([0.0])),
        controller.command_end_effector(np.array([1.0])),
        controller.command_end_effector(np.array([1.0])),
    ]

    assert results == [True, False, False, True, False]
    assert gripper.close_calls == 1
    assert gripper.open_calls == 1


def test_controller_cleanup_releases_all_resources_once(monkeypatch):
    gripper = _ContinuousGripper()
    controller = _make_controller(
        EndEffectorType.OMNIPICKER_GRIPPER,
        gripper,
        gripper_open=False,
    )
    joint_process = object()
    impedance_process = object()
    ros = SimpleNamespace(close=Mock())
    controller._joint = joint_process
    controller._impedance = impedance_process
    controller._ros = ros
    controller._logger = Mock()
    stopped_processes = []
    monkeypatch.setattr(
        "training.envs.realworld.franka.franka_controller.stop_process_group",
        lambda process, *, name, logger: stopped_processes.append((process, name)),
    )

    controller.cleanup()
    controller.cleanup()

    assert gripper.cleanup_calls == 1
    assert stopped_processes == [
        (joint_process, "joint controller"),
        (impedance_process, "impedance controller"),
    ]
    ros.close.assert_called_once_with()


class _RemoteEndEffector:
    def __init__(self):
        self.commands = []
        self.open_calls = 0
        self.close_calls = 0
        self.cleanup_calls = 0
        self.worker_close_calls = 0

    def command_end_effector(self, action):
        self.commands.append(np.asarray(action).copy())
        return _WaitResult(True)

    def open_gripper(self):
        self.open_calls += 1
        return _WaitResult(None)

    def close_gripper(self):
        self.close_calls += 1
        return _WaitResult(None)

    def cleanup(self):
        self.cleanup_calls += 1
        return _WaitResult(None)

    def _close(self):
        self.worker_close_calls += 1


def _make_env(end_effector_type):
    env = object.__new__(FrankaEnv)
    env.config = SimpleNamespace(
        end_effector_type=end_effector_type.value,
        action_scale=np.array([0.03, 0.1, 1.0]),
        binary_gripper_threshold=0.5,
        is_dummy=False,
    )
    env._controller = _RemoteEndEffector()
    env._franka_state = FrankaRobotState(gripper_open=True)
    return env


def test_env_sends_every_omnipicker_step_without_binary_filtering():
    env = _make_env(EndEffectorType.OMNIPICKER_GRIPPER)

    results = [
        env._end_effector_action(np.array([value])) for value in (-1.0, 0.0, 0.0, 1.0)
    ]

    assert results == [True, True, True, True]
    assert [float(command[0]) for command in env._controller.commands] == pytest.approx(
        [-1.0, 0.0, 0.0, 1.0]
    )
    assert env._controller.open_calls == 0
    assert env._controller.close_calls == 0


def test_env_keeps_legacy_gripper_transition_only(monkeypatch):
    env = _make_env(EndEffectorType.ROBOTIQ_GRIPPER)
    monkeypatch.setattr(
        "training.envs.realworld.franka.franka_env.time.sleep",
        lambda _seconds: None,
    )

    assert env._end_effector_action(np.array([-1.0]))
    env._franka_state.gripper_open = False
    assert not env._end_effector_action(np.array([-1.0]))
    assert env._end_effector_action(np.array([1.0]))
    env._franka_state.gripper_open = True
    assert not env._end_effector_action(np.array([1.0]))

    assert env._controller.close_calls == 1
    assert env._controller.open_calls == 1


def test_env_close_releases_controller_once():
    env = _make_env(EndEffectorType.OMNIPICKER_GRIPPER)

    env.close()
    env.close()

    assert env._controller.cleanup_calls == 1
    assert env._controller.worker_close_calls == 1


def test_single_tcp_reset_opens_gripper_before_arm_reset(monkeypatch):
    env = object.__new__(SingleFrankaTCPEnv)
    env.config = SimpleNamespace(
        is_dummy=False,
        open_gripper_on_reset=True,
        save_state_action_curve=False,
    )
    env._controller = _RemoteEndEffector()
    env._last_commanded_quat = np.ones(4)
    env._franka_state = FrankaRobotState(gripper_open=True)
    env._gripper_command_filter = SimpleNamespace(reset=Mock())
    call_order = []

    def open_gripper():
        call_order.append("open_gripper")
        return _WaitResult(None)

    def base_reset(_self, *, seed=None, options=None):
        del seed, options
        call_order.append("arm_reset")
        return {"state": {}}, {}

    env._controller.open_gripper = open_gripper
    monkeypatch.setattr(
        "training.envs.realworld.franka.tasks.single_franka_tcp_env.time.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(FrankaEnv, "reset", base_reset)

    result = env.reset()

    assert call_order == ["open_gripper", "arm_reset"]
    assert result == ({"state": {}}, {})
    assert env._last_commanded_quat is None
