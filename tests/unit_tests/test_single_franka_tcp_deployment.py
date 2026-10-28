# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UMI-left-hand to single-Franka deployment boundary."""

from types import MethodType
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from training.envs.realworld.common.wrappers.keyboard_eval_control_wrapper import (
    KeyboardEvalControlWrapper,
)
from training.envs.realworld.franka.pose_trace import AsyncPoseTraceContext
from training.envs.realworld.franka.tasks.single_franka_tcp_env import (
    STATE_ACTION_NAMES,
    GripperCommandFilter,
    SingleFrankaTCPEnv,
    clamp_tcp_step,
    policy_gripper_to_rlinf_action,
    pose_xyzw_to_transform,
    render_state_action_curve,
    tcp_tracking_error,
    validate_rigid_transform,
)
from training.envs.realworld.realworld_env import RealWorldEnv
from training.models.embodiment.openpi_pytorch.transforms_pipeline import (
    _explicit_norm_stats_location,
)
from training.utils.rot6d import matrix_to_rot6d


def _dummy_config(**overrides):
    config = {
        "is_dummy": True,
        "camera_serials": ["wrist", "front"],
        "camera_type": "realsense",
        "enable_camera_player": False,
        "image_width": 64,
        "image_height": 48,
        "camera_fps": 30,
        "step_frequency": 30.0,
        "max_num_steps": 300,
        "reset_ee_pose": [
            0.4279,
            0.0294,
            0.4525,
            3.1097,
            0.0192,
            0.0523,
        ],
        "target_ee_pose": [
            0.4279,
            0.0294,
            0.2025,
            3.1097,
            0.0192,
            0.0523,
        ],
        "ee_pose_limit_min": [
            0.2279,
            -0.1706,
            0.2015,
            2.7097,
            -0.3808,
            -0.3477,
        ],
        "ee_pose_limit_max": [
            0.6279,
            0.2294,
            0.4525,
            3.5097,
            0.4192,
            0.4523,
        ],
    }
    config.update(overrides)
    return config


def _make_dummy_env(**overrides):
    return SingleFrankaTCPEnv(
        override_cfg=_dummy_config(**overrides),
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )


def test_validate_rigid_transform_rejects_reflection():
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    with pytest.raises(ValueError, match=r"determinant \+1"):
        validate_rigid_transform(reflection)


def test_clamp_tcp_step_clips_translation_and_rotation():
    current = np.r_[[0.4, 0.0, 0.3], R.identity().as_quat()]
    target = np.r_[
        [0.5, 0.0, 0.3],
        R.from_rotvec([0.0, 0.0, 0.4]).as_quat(),
    ]

    safe, translation_clipped, rotation_clipped = clamp_tcp_step(
        current,
        target,
        max_translation_m=0.005,
        max_rotation_rad=0.05,
    )

    assert translation_clipped
    assert rotation_clipped
    assert np.linalg.norm(safe[:3] - current[:3]) == pytest.approx(0.005)
    relative = R.from_quat(current[3:]).inv() * R.from_quat(safe[3:])
    assert relative.magnitude() == pytest.approx(0.05)


def test_tcp_tracking_error_reports_position_and_shortest_rotation():
    actual = np.r_[[0.4, 0.0, 0.3], R.identity().as_quat()]
    commanded = np.r_[
        [0.403, 0.004, 0.3],
        R.from_rotvec([0.0, 0.0, -0.2]).as_quat(),
    ]

    position_error_m, rotation_error_rad = tcp_tracking_error(actual, commanded)

    assert position_error_m == pytest.approx(0.005)
    assert rotation_error_rad == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("policy_value", "expected_action"),
    [(0.2, -1.0), (0.6, 0.0), (1.0, 1.0), (-2.0, -1.0), (2.0, 1.0)],
)
def test_policy_gripper_maps_continuously_to_omnipicker(policy_value, expected_action):
    assert policy_gripper_to_rlinf_action(
        policy_value,
        closed_value=0.2,
        open_value=1.0,
    ) == pytest.approx(expected_action)


def test_gripper_filter_uses_hysteresis_and_consecutive_debounce():
    command_filter = GripperCommandFilter(
        enabled=True,
        close_threshold=0.4,
        open_threshold=0.6,
        debounce_steps=3,
        closed_value=0.2,
        open_value=1.0,
    )

    # One low outlier and values inside the hysteresis band do not close.
    assert command_filter.apply(0.3) == pytest.approx((1.0, False))
    assert command_filter.apply(0.5) == pytest.approx((1.0, False))
    # Three consecutive close requests commit exactly one transition.
    assert command_filter.apply(0.3) == pytest.approx((1.0, False))
    assert command_filter.apply(0.2) == pytest.approx((1.0, False))
    assert command_filter.apply(0.1) == pytest.approx((0.2, True))
    # Values between thresholds retain the closed latch.
    assert command_filter.apply(0.5) == pytest.approx((0.2, False))
    # Opening is independently debounced.
    assert command_filter.apply(0.8) == pytest.approx((0.2, False))
    assert command_filter.apply(0.9) == pytest.approx((0.2, False))
    assert command_filter.apply(1.0) == pytest.approx((1.0, True))


def test_gripper_filter_disabled_preserves_continuous_command():
    command_filter = GripperCommandFilter(
        enabled=False,
        close_threshold=0.4,
        open_threshold=0.6,
        debounce_steps=3,
        closed_value=0.2,
        open_value=1.0,
    )

    assert command_filter.apply(0.53) == pytest.approx((0.53, False))


def test_virtual_policy_tool_round_trip_and_step_limit():
    franka_to_policy = np.eye(4)
    franka_to_policy[:3, :3] = R.from_euler("z", 90, degrees=True).as_matrix()
    franka_to_policy[:3, 3] = [0.0, 0.02, 0.01]
    env = _make_dummy_env(franka_tcp_to_policy_tool=franka_to_policy)

    observation, _ = env.reset()
    policy_pose = observation["state"]["tcp_pose_rot6d"]
    same_pose_action = np.r_[policy_pose, 1.0]
    recovered_franka_pose = env._policy_action_to_franka_pose(same_pose_action)

    assert np.allclose(
        pose_xyzw_to_transform(recovered_franka_pose),
        pose_xyzw_to_transform(env._franka_state.tcp_pose),
        atol=1e-6,
    )

    move_action = same_pose_action.copy()
    move_action[0] += 0.1
    before = env._franka_state.tcp_pose.copy()
    _, _, _, _, info = env.step(move_action)
    after = env._franka_state.tcp_pose.copy()

    assert np.linalg.norm(after[:3] - before[:3]) == pytest.approx(0.005)
    assert info["safety"]["translation_clipped"]


def test_step_logs_and_returns_controller_error():
    env = _make_dummy_env()
    observation, _ = env.reset()
    env._logger = Mock()
    action = np.r_[observation["state"]["tcp_pose_rot6d"], 1.0]

    _, _, _, _, info = env.step(action)

    env._logger.info.assert_called_once()
    assert "controller_error" in env._logger.info.call_args.args[0]
    assert info["controller_error"]["position_error_m"] == pytest.approx(0.0)
    assert info["controller_error"]["rotation_error_rad"] == pytest.approx(0.0)


def test_pose_trace_is_disabled_without_explicit_parameter():
    env = _make_dummy_env()

    assert env._pose_trace_recorder is None
    assert env._pending_async_pose_trace_context is None

    env.close()


def test_async_pose_trace_records_in_memory_then_saves_atomically(tmp_path):
    env = _make_dummy_env(
        save_pose_trace=True,
        pose_trace_dir=str(tmp_path),
    )
    observation, _ = env.reset()
    buffer_action = np.r_[observation["state"]["tcp_pose_rot6d"], 1.0]
    buffer_action[0] += 0.001
    model_action = buffer_action.copy()
    model_action[0] += 0.019
    env.set_async_pose_trace_context(
        AsyncPoseTraceContext(
            model_action=model_action,
            control_step=17,
            request_id=3,
            observation_step=10,
            ready_step=15,
            chunk_index=7,
            candidate_count=2,
            inference_seconds=0.125,
            blended=True,
            exhausted=False,
        )
    )

    env.step(buffer_action)

    assert env._pose_trace_recorder.sample_count == 1
    assert list(tmp_path.iterdir()) == []

    trace_path = env.save_pose_trace(result="success")

    assert trace_path == str(tmp_path / "episode_000000_success_pose_trace.npz")
    assert not list(tmp_path.glob(".*.npz"))
    with np.load(trace_path, allow_pickle=False) as trace:
        assert trace["schema_version"].item() == 1
        assert trace["control_step"].tolist() == [17]
        assert trace["episode_step"].tolist() == [0]
        assert trace["request_id"].tolist() == [3]
        assert trace["candidate_count"].tolist() == [2]
        assert trace["blended"].tolist() == [True]
        np.testing.assert_allclose(trace["model_action"][0], model_action)
        np.testing.assert_allclose(trace["buffer_action"][0], buffer_action)
        assert trace["model_tool_pose_xyzw"][0, 0] == pytest.approx(model_action[0])
        assert trace["commanded_tool_pose_xyzw"][0, 0] == pytest.approx(
            buffer_action[0]
        )
        np.testing.assert_allclose(
            trace["actual_tool_pose_xyzw"],
            trace["commanded_tool_pose_xyzw"],
        )
        assert trace["model_gripper_value"].tolist() == pytest.approx([1.0])
        assert trace["commanded_gripper_value"].tolist() == pytest.approx([1.0])
        assert trace["actual_gripper_open_fraction"].tolist() == pytest.approx([1.0])

    assert env.save_pose_trace(result="duplicate") is None
    env.close()


def test_sync_final_action_wait_precedes_next_observation_and_is_one_shot(
    monkeypatch,
):
    env = _make_dummy_env()
    observation, _ = env.reset()
    action = np.r_[observation["state"]["tcp_pose_rot6d"], 1.0]
    events = []
    original_get_observation = env._get_observation

    def record_observation():
        events.append(("observation", None))
        return original_get_observation()

    env._get_observation = record_observation
    monkeypatch.setattr(
        "training.envs.realworld.franka.tasks.single_franka_tcp_env.time.sleep",
        lambda seconds: events.append(("sleep", seconds)),
    )

    env.set_post_action_observation_wait_s(0.5)
    env.step(action)

    assert events == [("sleep", 0.5), ("observation", None)]

    events.clear()
    env.step(action)
    assert events == [("observation", None)]


def test_realworld_chunk_applies_wait_only_to_final_action():
    class WaitAwareEnv:
        def __init__(self):
            self.pending_wait_s = 0.0
            self.wait_requests = []

        def get_wrapper_attr(self, name):
            assert name == "set_post_action_observation_wait_s"
            return self.set_post_action_observation_wait_s

        def set_post_action_observation_wait_s(self, wait_s):
            self.wait_requests.append(wait_s)
            self.pending_wait_s = wait_s

    low_level_env = WaitAwareEnv()
    env = object.__new__(RealWorldEnv)
    env.env = type("FakeVectorEnv", (), {"envs": [low_level_env]})()
    env.num_envs = 1
    env.auto_reset = False
    env.ignore_terminations = False
    observed_waits = []

    def fake_step(self, actions, auto_reset):
        del actions, auto_reset
        observed_waits.append(low_level_env.pending_wait_s)
        low_level_env.pending_wait_s = 0.0
        return (
            {"states": torch.zeros((1, 10))},
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    env.step = MethodType(fake_step, env)
    RealWorldEnv.chunk_step(
        env,
        np.zeros((1, 3, 10), dtype=np.float32),
        sync_action_chunk_final_wait_s=0.5,
    )

    assert observed_waits == pytest.approx([0.0, 0.0, 0.5])
    assert low_level_env.wait_requests == pytest.approx([0.5, 0.0])
    assert low_level_env.pending_wait_s == 0.0

    # The asynchronous worker uses one-action chunks without the synchronous
    # wait keyword, so it must not touch the one-shot wait hook.
    observed_waits.clear()
    low_level_env.wait_requests.clear()
    RealWorldEnv.chunk_step(
        env,
        np.zeros((1, 1, 10), dtype=np.float32),
    )

    assert observed_waits == pytest.approx([0.0])
    assert low_level_env.wait_requests == []


def test_state_action_curve_saves_aligned_png_and_npz(tmp_path):
    env = _make_dummy_env(
        save_state_action_curve=True,
        state_action_curve_dir=str(tmp_path),
    )
    observation, _ = env.reset()

    expected_actions = []
    for step in range(3):
        action = np.r_[observation["state"]["tcp_pose_rot6d"], 1.0]
        action[0] += 0.001 * (step + 1)
        expected_actions.append(action.copy())
        observation, _, _, _, _ = env.step(action)

    image_path = env.save_state_action_curve(result="success")

    assert image_path == str(tmp_path / "episode_000000_success_state_action.png")
    image = pytest.importorskip("cv2").imread(image_path)
    assert image is not None
    assert image.shape == (1600, 1560, 3)

    with np.load(tmp_path / "episode_000000_success_state_action.npz") as trace:
        assert trace["states"].shape == (3, 10)
        assert np.allclose(trace["actions"], np.asarray(expected_actions))
        assert np.allclose(trace["time_s"], [0.0, 1.0 / 30.0, 2.0 / 30.0])
        assert trace["channel_names"].tolist() == list(STATE_ACTION_NAMES)
        assert trace["result"].item() == "success"

    assert env.save_state_action_curve(result="duplicate") is None
    env.close()


def test_render_state_action_curve_rejects_misaligned_trace():
    with pytest.raises(ValueError, match="actions must match states shape"):
        render_state_action_curve(
            np.zeros((2, 10)),
            np.zeros((1, 10)),
            step_frequency=30.0,
            title="test",
        )


def test_keyboard_episode_result_automatically_saves_curve(monkeypatch):
    listener = Mock()
    listener.pop_pressed_keys.return_value = ["c"]
    monkeypatch.setattr(
        "training.envs.realworld.common.wrappers.keyboard_eval_control_wrapper."
        "KeyboardListener",
        lambda: listener,
    )

    class CurveEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.saved_results = []

        def step(self, action):
            return np.zeros(1), 0.0, False, False, {}

        def save_state_action_curve(self, *, result):
            self.saved_results.append(result)
            return "/tmp/episode_curve.png"

        def save_pose_trace(self, *, result):
            self.saved_results.append(f"pose:{result}")
            return "/tmp/episode_pose_trace.npz"

    base_env = CurveEnv()
    env = KeyboardEvalControlWrapper(base_env)
    env._running = True

    _, reward, terminated, truncated, info = env.step(np.zeros(10))

    assert reward == pytest.approx(1.0)
    assert terminated
    assert not truncated
    assert base_env.saved_results == ["success", "pose:success"]
    assert info["state_action_curve_path"] == "/tmp/episode_curve.png"
    assert info["pose_trace_path"] == "/tmp/episode_pose_trace.npz"


def test_calibrated_omnipicker_policy_target_recovers_franka_ee_pose():
    """Check the full E->P, C->U action-frame conjugation."""
    ee_to_policy_tool = np.array(
        [
            [0.011082338, -0.707880616, 0.706245152, 0.003064377],
            [0.014651449, -0.706097751, -0.707962783, 0.003577890],
            [0.999831244, 0.018193397, 0.002546264, 0.178781140],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    controller_to_umi_tool = np.array(
        [
            [0.272591650, -0.946236850, -0.174153830, 0.060007256],
            [-0.809545160, -0.323389230, 0.489955160, -0.199112670],
            [-0.519933100, 0.007427719, -0.854174730, -0.121099280],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    env = _make_dummy_env(
        franka_tcp_to_policy_tool=ee_to_policy_tool,
        oculus_controller_to_umi_tool=controller_to_umi_tool,
    )

    base_to_ee = pose_xyzw_to_transform(env._franka_state.tcp_pose)
    controller_delta = np.eye(4)
    controller_delta[:3, :3] = R.from_rotvec([0.08, -0.03, 0.05]).as_matrix()
    controller_delta[:3, 3] = [0.02, -0.01, 0.03]
    ee_to_controller = ee_to_policy_tool @ np.linalg.inv(controller_to_umi_tool)
    base_to_controller_target = base_to_ee @ ee_to_controller @ controller_delta
    policy_action = np.r_[
        base_to_controller_target[:3, 3],
        matrix_to_rot6d(base_to_controller_target[:3, :3]),
        1.0,
    ]

    recovered_ee_pose = env._policy_action_to_franka_pose(policy_action)
    model_physical_tool_pose = env._policy_action_to_physical_tool_pose(policy_action)
    recovered_physical_tool_pose = env._franka_pose_to_physical_tool_pose(
        recovered_ee_pose
    )
    expected_base_to_ee = (
        base_to_ee
        @ ee_to_policy_tool
        @ np.linalg.inv(controller_to_umi_tool)
        @ controller_delta
        @ controller_to_umi_tool
        @ np.linalg.inv(ee_to_policy_tool)
    )

    assert np.allclose(env._franka_to_policy, ee_to_controller, atol=1e-8)
    assert np.allclose(
        pose_xyzw_to_transform(recovered_ee_pose),
        expected_base_to_ee,
        atol=1e-8,
    )
    expected_base_to_physical_tool = base_to_controller_target @ controller_to_umi_tool
    assert np.allclose(
        pose_xyzw_to_transform(model_physical_tool_pose),
        expected_base_to_physical_tool,
        atol=1e-8,
    )
    assert np.allclose(
        pose_xyzw_to_transform(recovered_physical_tool_pose),
        expected_base_to_physical_tool,
        atol=1e-8,
    )


def test_invalid_action_holds_pose_and_dummy_reset_restores_state():
    env = _make_dummy_env()
    observation, _ = env.reset()
    original_pose = env._franka_state.tcp_pose.copy()

    invalid_action = np.r_[observation["state"]["tcp_pose_rot6d"], np.nan]
    _, _, _, _, info = env.step(invalid_action)
    assert info["safety"]["unsafe_action"]
    assert np.array_equal(env._franka_state.tcp_pose, original_pose)
    assert env.num_steps == 1

    env.reset()
    assert env.num_steps == 0
    assert np.allclose(env._franka_state.tcp_pose, original_pose)


def test_registered_env_accepts_manual_episode_control_flag():
    env = gym.make(
        "SingleFrankaTCPEnv-v1",
        override_cfg=_dummy_config(manual_episode_control_only=True),
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={
            "no_gripper": False,
            "use_spacemouse": False,
            "use_gello": False,
            "use_pico": False,
            "use_vive": False,
            "keyboard_reward_wrapper": None,
        },
    )
    assert env.unwrapped.config.manual_episode_control_only
    env.close()


def test_realworld_obs_preserves_state_and_single_extra_camera_shape():
    env = object.__new__(RealWorldEnv)
    env.state_keys = ["tcp_pose_rot6d", "gripper_position"]
    env.main_image_key = "wrist_1"
    env.extra_image_key = "wrist_2"
    env.task_descriptions = ["stack the cubes"]
    pose = np.arange(9, dtype=np.float32)[None]
    gripper = np.array([[0.8]], dtype=np.float32)
    raw_obs = {
        "state": {
            # Reverse insertion order to ensure state_keys is authoritative.
            "gripper_position": gripper,
            "tcp_pose_rot6d": pose,
        },
        "frames": {
            "wrist_1": np.zeros((1, 48, 64, 3), dtype=np.uint8),
            "wrist_2": np.ones((1, 48, 64, 3), dtype=np.uint8),
        },
    }

    wrapped = env._wrap_obs(raw_obs)

    assert torch.equal(
        wrapped["states"],
        torch.from_numpy(np.concatenate([pose, gripper], axis=-1)),
    )
    assert wrapped["main_images"].shape == (1, 48, 64, 3)
    assert wrapped["extra_view_images"].shape == (1, 48, 64, 3)
    assert wrapped["task_descriptions"] == ["stack the cubes"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/tmp/assets/task/norm_stats.json",
            ("/tmp/assets", "task"),
        ),
        (
            "/tmp/assets/task",
            ("/tmp/assets", "task"),
        ),
    ],
)
def test_explicit_norm_stats_location(path, expected):
    assert _explicit_norm_stats_location({"norm_stats_path": path}) == expected
