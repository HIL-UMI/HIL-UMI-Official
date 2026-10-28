# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the real-world time-aligned asynchronous action buffer."""

import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from training.data.embodied_io_struct import EnvOutput
from training.envs.realworld.franka.pose_trace import AsyncPoseTraceTracker
from training.envs.realworld.time_aligned_action_buffer import (
    TimeAlignedActionBuffer,
    blend_policy_actions,
    ensemble_policy_actions,
    resolve_time_aligned_action_buffer_config,
)
from training.scheduler.worker.routing import infer_batch_size
from training.utils.rot6d import matrix_to_rot6d, rot6d_to_matrix
from training.workers.env.env_worker import EnvWorker
from training.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _chunk(offset: float = 0.0) -> np.ndarray:
    actions = np.zeros((1, 20, 10), dtype=np.float32)
    actions[0, :, 0] = offset + np.arange(20, dtype=np.float32)
    actions[0, :, 3:9] = matrix_to_rot6d(np.repeat(np.eye(3)[None], 20, axis=0))
    actions[0, :, 9] = 1.0
    return actions


def test_async_config_defaults_to_execution_horizon_when_enabled():
    cfg = resolve_time_aligned_action_buffer_config(
        {
            "action_execution_horizon": 10,
            "async_action_buffer": {"enabled": True},
        },
        prediction_horizon=20,
    )

    assert cfg.enabled
    assert cfg.replan_interval_steps == 10
    assert cfg.handover_blend_steps == 3
    assert cfg.temporal_ensemble_decay == pytest.approx(0.05)
    assert cfg.motion_continuity_enabled
    assert cfg.motion_tracking_gain == pytest.approx(0.08)
    assert cfg.max_translation_velocity_m_per_step == pytest.approx(0.005)
    assert cfg.max_translation_acceleration_m_per_step2 == pytest.approx(0.001)
    assert cfg.max_translation_jerk_m_per_step3 == pytest.approx(0.00025)
    assert cfg.max_rotation_velocity_rad_per_step == pytest.approx(0.03)
    assert cfg.max_rotation_acceleration_rad_per_step2 == pytest.approx(0.004)
    assert cfg.max_rotation_jerk_rad_per_step3 == pytest.approx(0.001)
    assert cfg.translation_correction_activate_m == pytest.approx(0.0015)
    assert cfg.translation_correction_release_m == pytest.approx(0.00075)
    assert cfg.rotation_correction_activate_rad == pytest.approx(0.01)
    assert cfg.rotation_correction_release_rad == pytest.approx(0.005)


@pytest.mark.parametrize("interval", [0, 20, 21])
def test_async_config_rejects_interval_without_prediction_coverage(interval):
    with pytest.raises(ValueError, match="replan_interval_steps"):
        resolve_time_aligned_action_buffer_config(
            {
                "async_action_buffer": {
                    "enabled": True,
                    "replan_interval_steps": interval,
                }
            },
            prediction_horizon=20,
        )


@pytest.mark.parametrize("decay", [-0.1, float("inf"), float("nan")])
def test_async_config_rejects_invalid_temporal_ensemble_decay(decay):
    with pytest.raises(ValueError, match="temporal_ensemble_decay"):
        resolve_time_aligned_action_buffer_config(
            {
                "async_action_buffer": {
                    "enabled": True,
                    "replan_interval_steps": 10,
                    "temporal_ensemble_decay": decay,
                }
            },
            prediction_horizon=20,
        )


@pytest.mark.parametrize("gain", [-0.1, 1.1, float("inf"), float("nan")])
def test_async_config_rejects_invalid_motion_tracking_gain(gain):
    with pytest.raises(ValueError, match="motion_tracking_gain"):
        resolve_time_aligned_action_buffer_config(
            {
                "async_action_buffer": {
                    "enabled": True,
                    "replan_interval_steps": 10,
                    "motion_tracking_gain": gain,
                }
            },
            prediction_horizon=20,
        )


@pytest.mark.parametrize(
    "name",
    [
        "max_translation_velocity_m_per_step",
        "max_translation_acceleration_m_per_step2",
        "max_translation_jerk_m_per_step3",
        "max_rotation_velocity_rad_per_step",
        "max_rotation_acceleration_rad_per_step2",
        "max_rotation_jerk_rad_per_step3",
    ],
)
@pytest.mark.parametrize("value", [0.0, -0.1, float("inf"), float("nan")])
def test_async_config_rejects_invalid_motion_limits(name, value):
    with pytest.raises(ValueError, match=name):
        resolve_time_aligned_action_buffer_config(
            {
                "async_action_buffer": {
                    "enabled": True,
                    "replan_interval_steps": 10,
                    name: value,
                }
            },
            prediction_horizon=20,
        )


@pytest.mark.parametrize(
    ("activate_name", "release_name"),
    [
        (
            "translation_correction_activate_m",
            "translation_correction_release_m",
        ),
        ("rotation_correction_activate_rad", "rotation_correction_release_rad"),
    ],
)
def test_async_config_rejects_reversed_correction_hysteresis(
    activate_name, release_name
):
    with pytest.raises(ValueError, match="0 <= release <= activate"):
        resolve_time_aligned_action_buffer_config(
            {
                "async_action_buffer": {
                    "enabled": True,
                    "replan_interval_steps": 10,
                    activate_name: 0.001,
                    release_name: 0.002,
                }
            },
            prediction_horizon=20,
        )


def test_new_chunk_uses_elapsed_control_step_as_action_index():
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=0,
    )
    buffer.install(_chunk(), observation_step=0, ready_step=0)
    assert buffer.select(9).action[0, 0] == pytest.approx(9.0)

    result = buffer.install(_chunk(offset=100.0), observation_step=10, ready_step=18)
    selected = buffer.select(18)

    assert result.accepted
    assert result.aligned_index == 8
    assert selected.chunk_index == 8
    assert selected.action[0, 0] == pytest.approx(108.0)


def test_stale_chunk_is_rejected_and_previous_chunk_remains_active():
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=0,
    )
    buffer.install(_chunk(), observation_step=0, ready_step=0)

    result = buffer.install(_chunk(offset=100.0), observation_step=1, ready_step=21)

    assert not result.accepted
    assert result.aligned_index == 20
    selection = buffer.select(21)
    assert selection.exhausted
    assert selection.action[0, 0] == pytest.approx(19.0)


def test_temporal_ensemble_fuses_predictions_for_the_same_control_step():
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=3,
        temporal_ensemble_decay=0.0,
    )
    buffer.install(_chunk(), observation_step=0, ready_step=0)
    buffer.select(10)
    buffer.install(_chunk(offset=100.0), observation_step=10, ready_step=12)

    first = buffer.select(12)
    second = buffer.select(13)
    third = buffer.select(14)
    after_old_chunk_expires = buffer.select(20)

    assert first.blended
    assert second.blended
    assert third.blended
    assert not after_old_chunk_expires.blended
    assert first.action[0, 0] == pytest.approx((32.0 * 12.0 + 5.0 * 102.0) / 37.0)
    assert second.action[0, 0] == pytest.approx((2.0 * 13.0 + 103.0) / 3.0)
    assert third.action[0, 0] == pytest.approx((32.0 * 14.0 + 27.0 * 104.0) / 59.0)
    assert after_old_chunk_expires.action[0, 0] == pytest.approx(110.0)


def test_pose_trace_tracker_keeps_newest_raw_target_before_buffer_filters():
    old = _chunk()
    new = _chunk(offset=100.0)
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=3,
        temporal_ensemble_decay=0.0,
    )
    tracker = AsyncPoseTraceTracker(prediction_horizon=20)
    buffer.install(old, observation_step=0, ready_step=0)
    tracker.install(
        old,
        request_id=0,
        observation_step=0,
        ready_step=0,
        inference_seconds=0.1,
    )
    buffer.select(10)
    buffer.install(new, observation_step=10, ready_step=12)
    tracker.install(
        new,
        request_id=1,
        observation_step=10,
        ready_step=12,
        inference_seconds=0.2,
    )

    selection = buffer.select(12)
    context = tracker.select(12, blended=selection.blended)

    assert selection.blended
    assert selection.action[0, 0] != pytest.approx(102.0)
    assert context.model_action[0] == pytest.approx(102.0)
    assert context.control_step == 12
    assert context.request_id == 1
    assert context.observation_step == 10
    assert context.ready_step == 12
    assert context.chunk_index == 2
    assert context.candidate_count == 2
    assert context.inference_seconds == pytest.approx(0.2)


def test_temporal_ensemble_tapers_chunk_turnover_without_an_end_jump():
    old = _chunk()
    new = _chunk()
    old[:, :, 0] = 0.0
    new[:, :, 0] = 1.0
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=5,
        temporal_ensemble_decay=0.0,
    )
    buffer.install(old, observation_step=0, ready_step=0)
    before_install = buffer.select(13).action[0, 0]
    buffer.install(new, observation_step=8, ready_step=14)
    outputs = [before_install]
    outputs.extend(buffer.select(step).action[0, 0] for step in range(14, 21))
    increments = np.diff(outputs)

    assert np.all(increments >= 0.0)
    assert outputs[-1] == pytest.approx(1.0)
    assert increments[-1] < increments[-2]
    assert increments[-1] < np.max(increments)


def test_temporal_ensemble_keeps_every_chunk_covering_the_control_step():
    oldest = _chunk()
    middle = _chunk()
    newest = _chunk()
    oldest[:, :, 0] = 0.0
    middle[:, :, 0] = 1.0
    newest[:, :, 0] = 0.0
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=5,
        temporal_ensemble_decay=0.0,
    )
    buffer.install(oldest, observation_step=0, ready_step=0)
    buffer.install(middle, observation_step=4, ready_step=6)
    buffer.install(newest, observation_step=8, ready_step=10)

    selection = buffer.select(10)

    assert selection.blended
    assert 0.0 < selection.action[0, 0] < 1.0


def test_motion_feedforward_does_not_differentiate_chunk_offset_blending():
    old = _chunk()
    new = _chunk()
    old_steps = np.arange(20, dtype=np.float64)
    new_global_steps = 8.0 + old_steps
    old[0, :, 0] = 0.001 * old_steps
    new[0, :, 0] = 0.001 * new_global_steps + 0.004
    old[0, :, 3:9] = matrix_to_rot6d(
        R.from_rotvec(
            (0.005 * old_steps)[:, None] * np.array([[0.0, 0.0, 1.0]])
        ).as_matrix()
    )
    new[0, :, 3:9] = matrix_to_rot6d(
        R.from_rotvec(
            (0.005 * new_global_steps + 0.04)[:, None] * np.array([[0.0, 0.0, 1.0]])
        ).as_matrix()
    )
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=5,
        temporal_ensemble_decay=0.0,
        motion_continuity_enabled=True,
        motion_tracking_gain=0.0,
        max_translation_velocity_m_per_step=0.01,
        max_translation_acceleration_m_per_step2=0.01,
        max_translation_jerk_m_per_step3=0.01,
        max_rotation_velocity_rad_per_step=0.1,
        max_rotation_acceleration_rad_per_step2=0.1,
        max_rotation_jerk_rad_per_step3=0.1,
    )
    buffer.install(old, observation_step=0, ready_step=0)
    outputs = [buffer.select(step).action[0] for step in range(14)]
    buffer.install(new, observation_step=8, ready_step=14)
    outputs.extend(buffer.select(step).action[0] for step in range(14, 28))
    outputs = np.stack(outputs)

    translation_velocities = np.diff(outputs[:, :3], axis=0)
    output_rotations = rot6d_to_matrix(outputs[:, 3:9])
    rotation_deltas = np.einsum(
        "nij,nkj->nik",
        output_rotations[1:],
        output_rotations[:-1],
    )
    angular_velocities = R.from_matrix(rotation_deltas).as_rotvec()

    # The new chunk is offset by 4 mm and 0.04 rad. Its ramping ensemble weight
    # changes the fused pose, but it predicts the same within-chunk velocity as
    # the old chunk, so the emitted motion must remain constant through handover.
    np.testing.assert_allclose(
        translation_velocities[:, 0],
        0.001,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        angular_velocities[:, 2],
        0.005,
        atol=1e-6,
    )


def test_motion_continuity_bounds_translation_acceleration_during_reversal():
    actions = _chunk()
    actions[0, :, 0] = -0.002
    actions[0, :4, 0] = np.array([0.0, 0.002, 0.004, -0.002])
    actions[0, :, 9] = np.linspace(0.0, 1.0, 20)
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=0,
        motion_continuity_enabled=True,
        motion_tracking_gain=0.25,
        max_translation_velocity_m_per_step=0.003,
        max_translation_acceleration_m_per_step2=0.001,
        max_rotation_velocity_rad_per_step=0.03,
        max_rotation_acceleration_rad_per_step2=0.006,
    )
    buffer.install(actions, observation_step=0, ready_step=0)

    outputs = np.stack([buffer.select(step).action[0] for step in range(10)])
    velocities = np.diff(outputs[:, :3], axis=0)
    accelerations = np.diff(
        np.concatenate([np.zeros((1, 3)), velocities], axis=0),
        axis=0,
    )

    assert np.max(np.linalg.norm(velocities, axis=1)) <= 0.003 + 1e-7
    assert np.max(np.linalg.norm(accelerations, axis=1)) <= 0.001 + 1e-7
    # The raw target reverses at step 3, but the emitted velocity decelerates
    # before reversing instead of flipping direction in one 30 Hz tick.
    assert outputs[3, 0] > outputs[2, 0]
    # Gripper timing remains untouched by the SE(3) continuity filter.
    np.testing.assert_allclose(outputs[:, 9], actions[0, :10, 9])


def test_motion_continuity_bounds_so3_angular_acceleration_during_reversal():
    actions = _chunk()
    angles = np.full(20, -0.01)
    angles[:4] = np.array([0.0, 0.01, 0.02, -0.01])
    actions[0, :, 3:9] = matrix_to_rot6d(
        R.from_rotvec(angles[:, None] * np.array([[0.0, 0.0, 1.0]])).as_matrix()
    )
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=0,
        motion_continuity_enabled=True,
        motion_tracking_gain=0.25,
        max_translation_velocity_m_per_step=0.005,
        max_translation_acceleration_m_per_step2=0.001,
        max_rotation_velocity_rad_per_step=0.03,
        max_rotation_acceleration_rad_per_step2=0.005,
    )
    buffer.install(actions, observation_step=0, ready_step=0)

    outputs = np.stack([buffer.select(step).action[0] for step in range(10)])
    output_rotations = rot6d_to_matrix(outputs[:, 3:9])
    rotation_deltas = np.einsum(
        "nij,nkj->nik",
        output_rotations[1:],
        output_rotations[:-1],
    )
    angular_velocities = R.from_matrix(rotation_deltas).as_rotvec()
    angular_accelerations = np.diff(
        np.concatenate([np.zeros((1, 3)), angular_velocities], axis=0),
        axis=0,
    )
    output_angles = R.from_matrix(output_rotations).as_euler("xyz")[:, 2]

    assert np.max(np.linalg.norm(angular_velocities, axis=1)) <= 0.03 + 1e-7
    assert np.max(np.linalg.norm(angular_accelerations, axis=1)) <= 0.005 + 1e-7
    assert output_angles[3] > output_angles[2]


def test_motion_continuity_bounds_translation_jerk():
    actions = _chunk()
    actions[0, :, 0] = np.array(
        [0.0, 0.004, -0.004, 0.004, -0.004] + [0.0] * 15,
        dtype=np.float32,
    )
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=0,
        motion_continuity_enabled=True,
        motion_tracking_gain=0.25,
        max_translation_velocity_m_per_step=0.005,
        max_translation_acceleration_m_per_step2=0.001,
        max_translation_jerk_m_per_step3=0.0002,
        max_rotation_velocity_rad_per_step=0.03,
        max_rotation_acceleration_rad_per_step2=0.004,
        max_rotation_jerk_rad_per_step3=0.001,
        translation_correction_activate_m=0.0,
        translation_correction_release_m=0.0,
    )
    buffer.install(actions, observation_step=0, ready_step=0)

    outputs = np.stack([buffer.select(step).action[0] for step in range(12)])
    velocities = np.diff(outputs[:, :3], axis=0)
    accelerations = np.diff(
        np.concatenate([np.zeros((1, 3)), velocities], axis=0),
        axis=0,
    )
    jerks = np.diff(
        np.concatenate([np.zeros((1, 3)), accelerations], axis=0),
        axis=0,
    )

    assert np.max(np.linalg.norm(accelerations, axis=1)) <= 0.001 + 1e-7
    assert np.max(np.linalg.norm(jerks, axis=1)) <= 0.0002 + 1e-7


def test_motion_continuity_bounds_so3_angular_jerk():
    actions = _chunk()
    angles = np.array(
        [0.0, 0.02, -0.02, 0.02, -0.02] + [0.0] * 15,
        dtype=np.float64,
    )
    actions[0, :, 3:9] = matrix_to_rot6d(
        R.from_rotvec(angles[:, None] * np.array([[0.0, 0.0, 1.0]])).as_matrix()
    )
    buffer = TimeAlignedActionBuffer(
        prediction_horizon=20,
        handover_blend_steps=0,
        motion_continuity_enabled=True,
        motion_tracking_gain=0.25,
        max_translation_velocity_m_per_step=0.005,
        max_translation_acceleration_m_per_step2=0.001,
        max_translation_jerk_m_per_step3=0.001,
        max_rotation_velocity_rad_per_step=0.03,
        max_rotation_acceleration_rad_per_step2=0.004,
        max_rotation_jerk_rad_per_step3=0.001,
        rotation_correction_activate_rad=0.0,
        rotation_correction_release_rad=0.0,
    )
    buffer.install(actions, observation_step=0, ready_step=0)

    outputs = np.stack([buffer.select(step).action[0] for step in range(12)])
    rotations = rot6d_to_matrix(outputs[:, 3:9])
    deltas = np.einsum("nij,nkj->nik", rotations[1:], rotations[:-1])
    angular_velocities = R.from_matrix(deltas).as_rotvec()
    angular_accelerations = np.diff(
        np.concatenate([np.zeros((1, 3)), angular_velocities], axis=0),
        axis=0,
    )
    angular_jerks = np.diff(
        np.concatenate([np.zeros((1, 3)), angular_accelerations], axis=0),
        axis=0,
    )

    assert np.max(np.linalg.norm(angular_accelerations, axis=1)) <= 0.004 + 1e-7
    assert np.max(np.linalg.norm(angular_jerks, axis=1)) <= 0.001 + 1e-7


def test_policy_action_rotation_blends_on_so3():
    first = np.zeros((1, 10), dtype=np.float32)
    second = np.zeros((1, 10), dtype=np.float32)
    first[:, 3:9] = matrix_to_rot6d(np.eye(3))
    second[:, 3:9] = matrix_to_rot6d(R.from_euler("z", 90.0, degrees=True).as_matrix())

    midpoint = blend_policy_actions(first, second, 0.5)
    midpoint_rotation = R.from_matrix(rot6d_to_matrix(midpoint[:, 3:9]))

    assert midpoint_rotation.magnitude()[0] == pytest.approx(np.pi / 4.0, abs=1e-6)


def test_temporal_ensemble_rotation_uses_weighted_quaternion_mean():
    first = np.zeros((1, 10), dtype=np.float32)
    second = np.zeros((1, 10), dtype=np.float32)
    first[:, 3:9] = matrix_to_rot6d(np.eye(3))
    second[:, 3:9] = matrix_to_rot6d(R.from_euler("z", 90.0, degrees=True).as_matrix())

    midpoint = ensemble_policy_actions(
        np.stack([first, second]),
        np.array([0.5, 0.5]),
    )
    midpoint_rotation = R.from_matrix(rot6d_to_matrix(midpoint[:, 3:9]))

    assert midpoint_rotation.magnitude()[0] == pytest.approx(np.pi / 4.0, abs=1e-6)


def test_env_worker_async_protocol_stamps_requests_and_sizes_stop_messages():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create({"rollout": {"group_name": "RolloutGroup"}})
    worker.eval_batch_size = 1
    sent = []
    received = []
    receive_work = object()
    worker.send_to = lambda **kwargs: sent.append(kwargs)
    worker.recv_from = lambda **kwargs: received.append(kwargs) or receive_work
    worker._build_rollout_input_data = lambda env_batch: {
        "obs": env_batch["obs"],
        "final_obs": env_batch["final_obs"],
    }
    env_output = EnvOutput(obs={"states": torch.zeros((1, 10))})

    result = EnvWorker._send_async_action_buffer_request(
        worker,
        rollout_channel=object(),
        input_channel=object(),
        env_output=env_output,
        request_id=7,
        observation_step=31,
    )
    EnvWorker._send_async_action_buffer_stop(worker, object())

    assert result is receive_work
    assert sent[0]["data"]["async_action_buffer_request"] == {
        "kind": "predict",
        "request_id": 7,
        "observation_step": 31,
    }
    assert received[0]["batch_size"] == 1
    assert sent[1]["data"]["async_action_buffer_request"] == {"kind": "stop"}
    assert sent[1]["data"]["async_action_buffer_route_batch"].shape == (1, 1)
    assert infer_batch_size(sent[1]["data"]) == 1
    assert sent[1]["batch_size"] == 1


def test_rollout_worker_async_protocol_returns_full_chunk_and_timestamps():
    class ImmediateAsyncWork:
        def __init__(self, value):
            self.value = value

        async def async_wait(self):
            return self.value

    worker = object.__new__(MultiStepRolloutWorker)
    worker.cfg = OmegaConf.create({"env": {"group_name": "EnvGroup"}})
    worker.env_decoupled_mode = False
    worker.eval_batch_size = 1
    requests = iter(
        [
            {
                "obs": {"states": torch.zeros((1, 10))},
                "async_action_buffer_request": {
                    "kind": "predict",
                    "request_id": 4,
                    "observation_step": 20,
                },
            },
            {"async_action_buffer_request": {"kind": "stop"}},
        ]
    )
    received = []
    sent = []

    def recv_from(**kwargs):
        received.append(kwargs)
        return ImmediateAsyncWork(next(requests))

    worker.recv_from = recv_from
    worker.send_to = lambda **kwargs: sent.append(kwargs)
    worker._predict_rollout_actions = lambda *args, **kwargs: (_chunk(), None)

    asyncio.run(
        MultiStepRolloutWorker._evaluate_with_time_aligned_action_buffer(
            worker,
            input_channel=object(),
            output_channel=object(),
        )
    )

    assert all(call["batch_size"] == 1 for call in received)
    assert len(sent) == 1
    assert sent[0]["data"]["actions"].shape == (1, 20, 10)
    metadata = sent[0]["data"]["async_action_buffer_response"]
    assert metadata["request_id"] == 4
    assert metadata["observation_step"] == 20
    assert metadata["inference_seconds"] >= 0.0
