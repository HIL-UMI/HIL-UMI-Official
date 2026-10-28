#!/usr/bin/env python3
# Modified for HIL-UMI.
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Replay one UMI demonstration on Franka without loading a policy.

The converted LeRobot parquet stores absolute Oculus-controller poses
``T_H_C``.  Training turns a future absolute target into a body-frame delta

``D_C = inverse(T_H_C_state) @ T_H_C_action``.

For a complete demonstration this program builds the equivalent consecutive
training deltas, verifies that they reconstruct the source trajectory, and
anchors the trajectory at the live Franka endpoint pose.  The controller-frame
motion is mapped to the Franka endpoint with the same extrinsics as deployment:

``A = T_E_P @ inverse(T_C_U)``

``T_B_E[k] = T_B_E[0] @ A @ inverse(T_H_C[0]) @ T_H_C[k] @ inverse(A)``.

The source parquet is never modified.  Hardware execution is opt-in through
``--execute``; otherwise the configured reset pose is used for an offline
workspace and step-size check.  OmniPicker commands are separately opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from training.utils.rot6d import matrix_to_rot6d, pose_to_SE3
from toolkits.realworld_check.check_franka_policy_frame_axes import (
    FrankaStateMonitor,
    PolicyFrameAxisConfig,
    _confirm_exact,
    _format_pose,
    _process_group_stop,
    _start_ros_master,
    load_axis_test_config,
    pose_euler_to_transform,
    pose_xyzw_to_transform,
    transform_to_pose_xyzw,
    validate_rigid_transform,
    validate_workspace_target,
)

DEFAULT_CONFIG = Path(
    "evaluations/realworld/env/realworld_single_franka_tcp_rot6d.yaml"
)
DEFAULT_DATASET = Path("dataset/stack_cubes_umi_1_franka")
DEFAULT_SOURCE_DATASET = Path("dataset/stack_cubes_umi_1")
DEFAULT_ROBOT_IP = "172.16.1.1"


@dataclass(frozen=True)
class ReplayConfig:
    """Deployment geometry and safety settings needed by replay."""

    axis: PolicyFrameAxisConfig
    controller_to_umi_tool: np.ndarray
    endpoint_to_controller: np.ndarray
    endpoint_config: dict[str, Any]
    policy_gripper_closed_value: float
    policy_gripper_open_value: float


@dataclass(frozen=True)
class UmiEpisode:
    """Absolute source poses and their training-compatible one-step deltas."""

    episode_index: int
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    absolute_actions: np.ndarray
    controller_poses: np.ndarray
    training_step_actions: np.ndarray
    tracking_invalid_frames: int


@dataclass(frozen=True)
class ReplayPlan:
    """Time-resampled endpoint targets anchored at one Franka pose."""

    target_times_s: np.ndarray
    source_times_s: np.ndarray
    endpoint_targets: np.ndarray
    gripper_values: np.ndarray
    max_step_translation_m: float
    max_step_rotation_rad: float
    max_anchor_translation_m: float
    max_anchor_rotation_rad: float


def load_replay_config(path: Path) -> ReplayConfig:
    """Load the deployment transforms without changing the YAML."""
    axis = load_axis_test_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        override = raw["override_cfg"]
        controller_to_umi = validate_rigid_transform(
            np.asarray(override["oculus_controller_to_umi_tool"], dtype=np.float64),
            "oculus_controller_to_umi_tool",
        )
        endpoint_config = dict(override["end_effector_config"])
        closed_value = float(override["policy_gripper_closed_value"])
        open_value = float(override["policy_gripper_open_value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid UMI replay config {path}: {exc}") from exc

    endpoint_to_controller = axis.ee_to_policy @ np.linalg.inv(controller_to_umi)
    validate_rigid_transform(endpoint_to_controller, "endpoint_to_controller")
    if not 0.0 <= closed_value < open_value <= 1.0:
        raise ValueError("Policy gripper values must satisfy 0 <= closed < open <= 1.")
    return ReplayConfig(
        axis=axis,
        controller_to_umi_tool=controller_to_umi,
        endpoint_to_controller=endpoint_to_controller,
        endpoint_config=endpoint_config,
        policy_gripper_closed_value=closed_value,
        policy_gripper_open_value=open_value,
    )


def load_anchor_file(path: Path, episode_index: int) -> np.ndarray:
    """Load a previously captured Franka replay anchor."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        recorded_episode = int(record["episode_index"])
        if recorded_episode != episode_index:
            raise ValueError(
                f"anchor episode {recorded_episode} does not match {episode_index}"
            )
        if "anchor_base_to_endpoint" in record:
            anchor = np.asarray(record["anchor_base_to_endpoint"], dtype=np.float64)
        else:
            anchor = pose_euler_to_transform(record["anchor_pose_xyz_euler"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid replay anchor file {path}: {exc}") from exc
    return validate_rigid_transform(anchor, "saved_replay_anchor")


def save_anchor_file(
    path: Path,
    anchor: np.ndarray,
    episode: UmiEpisode,
) -> None:
    """Persist the exact live anchor before any real-robot movement."""
    anchor = validate_rigid_transform(anchor, "saved_replay_anchor")
    pose = np.concatenate(
        [anchor[:3, 3], R.from_matrix(anchor[:3, :3]).as_euler("xyz")]
    )
    record = {
        "schema_version": 1,
        "episode_index": episode.episode_index,
        "start_frame": int(episode.frame_indices[0]),
        "anchor_pose_xyz_euler": pose.tolist(),
        "anchor_base_to_endpoint": anchor.tolist(),
        "created_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _episode_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _column_matrix(table: Any, key: str, width: int) -> np.ndarray:
    if key not in table.column_names:
        raise ValueError(f"Parquet is missing required column {key!r}.")
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{key} must have shape (N, {width}), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} contains NaN or Inf.")
    return values


def _load_tracking_invalid_count(
    source_root: Path | None,
    episode_index: int,
    start_frame: int,
    stop_frame: int,
) -> int:
    """Count raw left-controller tracking-invalid rows when available."""
    if source_root is None:
        return 0
    source_path = _episode_path(source_root, episode_index)
    if not source_path.is_file():
        print(f"WARN: raw tracking diagnostics not found: {source_path}")
        return 0
    table = pq.read_table(source_path)
    keys = (
        "diagnostics.oculus.raw_not_full_valid.left",
        "diagnostics.oculus.orientation_valid.left",
        "diagnostics.oculus.position_valid.left",
    )
    if not all(key in table.column_names for key in keys):
        print("WARN: raw episode does not contain complete left Oculus diagnostics.")
        return 0
    invalid = np.asarray(table[keys[0]].to_pylist(), dtype=np.float64) > 0.5
    orientation_valid = np.asarray(table[keys[1]].to_pylist(), dtype=np.float64) > 0.5
    position_valid = np.asarray(table[keys[2]].to_pylist(), dtype=np.float64) > 0.5
    invalid |= ~orientation_valid | ~position_valid
    return int(np.count_nonzero(invalid[start_frame:stop_frame]))


def load_umi_episode(
    dataset_root: Path,
    source_root: Path | None,
    episode_index: int,
    *,
    start_frame: int,
    end_frame: int | None,
) -> UmiEpisode:
    """Load one episode and construct the step-1 target used by training."""
    path = _episode_path(dataset_root, episode_index)
    if not path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    table = pq.read_table(path)
    actions_all = _column_matrix(table, "actions", 10)
    states_all = _column_matrix(table, "state", 10)
    timestamps_all = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    frame_indices_all = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)

    stop_frame = len(actions_all) if end_frame is None else end_frame + 1
    if not 0 <= start_frame < stop_frame <= len(actions_all):
        raise ValueError(
            f"Invalid frame range [{start_frame}, {stop_frame}); episode has "
            f"{len(actions_all)} frames."
        )
    if stop_frame - start_frame < 2:
        raise ValueError("Replay requires at least two source frames.")

    actions = actions_all[start_frame:stop_frame].copy()
    states = states_all[start_frame:stop_frame].copy()
    timestamps = timestamps_all[start_frame:stop_frame].copy()
    frame_indices = frame_indices_all[start_frame:stop_frame].copy()
    timestamps -= timestamps[0]
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("Episode timestamps must be strictly increasing.")

    pose_difference = float(np.max(np.abs(actions[:, :9] - states[:, :9])))
    if pose_difference > 1e-5:
        raise ValueError(
            "This replay expects the converted UMI action pose to equal the "
            f"same-row state pose; max difference is {pose_difference:.3e}."
        )

    controller_poses = pose_to_SE3(actions[:, :3], actions[:, 3:9])
    step_deltas = np.linalg.inv(controller_poses[:-1]) @ controller_poses[1:]
    training_actions = np.empty((len(step_deltas), 10), dtype=np.float64)
    training_actions[:, :3] = step_deltas[:, :3, 3]
    training_actions[:, 3:9] = matrix_to_rot6d(step_deltas[:, :3, :3])
    # Training keeps the gripper absolute.  This is horizon element k=1 for
    # each current row; horizon element k=0 is identity pose + same-row grip.
    training_actions[:, 9] = actions[1:, 9]

    reconstructed = np.empty_like(controller_poses)
    reconstructed[0] = controller_poses[0]
    for index, delta in enumerate(step_deltas, start=1):
        reconstructed[index] = reconstructed[index - 1] @ delta
    if not np.allclose(reconstructed, controller_poses, atol=2e-6):
        raise RuntimeError("Training delta reconstruction does not match source poses.")

    invalid_count = _load_tracking_invalid_count(
        source_root, episode_index, start_frame, stop_frame
    )
    return UmiEpisode(
        episode_index=episode_index,
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        absolute_actions=actions,
        controller_poses=controller_poses,
        training_step_actions=training_actions,
        tracking_invalid_frames=invalid_count,
    )


def _motion_magnitudes(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translations = np.linalg.norm(deltas[..., :3, 3], axis=-1)
    rotations = R.from_matrix(deltas[..., :3, :3].reshape(-1, 3, 3)).magnitude()
    return translations, rotations.reshape(translations.shape)


def validate_source_deltas(
    episode: UmiEpisode,
    *,
    max_translation_m: float,
    max_rotation_rad: float,
    allow_invalid_tracking: bool,
) -> tuple[float, float]:
    """Reject tracking jumps before they can become robot commands."""
    deltas = np.linalg.inv(episode.controller_poses[:-1]) @ episode.controller_poses[1:]
    translations, rotations = _motion_magnitudes(deltas)
    max_translation = float(np.max(translations))
    max_rotation = float(np.max(rotations))
    if max_translation > max_translation_m:
        index = int(np.argmax(translations))
        raise ValueError(
            f"Source translation jump {max_translation:.4f}m at frames "
            f"{episode.frame_indices[index]}->{episode.frame_indices[index + 1]} "
            f"exceeds {max_translation_m:.4f}m."
        )
    if max_rotation > max_rotation_rad:
        index = int(np.argmax(rotations))
        raise ValueError(
            f"Source rotation jump {max_rotation:.4f}rad at frames "
            f"{episode.frame_indices[index]}->{episode.frame_indices[index + 1]} "
            f"exceeds {max_rotation_rad:.4f}rad."
        )
    if episode.tracking_invalid_frames and not allow_invalid_tracking:
        raise ValueError(
            f"Raw episode reports {episode.tracking_invalid_frames} invalid left "
            "Oculus frames; pass --allow-invalid-tracking only after inspection."
        )
    return max_translation, max_rotation


def _resample_transforms(
    key_times_s: np.ndarray,
    transforms: np.ndarray,
    query_times_s: np.ndarray,
) -> np.ndarray:
    positions = np.stack(
        [
            np.interp(query_times_s, key_times_s, transforms[:, axis, 3])
            for axis in range(3)
        ],
        axis=1,
    )
    rotations = Slerp(
        key_times_s,
        R.from_matrix(transforms[:, :3, :3]),
    )(query_times_s).as_matrix()
    output = np.repeat(np.eye(4, dtype=np.float64)[None], len(query_times_s), axis=0)
    output[:, :3, 3] = positions
    output[:, :3, :3] = rotations
    return output


def build_replay_plan(
    episode: UmiEpisode,
    config: ReplayConfig,
    anchor_base_to_endpoint: np.ndarray,
    *,
    playback_speed: float,
    frequency_hz: float,
    max_command_translation_m: float,
    max_command_rotation_rad: float,
    reverse: bool = False,
) -> ReplayPlan:
    """Apply C/U and E/P extrinsics, anchor, resample, and safety-check."""
    anchor = validate_rigid_transform(anchor_base_to_endpoint, "replay_anchor")
    source_relative = np.linalg.inv(episode.controller_poses[0]) @ episode.controller_poses
    endpoint_to_controller = config.endpoint_to_controller
    controller_to_endpoint = np.linalg.inv(endpoint_to_controller)
    endpoint_relative = endpoint_to_controller @ source_relative @ controller_to_endpoint
    key_targets = anchor @ endpoint_relative

    wall_duration_s = float(episode.timestamps_s[-1] / playback_speed)
    sample_count = max(2, int(np.ceil(wall_duration_s * frequency_hz)) + 1)
    target_times = np.linspace(0.0, wall_duration_s, sample_count, dtype=np.float64)
    source_times = np.minimum(target_times * playback_speed, episode.timestamps_s[-1])
    endpoint_targets = _resample_transforms(
        episode.timestamps_s, key_targets, source_times
    )
    grip_indices = np.searchsorted(
        episode.timestamps_s, source_times, side="right"
    ) - 1
    grip_indices = np.clip(grip_indices, 0, len(episode.absolute_actions) - 1)
    gripper_values = episode.absolute_actions[grip_indices, 9].copy()
    if reverse:
        endpoint_targets = endpoint_targets[::-1].copy()
        gripper_values = gripper_values[::-1].copy()
        source_times = source_times[::-1].copy()

    target_deltas = np.linalg.inv(endpoint_targets[:-1]) @ endpoint_targets[1:]
    step_translation, step_rotation = _motion_magnitudes(target_deltas)
    max_step_translation = float(np.max(step_translation))
    max_step_rotation = float(np.max(step_rotation))
    if max_step_translation > max_command_translation_m:
        raise ValueError(
            f"Resampled endpoint step {max_step_translation:.4f}m exceeds "
            f"{max_command_translation_m:.4f}m; lower --playback-speed."
        )
    if max_step_rotation > max_command_rotation_rad:
        raise ValueError(
            f"Resampled endpoint rotation step {max_step_rotation:.4f}rad exceeds "
            f"{max_command_rotation_rad:.4f}rad; lower --playback-speed."
        )
    for index, target in enumerate(endpoint_targets):
        validate_workspace_target(target, config.axis, label=f"replay[{index}]")

    anchor_relative = np.linalg.inv(anchor) @ endpoint_targets
    anchor_translation, anchor_rotation = _motion_magnitudes(anchor_relative)
    return ReplayPlan(
        target_times_s=target_times,
        source_times_s=source_times,
        endpoint_targets=endpoint_targets,
        gripper_values=gripper_values,
        max_step_translation_m=max_step_translation,
        max_step_rotation_rad=max_step_rotation,
        max_anchor_translation_m=float(np.max(anchor_translation)),
        max_anchor_rotation_rad=float(np.max(anchor_rotation)),
    )


def _print_episode_summary(
    episode: UmiEpisode,
    source_max_translation_m: float,
    source_max_rotation_rad: float,
) -> None:
    xyz = episode.training_step_actions[:, :3]
    rotation = R.from_matrix(
        pose_to_SE3(
            np.zeros_like(xyz), episode.training_step_actions[:, 3:9]
        )[:, :3, :3]
    ).magnitude()
    print("\n=== UMI episode -> training delta ===")
    print(
        f"episode={episode.episode_index:06d} frames="
        f"{episode.frame_indices[0]}..{episode.frame_indices[-1]} "
        f"count={len(episode.frame_indices)} duration={episode.timestamps_s[-1]:.3f}s"
    )
    print(
        "训练一步动作: D_C[t]=inv(T_H_C[t])@T_H_C[t+1]；"
        "rot6d 重新编码，gripper 保持下一帧绝对值。"
    )
    print(
        f"source max step: translation={1000.0 * source_max_translation_m:.2f}mm "
        f"rotation={source_max_rotation_rad:.4f}rad"
    )
    print(
        f"training delta xyz range: min={xyz.min(axis=0).round(6).tolist()} "
        f"max={xyz.max(axis=0).round(6).tolist()}"
    )
    print(f"training delta rotation max={float(np.max(rotation)):.4f}rad")
    print(
        f"raw left Oculus invalid frames={episode.tracking_invalid_frames}; "
        f"gripper range=[{episode.absolute_actions[:, 9].min():.3f}, "
        f"{episode.absolute_actions[:, 9].max():.3f}]"
    )


def _print_plan(
    plan: ReplayPlan,
    anchor: np.ndarray,
    *,
    playback_speed: float,
    reverse: bool,
) -> None:
    xyz = plan.endpoint_targets[:, :3, 3]
    print("\n=== Franka replay plan ===")
    print(f"anchor E: {_format_pose(anchor)}")
    print(
        f"direction={'reverse/return' if reverse else 'forward'} "
        f"playback_speed={playback_speed:.3f}x duration="
        f"{plan.target_times_s[-1]:.2f}s commands={len(plan.target_times_s)}"
    )
    print(
        f"target xyz min={xyz.min(axis=0).round(6).tolist()} "
        f"max={xyz.max(axis=0).round(6).tolist()}"
    )
    print(
        f"max command step={1000.0 * plan.max_step_translation_m:.2f}mm, "
        f"{plan.max_step_rotation_rad:.4f}rad"
    )
    print(
        f"max displacement from anchor={1000.0 * plan.max_anchor_translation_m:.1f}mm, "
        f"{plan.max_anchor_rotation_rad:.4f}rad"
    )


def _policy_gripper_to_opening(value: float, config: ReplayConfig) -> float:
    ratio = (value - config.policy_gripper_closed_value) / (
        config.policy_gripper_open_value - config.policy_gripper_closed_value
    )
    return float(np.clip(ratio, 0.0, 1.0))


def _controller_tracking_error(
    actual: np.ndarray,
    commanded: np.ndarray,
) -> tuple[float, float]:
    """Return endpoint position and rotation tracking errors."""
    position_error_m = float(np.linalg.norm(actual[:3, 3] - commanded[:3, 3]))
    rotation_error_rad = float(
        R.from_matrix(actual[:3, :3].T @ commanded[:3, :3]).magnitude()
    )
    return position_error_m, rotation_error_rad


def _run_hardware_replay(
    args: argparse.Namespace,
    episode: UmiEpisode,
    config: ReplayConfig,
) -> None:
    """Anchor to the live endpoint and stream the checked replay plan."""
    import geometry_msgs.msg as geometry_msgs
    import rospy
    from dynamic_reconfigure.client import Client as ReconfigureClient
    from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState

    master_process = None
    launch_process = None
    pose_publisher = None
    gripper = None
    monitor = FrankaStateMonitor()
    interrupted = False

    def publish_transform(transform: np.ndarray) -> None:
        pose = transform_to_pose_xyzw(transform)
        message = geometry_msgs.PoseStamped()
        message.header.frame_id = "0"
        message.header.stamp = rospy.Time.now()
        message.pose.position = geometry_msgs.Point(*pose[:3])
        message.pose.orientation = geometry_msgs.Quaternion(*pose[3:])
        pose_publisher.publish(message)

    def live_transform(*, commanded: np.ndarray | None = None) -> np.ndarray:
        pose, age_s, errors, contact = monitor.snapshot()
        transform = pose_xyzw_to_transform(pose)
        if age_s > args.state_timeout_s:
            raise RuntimeError(f"Franka state is stale ({age_s:.3f}s).")
        if errors:
            raise RuntimeError(f"Franka current_errors={list(errors)}")
        if args.abort_on_contact and contact:
            publish_transform(transform)
            raise RuntimeError("Franka contact/collision detected; holding live pose.")
        if commanded is not None:
            position_error = float(
                np.linalg.norm(transform[:3, 3] - commanded[:3, 3])
            )
            rotation_error = float(
                R.from_matrix(transform[:3, :3].T @ commanded[:3, :3]).magnitude()
            )
            if position_error > args.max_tracking_error_m:
                publish_transform(transform)
                raise RuntimeError(
                    f"Franka tracking error {position_error:.4f}m exceeds "
                    f"{args.max_tracking_error_m:.4f}m."
                )
            if rotation_error > args.max_tracking_rotation_rad:
                publish_transform(transform)
                raise RuntimeError(
                    f"Franka rotation tracking error {rotation_error:.4f}rad exceeds "
                    f"{args.max_tracking_rotation_rad:.4f}rad."
                )
        return transform

    def move_cartesian(
        live: np.ndarray,
        target: np.ndarray,
        *,
        label: str,
        max_translation_m: float,
        max_rotation_rad: float,
        linear_speed_m_s: float,
        angular_speed_rad_s: float,
    ) -> None:
        """Slowly interpolate from a live endpoint pose to one checked target."""
        delta = np.linalg.inv(live) @ target
        translation, rotation = _motion_magnitudes(delta[None])
        distance = float(translation[0])
        angle = float(rotation[0])
        if distance > max_translation_m:
            raise RuntimeError(
                f"Current pose is {distance:.4f}m from {label}; "
                f"limit is {max_translation_m:.4f}m."
            )
        if angle > max_rotation_rad:
            raise RuntimeError(
                f"Current rotation is {angle:.4f}rad from {label}; "
                f"limit is {max_rotation_rad:.4f}rad."
            )
        duration = max(
            distance / linear_speed_m_s,
            angle / angular_speed_rad_s,
            1.0 / args.frequency_hz,
        )
        count = max(1, int(np.ceil(duration * args.frequency_hz)))
        query = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)[1:]
        path = _resample_transforms(
            np.array([0.0, 1.0]),
            np.stack([live, target]),
            query,
        )
        validate_workspace_target(live, config.axis, label=f"{label}_live_start")
        for index, waypoint in enumerate(path):
            validate_workspace_target(
                waypoint,
                config.axis,
                label=f"{label}_waypoint_{index + 1}",
            )
        print(
            f"{label}: distance={1000.0 * distance:.1f}mm "
            f"rotation={angle:.4f}rad duration={duration:.2f}s"
        )
        next_tick = time.monotonic()
        last = live
        for waypoint in path:
            live_transform(commanded=last)
            publish_transform(waypoint)
            last = waypoint
            next_tick += 1.0 / args.frequency_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))

    try:
        master_process = _start_ros_master()
        rospy.init_node("umi_episode_franka_replay", anonymous=True)
        pose_publisher = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geometry_msgs.PoseStamped,
            queue_size=10,
        )
        recovery_publisher = rospy.Publisher(
            "/franka_control/error_recovery/goal",
            ErrorRecoveryActionGoal,
            queue_size=1,
        )
        rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            monitor.callback,
            queue_size=1,
        )

        print(f"启动 Franka 阻抗控制器: robot_ip={args.robot_ip}")
        launch_process = subprocess.Popen(
            [
                "roslaunch",
                "serl_franka_controllers",
                "impedance.launch",
                f"robot_ip:={args.robot_ip}",
                "load_gripper:=false",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )
        if not monitor.wait(args.startup_timeout_s):
            raise TimeoutError(
                "Timed out waiting for /franka_state_controller/franka_states."
            )
        deadline = time.monotonic() + args.startup_timeout_s
        while pose_publisher.get_num_connections() == 0:
            if launch_process.poll() is not None:
                raise RuntimeError("Franka roslaunch exited during startup.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Impedance equilibrium-pose subscriber is not ready.")
            time.sleep(0.1)

        if config.axis.compliance_params:
            reconfigure = ReconfigureClient(
                "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node",
                timeout=args.startup_timeout_s,
            )
            applied_compliance = reconfigure.update_configuration(
                config.axis.compliance_params
            )
            print(
                "已加载部署配置中的笛卡尔阻抗参数："
                f"target_filter_alpha="
                f"{float(applied_compliance['target_filter_alpha']):.3f}。"
            )

        live_at_start = live_transform()
        if args.anchor_file is not None:
            anchor = load_anchor_file(args.anchor_file, episode.episode_index)
            print(f"使用保存的 replay anchor: {args.anchor_file}")
        else:
            anchor = live_at_start
            anchor_path = args.save_anchor_file
            if anchor_path is None:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                anchor_path = Path(
                    f"logs/umi_replay/episode_{episode.episode_index:06d}_"
                    f"anchor_{stamp}.json"
                )
            save_anchor_file(anchor_path, anchor, episode)
            print(f"已在运动前保存实时 replay anchor: {anchor_path}")

        if args.move_to_anchor_only:
            validate_workspace_target(
                anchor,
                config.axis,
                label="saved_replay_anchor",
            )
            relative = np.linalg.inv(live_at_start) @ anchor
            translation, rotation = _motion_magnitudes(relative[None])
            print("\n=== Direct move to saved replay anchor ===")
            print(f"current E: {_format_pose(live_at_start)}")
            print(f"target E:  {_format_pose(anchor)}")
            print(
                f"direct Cartesian displacement={1000.0 * float(translation[0]):.1f}mm, "
                f"rotation={float(rotation[0]):.4f}rad"
            )
            print(
                "只沿笛卡尔插值路径移动到保存的初始位姿；"
                "不会回放 episode；移动前会打开 OmniPicker，到达后会再次确认打开。"
            )
            _confirm_exact(
                "确认夹爪可安全打开、急停可用，且到目标位姿的整条直线路径无人无障碍。",
                "MOVE2",
            )

            from training.envs.realworld.common.gripper.omnipicker_gripper import (
                OmniPickerGripper,
            )

            gripper = OmniPickerGripper(**config.endpoint_config)

            def stream_gripper_open(stage: str, hold_pose: np.ndarray) -> None:
                command_period_s = 1.0 / 20.0
                command_count = max(1, int(1.5 / command_period_s))
                for _ in range(command_count):
                    live_transform(commanded=hold_pose)
                    publish_transform(hold_pose)
                    gripper.open(speed=1.0)
                    time.sleep(command_period_s)
                status = gripper.get_status()
                if not gripper.is_ready():
                    raise RuntimeError(
                        f"OmniPicker did not become ready while opening ({stage}); "
                        f"device_online={status.get('device_online')}, "
                        f"sender_alive={status.get('sender_alive')}, "
                        f"receiver_alive={status.get('receiver_alive')}, "
                        f"disconnected={status.get('disconnected')}."
                    )
                print(
                    f"OmniPicker 已发送全开命令（{stage}），"
                    f"feedback_position={status.get('position', 'unknown')}。"
                )

            recovery_publisher.publish(ErrorRecoveryActionGoal())
            stream_gripper_open("移动前", live_at_start)
            move_cartesian(
                live_at_start,
                anchor,
                label="DIRECT_TO_ANCHOR",
                max_translation_m=args.direct_max_translation_m,
                max_rotation_rad=args.direct_max_rotation_rad,
                linear_speed_m_s=args.direct_linear_speed_m_s,
                angular_speed_rad_s=args.direct_angular_speed_rad_s,
            )
            for _ in range(max(1, int(args.settle_seconds * args.frequency_hz))):
                live_transform(commanded=anchor)
                publish_transform(anchor)
                time.sleep(1.0 / args.frequency_hz)
            stream_gripper_open("到达后", anchor)
            final_live = live_transform(commanded=anchor)
            final_position_error = float(
                np.linalg.norm(final_live[:3, 3] - anchor[:3, 3])
            )
            final_rotation_error = float(
                R.from_matrix(final_live[:3, :3].T @ anchor[:3, :3]).magnitude()
            )
            print(
                f"已移动到保存的 episode {episode.episode_index} 初始位姿："
                f"position_error={1000.0 * final_position_error:.2f}mm, "
                f"rotation_error={final_rotation_error:.4f}rad"
            )
            return

        plan = build_replay_plan(
            episode,
            config,
            anchor,
            playback_speed=args.playback_speed,
            frequency_hz=args.frequency_hz,
            max_command_translation_m=args.max_command_translation_m,
            max_command_rotation_rad=args.max_command_rotation_rad,
            reverse=args.reverse,
        )
        _print_plan(
            plan,
            anchor,
            playback_speed=args.playback_speed,
            reverse=args.reverse,
        )
        if args.anchor_file is None:
            print("轨迹锚定到当前实时 E 位姿，不会移动到 YAML reset_ee_pose。")
        else:
            print("轨迹使用保存的原始 anchor；当前位姿只允许小误差安全接入。")
        if args.with_gripper:
            print("警告：--with-gripper 将按照数据中的绝对夹爪动作控制 OmniPicker。")
            _confirm_exact("确认夹爪内部无异物且允许夹取。", "GRIPPER")
        _confirm_exact(
            "确认急停可用、完整目标范围无人且桌面高度与示教一致。",
            f"{'RETURN' if args.reverse else 'REPLAY'}{episode.episode_index}",
        )

        if args.with_gripper:
            from training.envs.realworld.common.gripper.omnipicker_gripper import (
                OmniPickerGripper,
            )

            gripper = OmniPickerGripper(**config.endpoint_config)

        recovery_publisher.publish(ErrorRecoveryActionGoal())
        first_target = plan.endpoint_targets[0]
        move_cartesian(
            live_at_start,
            first_target,
            label="安全接入首目标",
            max_translation_m=args.max_join_translation_m,
            max_rotation_rad=args.max_join_rotation_rad,
            linear_speed_m_s=args.join_linear_speed_m_s,
            angular_speed_rad_s=args.join_angular_speed_rad_s,
        )
        for _ in range(max(1, int(args.pre_hold_seconds * args.frequency_hz))):
            live_transform()
            publish_transform(first_target)
            time.sleep(1.0 / args.frequency_hz)

        print(
            f"开始 {'反向返回' if args.reverse else 'replay'}；"
            "Ctrl-C 将立即保持实时位姿并停止。"
        )
        next_tick = time.monotonic()
        for index, (target, gripper_value) in enumerate(
            zip(plan.endpoint_targets, plan.gripper_values)
        ):
            publish_transform(target)
            if gripper is not None:
                gripper.command_position(
                    _policy_gripper_to_opening(float(gripper_value), config)
                )
            next_tick += 1.0 / args.frequency_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))
            live = live_transform(commanded=target)
            position_error_m, rotation_error_rad = _controller_tracking_error(
                live,
                target,
            )
            print(
                f"  {'return' if args.reverse else 'replay'} "
                f"{index + 1}/{len(plan.endpoint_targets)} "
                f"source_t={plan.source_times_s[index]:.2f}s "
                f"controller_error: position_error="
                f"{1000.0 * position_error_m:.2f}mm "
                f"rotation_error={rotation_error_rad:.4f}rad "
                f"grip={gripper_value:.3f}"
            )

        final_target = plan.endpoint_targets[-1]
        for _ in range(max(1, int(args.settle_seconds * args.frequency_hz))):
            live_transform(commanded=final_target)
            publish_transform(final_target)
            time.sleep(1.0 / args.frequency_hz)
        final_live = live_transform(commanded=final_target)
        final_position_error = float(
            np.linalg.norm(final_live[:3, 3] - final_target[:3, 3])
        )
        final_rotation_error = float(
            R.from_matrix(
                final_live[:3, :3].T @ final_target[:3, :3]
            ).magnitude()
        )
        print(
            f"{'反向返回完成；机械臂回到原 anchor' if args.reverse else 'Replay 完成；机械臂停在 episode 末端'}："
            f"position_error={1000.0 * final_position_error:.2f}mm, "
            f"rotation_error={final_rotation_error:.4f}rad"
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nReplay 被操作员终止；保持实时位姿。")
        if pose_publisher is not None and monitor.wait(0.0):
            try:
                publish_transform(live_transform())
                time.sleep(0.2)
            except Exception:
                pass
    finally:
        if gripper is not None:
            try:
                gripper.cleanup()
            except Exception as exc:
                print(f"WARN: OmniPicker cleanup failed: {exc}")
        try:
            rospy.signal_shutdown("UMI replay finished")
        except Exception:
            pass
        _process_group_stop(launch_process)
        _process_group_stop(master_process)
        print("Franka/ROS replay 进程已清理。")
    if interrupted:
        raise SystemExit(130)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--episode-index", type=int, default=2)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Inclusive final source frame (default: episode end).",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--robot-ip", default=os.environ.get("FRANKA_ROBOT_IP", DEFAULT_ROBOT_IP)
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.20,
        help=(
            "Fraction of recorded speed in (0, 1]; 1.0 follows the recorded "
            "timestamps and 0.20 replays five times slower."
        ),
    )
    parser.add_argument("--frequency-hz", type=float, default=30.0)
    parser.add_argument("--max-source-translation-m", type=float, default=0.012)
    parser.add_argument("--max-source-rotation-rad", type=float, default=0.08)
    parser.add_argument("--max-command-translation-m", type=float, default=0.003)
    parser.add_argument("--max-command-rotation-rad", type=float, default=0.02)
    parser.add_argument("--max-tracking-error-m", type=float, default=0.025)
    parser.add_argument("--max-tracking-rotation-rad", type=float, default=0.12)
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.5)
    parser.add_argument("--pre-hold-seconds", type=float, default=1.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--anchor-file",
        type=Path,
        default=None,
        help="Reuse a saved initial Franka anchor instead of the current pose.",
    )
    parser.add_argument(
        "--save-anchor-file",
        type=Path,
        default=None,
        help="Where to save the live anchor before forward execution.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Follow the selected source frames backwards; requires --anchor-file.",
    )
    parser.add_argument(
        "--move-to-anchor-only",
        action="store_true",
        help=(
            "Move directly from the live endpoint pose to --anchor-file without "
            "replaying the episode; requires --execute."
        ),
    )
    parser.add_argument("--max-join-translation-m", type=float, default=0.030)
    parser.add_argument("--max-join-rotation-rad", type=float, default=0.15)
    parser.add_argument("--join-linear-speed-m-s", type=float, default=0.010)
    parser.add_argument("--join-angular-speed-rad-s", type=float, default=0.10)
    parser.add_argument("--direct-max-translation-m", type=float, default=0.25)
    parser.add_argument("--direct-max-rotation-rad", type=float, default=0.50)
    parser.add_argument("--direct-linear-speed-m-s", type=float, default=0.010)
    parser.add_argument("--direct-angular-speed-rad-s", type=float, default=0.10)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Control the real Franka. Without this flag only an offline plan is built.",
    )
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="Also replay absolute OmniPicker commands (requires --execute).",
    )
    parser.add_argument(
        "--allow-invalid-tracking",
        action="store_true",
        help="Allow raw rows marked invalid by Oculus diagnostics (not recommended).",
    )
    parser.add_argument(
        "--no-abort-on-contact",
        dest="abort_on_contact",
        action="store_false",
        help="Do not abort on Franka contact/collision flags (not recommended).",
    )
    parser.set_defaults(abort_on_contact=True)
    args = parser.parse_args()
    if args.episode_index < 0:
        parser.error("--episode-index must be non-negative.")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative.")
    if args.end_frame is not None and args.end_frame <= args.start_frame:
        parser.error("--end-frame must be greater than --start-frame.")
    if not 0.0 < args.playback_speed <= 1.0:
        parser.error("--playback-speed must be in (0, 1.0].")
    for name in (
        "frequency_hz",
        "max_source_translation_m",
        "max_source_rotation_rad",
        "max_command_translation_m",
        "max_command_rotation_rad",
        "max_tracking_error_m",
        "max_tracking_rotation_rad",
        "startup_timeout_s",
        "state_timeout_s",
        "pre_hold_seconds",
        "settle_seconds",
        "max_join_translation_m",
        "max_join_rotation_rad",
        "join_linear_speed_m_s",
        "join_angular_speed_rad_s",
        "direct_max_translation_m",
        "direct_max_rotation_rad",
        "direct_linear_speed_m_s",
        "direct_angular_speed_rad_s",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.with_gripper and not args.execute:
        parser.error("--with-gripper requires --execute.")
    if args.reverse and args.anchor_file is None:
        parser.error("--reverse requires --anchor-file.")
    if args.move_to_anchor_only and not args.execute:
        parser.error("--move-to-anchor-only requires --execute.")
    if args.move_to_anchor_only and args.anchor_file is None:
        parser.error("--move-to-anchor-only requires --anchor-file.")
    if args.move_to_anchor_only and args.reverse:
        parser.error("--move-to-anchor-only and --reverse are mutually exclusive.")
    if args.move_to_anchor_only and args.with_gripper:
        parser.error(
            "--move-to-anchor-only opens the gripper automatically; "
            "do not also pass --with-gripper."
        )
    if args.anchor_file is not None and args.save_anchor_file is not None:
        parser.error("--anchor-file and --save-anchor-file are mutually exclusive.")
    return args


def main() -> int:
    """Build an offline plan or execute a confirmed hardware replay."""
    args = _parse_args()
    config = load_replay_config(args.config)
    episode = load_umi_episode(
        args.dataset,
        args.source_dataset,
        args.episode_index,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    source_translation, source_rotation = validate_source_deltas(
        episode,
        max_translation_m=args.max_source_translation_m,
        max_rotation_rad=args.max_source_rotation_rad,
        allow_invalid_tracking=args.allow_invalid_tracking,
    )
    _print_episode_summary(episode, source_translation, source_rotation)

    if not args.execute:
        anchor = (
            load_anchor_file(args.anchor_file, episode.episode_index)
            if args.anchor_file is not None
            else config.axis.reset_base_to_ee
        )
        plan = build_replay_plan(
            episode,
            config,
            anchor,
            playback_speed=args.playback_speed,
            frequency_hz=args.frequency_hz,
            max_command_translation_m=args.max_command_translation_m,
            max_command_rotation_rad=args.max_command_rotation_rad,
            reverse=args.reverse,
        )
        _print_plan(
            plan,
            anchor,
            playback_speed=args.playback_speed,
            reverse=args.reverse,
        )
        anchor_source = (
            f"保存的 anchor {args.anchor_file}"
            if args.anchor_file is not None
            else "YAML reset"
        )
        print(
            f"Dry-run 完成：使用{anchor_source}作为离线 anchor；"
            "没有连接或控制机器人。\n"
            "真实执行会重新读取启动时的 Franka 当前位姿并再次检查整条轨迹。"
        )
        return 0

    _confirm_exact(
        "此操作将启动真实 Franka；启动后仍会基于实时位置重新规划并再次确认。",
        "START",
    )
    _run_hardware_replay(args, episode, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
