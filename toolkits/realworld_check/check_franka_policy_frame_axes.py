#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Interactively verify the Franka policy-tool frame without loading a model.

The test reads ``reset_ee_pose`` and ``franka_tcp_to_policy_tool`` from the
single-Franka deployment config. It first moves the Franka endpoint ``E`` to
the configured reset pose, then tests ``+/-X``, ``+/-Y``, and ``+/-Z`` of the
policy/OmniPicker TCP frame ``P``. Every target is constructed as

``T_B_P_target = T_B_P_reset @ Trans_P(axis * distance)``

and converted back to the endpoint command with

``T_B_E_target = T_B_P_target @ inverse(T_E_P)``.

The program never opens or commands the gripper and never loads a policy.
Each outbound movement requires an exact operator confirmation and returns to
reset only after a second confirmation.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

DEFAULT_CONFIG = Path(
    "evaluations/realworld/env/realworld_single_franka_tcp_rot6d.yaml"
)
DEFAULT_ROBOT_IP = "172.16.1.1"
DEFAULT_DISTANCE_M = 0.05
MAX_DISTANCE_M = 0.05
MAX_LINEAR_SPEED_M_S = 0.05

AXIS_DELTAS: dict[str, np.ndarray] = {
    "x+": np.array([1.0, 0.0, 0.0]),
    "x-": np.array([-1.0, 0.0, 0.0]),
    "y+": np.array([0.0, 1.0, 0.0]),
    "y-": np.array([0.0, -1.0, 0.0]),
    "z+": np.array([0.0, 0.0, 1.0]),
    "z-": np.array([0.0, 0.0, -1.0]),
}


@dataclass(frozen=True)
class PolicyFrameAxisConfig:
    """Resolved motion and safety values read from the deployment YAML."""

    reset_base_to_ee: np.ndarray
    ee_to_policy: np.ndarray
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    orientation_center: np.ndarray
    frequency_hz: float
    compliance_params: dict[str, float]


@dataclass(frozen=True)
class PolicyAxisTarget:
    """One local-P translation and its equivalent Franka endpoint target."""

    label: str
    policy_delta_m: np.ndarray
    base_delta_m: np.ndarray
    base_to_ee: np.ndarray


def pose_euler_to_transform(pose: Sequence[float]) -> np.ndarray:
    """Convert ``[xyz, euler_xyz]`` into a homogeneous transform."""
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (6,) or not np.all(np.isfinite(pose_array)):
        raise ValueError(f"Expected a finite 6-D xyz/euler pose, got {pose!r}.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("xyz", pose_array[3:]).as_matrix()
    transform[:3, 3] = pose_array[:3]
    return transform


def pose_xyzw_to_transform(pose: Sequence[float]) -> np.ndarray:
    """Convert ``[xyz, quat_xyzw]`` into a homogeneous transform."""
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (7,) or not np.all(np.isfinite(pose_array)):
        raise ValueError(f"Expected a finite 7-D xyz/quaternion pose, got {pose!r}.")
    quaternion_norm = float(np.linalg.norm(pose_array[3:]))
    if quaternion_norm < 1e-8:
        raise ValueError("Quaternion norm is zero.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(pose_array[3:] / quaternion_norm).as_matrix()
    transform[:3, 3] = pose_array[:3]
    return transform


def transform_to_pose_xyzw(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to ``[xyz, quat_xyzw]``."""
    transform = validate_rigid_transform(transform, "transform")
    return np.concatenate(
        [transform[:3, 3], R.from_matrix(transform[:3, :3]).as_quat()]
    )


def validate_rigid_transform(transform: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite, proper homogeneous transform."""
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 4x4 matrix, got {value!r}.")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must end with [0, 0, 0, 1].")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not +1.")
    return value


def load_axis_test_config(path: Path) -> PolicyFrameAxisConfig:
    """Load the subset of deployment configuration used by this test."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        cfg = raw["override_cfg"]
        reset = pose_euler_to_transform(cfg["reset_ee_pose"])
        ee_to_policy = validate_rigid_transform(
            np.asarray(cfg["franka_tcp_to_policy_tool"], dtype=np.float64),
            "franka_tcp_to_policy_tool",
        )
        workspace_min = np.asarray(cfg["ee_pose_limit_min"], dtype=np.float64)
        workspace_max = np.asarray(cfg["ee_pose_limit_max"], dtype=np.float64)
        orientation_center = np.asarray(cfg["target_ee_pose"][3:], dtype=np.float64)
        frequency_hz = float(cfg["step_frequency"])
        compliance_params = {
            str(key): float(value)
            for key, value in cfg.get("compliance_param", {}).items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Franka axis-test config {path}: {exc}") from exc

    if workspace_min.shape != (6,) or workspace_max.shape != (6,):
        raise ValueError("Workspace limits must each contain six values.")
    if np.any(workspace_min > workspace_max):
        raise ValueError("Workspace minimum exceeds maximum.")
    if frequency_hz <= 0.0:
        raise ValueError("step_frequency must be positive.")
    return PolicyFrameAxisConfig(
        reset_base_to_ee=reset,
        ee_to_policy=ee_to_policy,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        orientation_center=orientation_center,
        frequency_hz=frequency_hz,
        compliance_params=compliance_params,
    )


def _euler_near_center(rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Return the xyz Euler branch nearest a configured orientation center."""
    euler = R.from_matrix(rotation).as_euler("xyz")
    return center + (euler - center + np.pi) % (2.0 * np.pi) - np.pi


def validate_workspace_target(
    base_to_ee: np.ndarray,
    config: PolicyFrameAxisConfig,
    *,
    label: str,
) -> None:
    """Reject a target outside the configured endpoint workspace."""
    base_to_ee = validate_rigid_transform(base_to_ee, label)
    xyz = base_to_ee[:3, 3]
    euler = _euler_near_center(base_to_ee[:3, :3], config.orientation_center)
    pose = np.concatenate([xyz, euler])
    if np.any(pose < config.workspace_min - 1e-9) or np.any(
        pose > config.workspace_max + 1e-9
    ):
        raise ValueError(
            f"{label} is outside the configured workspace: pose={pose.tolist()}, "
            f"min={config.workspace_min.tolist()}, "
            f"max={config.workspace_max.tolist()}"
        )


def build_policy_axis_targets(
    config: PolicyFrameAxisConfig,
    labels: Sequence[str],
    distance_m: float,
) -> list[PolicyAxisTarget]:
    """Build endpoint targets for local translations of the reset P frame."""
    if not 0.0 < distance_m <= MAX_DISTANCE_M:
        raise ValueError(
            f"distance_m must be in (0, {MAX_DISTANCE_M}], got {distance_m}."
        )
    validate_workspace_target(
        config.reset_base_to_ee,
        config,
        label="reset_ee_pose",
    )
    base_to_policy_reset = config.reset_base_to_ee @ config.ee_to_policy
    policy_to_ee = np.linalg.inv(config.ee_to_policy)
    targets: list[PolicyAxisTarget] = []
    for label in labels:
        if label not in AXIS_DELTAS:
            raise ValueError(f"Unknown policy axis {label!r}.")
        policy_delta = AXIS_DELTAS[label] * distance_m
        policy_motion = np.eye(4, dtype=np.float64)
        policy_motion[:3, 3] = policy_delta
        base_to_policy_target = base_to_policy_reset @ policy_motion
        base_to_ee_target = base_to_policy_target @ policy_to_ee
        validate_workspace_target(base_to_ee_target, config, label=label)
        targets.append(
            PolicyAxisTarget(
                label=label,
                policy_delta_m=policy_delta,
                base_delta_m=(
                    base_to_ee_target[:3, 3] - config.reset_base_to_ee[:3, 3]
                ),
                base_to_ee=base_to_ee_target,
            )
        )
    return targets


def interpolate_pose_xyzw(
    start_pose: Sequence[float],
    target_pose: Sequence[float],
    alphas: np.ndarray,
) -> np.ndarray:
    """Interpolate Cartesian position and quaternion over ``alphas``."""
    start = np.asarray(start_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if start.shape != (7,) or target.shape != (7,):
        raise ValueError("start_pose and target_pose must be 7-D xyzw poses.")
    if alphas.ndim != 1 or np.any(alphas < 0.0) or np.any(alphas > 1.0):
        raise ValueError("alphas must be a one-dimensional array in [0, 1].")
    positions = start[:3] + alphas[:, None] * (target[:3] - start[:3])
    rotations = Slerp(
        [0.0, 1.0],
        R.from_quat(np.stack([start[3:], target[3:]], axis=0)),
    )(alphas).as_quat()
    return np.concatenate([positions, rotations], axis=1)


class FrankaStateMonitor:
    """Thread-safe subset of the latest ``franka_msgs/FrankaState``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pose: np.ndarray | None = None
        self._received_monotonic = 0.0
        self._active_errors: tuple[str, ...] = ()
        self._contact = False

    def callback(self, message: Any) -> None:
        transform = np.asarray(message.O_T_EE, dtype=np.float64).reshape(4, 4).T
        pose = transform_to_pose_xyzw(transform)
        errors = tuple(
            slot
            for slot in getattr(message.current_errors, "__slots__", ())
            if bool(getattr(message.current_errors, slot))
        )
        contact = bool(
            np.any(np.asarray(message.cartesian_contact, dtype=np.float64) > 0.5)
            or np.any(np.asarray(message.cartesian_collision, dtype=np.float64) > 0.5)
            or np.any(np.asarray(message.joint_collision, dtype=np.float64) > 0.5)
        )
        with self._lock:
            self._pose = pose
            self._received_monotonic = time.monotonic()
            self._active_errors = errors
            self._contact = contact
            self._event.set()

    def wait(self, timeout_s: float) -> bool:
        """Wait until the first state message arrives."""
        return self._event.wait(timeout_s)

    def snapshot(self) -> tuple[np.ndarray, float, tuple[str, ...], bool]:
        """Return pose, age, active errors, and contact flag."""
        with self._lock:
            if self._pose is None:
                raise RuntimeError("No Franka state has been received.")
            return (
                self._pose.copy(),
                time.monotonic() - self._received_monotonic,
                self._active_errors,
                self._contact,
            )


def _process_group_stop(process: subprocess.Popen | None) -> None:
    """Stop a process group created by this program."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)


def _ros_master_is_online() -> bool:
    """Return whether the configured ROS 1 master responds."""
    try:
        import rosgraph

        rosgraph.Master("/franka_policy_frame_axes_probe").getPid()
        return True
    except Exception:
        return False


def _start_ros_master() -> subprocess.Popen | None:
    """Start roscore only when this shell does not already have one."""
    if _ros_master_is_online():
        return None
    process = subprocess.Popen(
        ["roscore"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("roscore exited before becoming ready.")
        if _ros_master_is_online():
            return process
        time.sleep(0.2)
    _process_group_stop(process)
    raise TimeoutError("Timed out waiting for roscore.")


def _confirm_exact(message: str, expected: str) -> None:
    """Require an exact interactive confirmation token."""
    try:
        response = input(
            f"{message}\n输入 {expected} 确认，其他输入将终止测试: "
        ).strip()
    except EOFError as exc:
        raise KeyboardInterrupt from exc
    if response != expected:
        raise KeyboardInterrupt


def _format_pose(transform: np.ndarray) -> str:
    """Format one endpoint transform as xyz and the nearby xyz Euler branch."""
    xyz = transform[:3, 3]
    euler = R.from_matrix(transform[:3, :3]).as_euler("xyz")
    return (
        f"xyz=[{xyz[0]:+.6f}, {xyz[1]:+.6f}, {xyz[2]:+.6f}] m, "
        f"euler_xyz=[{euler[0]:+.6f}, {euler[1]:+.6f}, {euler[2]:+.6f}] rad"
    )


def _print_plan(
    config: PolicyFrameAxisConfig,
    targets: Sequence[PolicyAxisTarget],
) -> None:
    """Print every local-P move in both P and Franka-base coordinates."""
    base_to_policy = config.reset_base_to_ee @ config.ee_to_policy
    print("\n=== Franka / OmniPicker P 坐标系测试计划 ===")
    print(f"Reset E: {_format_pose(config.reset_base_to_ee)}")
    print(
        "Reset P xyz: "
        f"[{base_to_policy[0, 3]:+.6f}, {base_to_policy[1, 3]:+.6f}, "
        f"{base_to_policy[2, 3]:+.6f}] m"
    )
    for target in targets:
        p_mm = 1000.0 * target.policy_delta_m
        b_mm = 1000.0 * target.base_delta_m
        print(
            f"  {target.label.upper():>2}: "
            f"delta_P=[{p_mm[0]:+.1f}, {p_mm[1]:+.1f}, {p_mm[2]:+.1f}] mm -> "
            f"delta_base=[{b_mm[0]:+.1f}, {b_mm[1]:+.1f}, {b_mm[2]:+.1f}] mm"
        )
    print("夹爪不会被打开、关闭或初始化；本程序只启动 Franka 笛卡尔阻抗控制器。")


def _run_hardware_test(
    args: argparse.Namespace,
    config: PolicyFrameAxisConfig,
    targets: Sequence[PolicyAxisTarget],
) -> None:
    """Connect ROS, reset the arm, and execute confirmed policy-axis moves."""
    import geometry_msgs.msg as geometry_msgs
    import rospy
    from dynamic_reconfigure.client import Client as ReconfigureClient
    from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState

    master_process: subprocess.Popen | None = None
    launch_process: subprocess.Popen | None = None
    monitor = FrankaStateMonitor()
    pose_publisher = None
    interrupted = False

    def publish_pose(pose: np.ndarray) -> None:
        message = geometry_msgs.PoseStamped()
        message.header.frame_id = "0"
        message.header.stamp = rospy.Time.now()
        message.pose.position = geometry_msgs.Point(*pose[:3])
        message.pose.orientation = geometry_msgs.Quaternion(*pose[3:])
        pose_publisher.publish(message)

    def assert_live_state() -> np.ndarray:
        pose, age_s, errors, contact = monitor.snapshot()
        if age_s > args.state_timeout_s:
            raise RuntimeError(f"Franka state is stale ({age_s:.3f}s).")
        if errors:
            raise RuntimeError(f"Franka current_errors={list(errors)}")
        if args.abort_on_contact and contact:
            publish_pose(pose)
            raise RuntimeError("Franka contact/collision detected; holding live pose.")
        return pose

    def move_to(target_transform: np.ndarray, name: str) -> np.ndarray:
        target_pose = transform_to_pose_xyzw(target_transform)
        start_pose = assert_live_state()
        translation_distance = float(np.linalg.norm(target_pose[:3] - start_pose[:3]))
        relative_rotation = R.from_quat(start_pose[3:]).inv() * R.from_quat(
            target_pose[3:]
        )
        rotation_distance = float(relative_rotation.magnitude())
        duration_s = max(
            translation_distance / args.linear_speed_m_s,
            rotation_distance / args.angular_speed_rad_s,
            1.0 / config.frequency_hz,
        )
        steps = max(1, int(math.ceil(duration_s * config.frequency_hz)))
        path = interpolate_pose_xyzw(
            start_pose,
            target_pose,
            np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[1:],
        )
        print(
            f"{name}: distance={translation_distance:.4f}m, "
            f"rotation={rotation_distance:.4f}rad, duration={duration_s:.2f}s, "
            f"steps={steps}"
        )
        next_tick = time.monotonic()
        for pose in path:
            assert_live_state()
            publish_pose(pose)
            next_tick += 1.0 / config.frequency_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))

        settle_steps = max(1, int(args.settle_seconds * config.frequency_hz))
        for _ in range(settle_steps):
            assert_live_state()
            publish_pose(target_pose)
            time.sleep(1.0 / config.frequency_hz)

        actual_pose = assert_live_state()
        position_error = float(np.linalg.norm(actual_pose[:3] - target_pose[:3]))
        rotation_error = float(
            (
                R.from_quat(actual_pose[3:]).inv() * R.from_quat(target_pose[3:])
            ).magnitude()
        )
        print(
            f"{name} reached: position_error={1000.0 * position_error:.2f}mm, "
            f"rotation_error={rotation_error:.4f}rad"
        )
        if position_error > args.position_tolerance_m:
            publish_pose(actual_pose)
            raise RuntimeError(
                f"{name} position error {position_error:.4f}m exceeds "
                f"{args.position_tolerance_m:.4f}m."
            )
        if rotation_error > args.rotation_tolerance_rad:
            publish_pose(actual_pose)
            raise RuntimeError(
                f"{name} rotation error {rotation_error:.4f}rad exceeds "
                f"{args.rotation_tolerance_rad:.4f}rad."
            )
        return actual_pose

    try:
        master_process = _start_ros_master()
        rospy.init_node("franka_policy_frame_axes_test", anonymous=True)
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
                raise TimeoutError(
                    "Impedance equilibrium-pose subscriber is not ready."
                )
            time.sleep(0.1)

        if config.compliance_params:
            reconfigure = ReconfigureClient(
                "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node",
                timeout=args.startup_timeout_s,
            )
            reconfigure.update_configuration(config.compliance_params)
            print("已加载部署配置中的笛卡尔阻抗参数。")

        live_pose = assert_live_state()
        print(f"当前 E: {_format_pose(pose_xyzw_to_transform(live_pose))}")
        _confirm_exact(
            "即将从当前位置缓慢移动到配置的 reset_ee_pose。确认急停可用且工作空间无人。",
            "RESET",
        )
        recovery_publisher.publish(ErrorRecoveryActionGoal())
        time.sleep(0.5)
        move_to(config.reset_base_to_ee, "RESET")

        base_to_policy_reset = config.reset_base_to_ee @ config.ee_to_policy
        for target in targets:
            base_mm = 1000.0 * target.base_delta_m
            _confirm_exact(
                f"准备执行 P{target.label.upper()} 5cm；对应 base 位移 "
                f"[{base_mm[0]:+.1f}, {base_mm[1]:+.1f}, {base_mm[2]:+.1f}]mm。",
                target.label.upper(),
            )
            recovery_publisher.publish(ErrorRecoveryActionGoal())
            time.sleep(0.2)
            actual_pose = move_to(target.base_to_ee, f"P{target.label.upper()}")
            actual_base_to_policy = (
                pose_xyzw_to_transform(actual_pose) @ config.ee_to_policy
            )
            measured_delta = (
                np.linalg.inv(base_to_policy_reset) @ actual_base_to_policy
            )[:3, 3]
            measured_mm = 1000.0 * measured_delta
            print(
                "按当前 T_E^P 计算的实际 delta_P="
                f"[{measured_mm[0]:+.1f}, {measured_mm[1]:+.1f}, "
                f"{measured_mm[2]:+.1f}]mm。请以实物/RViz 方向判断坐标轴是否正确。"
            )
            _confirm_exact(
                "观察并记录实际运动方向后，返回 reset。",
                "RETURN",
            )
            move_to(config.reset_base_to_ee, f"RETURN_FROM_P{target.label.upper()}")

        print("六个 P 坐标轴方向测试完成；机械臂已返回 reset 位姿。")
    except KeyboardInterrupt:
        interrupted = True
        print("\n测试被操作员终止；不会自动执行额外运动。")
        if pose_publisher is not None and monitor.wait(0.0):
            try:
                live_pose, _, _, _ = monitor.snapshot()
                publish_pose(live_pose)
                time.sleep(0.2)
            except Exception:
                pass
    finally:
        try:
            rospy.signal_shutdown("policy-frame axis test finished")
        except Exception:
            pass
        _process_group_stop(launch_process)
        _process_group_stop(master_process)
        print("Franka/ROS 测试进程已清理。")
    if interrupted:
        raise SystemExit(130)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("FRANKA_ROBOT_IP", DEFAULT_ROBOT_IP),
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=tuple(AXIS_DELTAS),
        default=list(AXIS_DELTAS),
        help="Policy axes to test in order (default: all six).",
    )
    parser.add_argument("--distance-m", type=float, default=DEFAULT_DISTANCE_M)
    parser.add_argument("--linear-speed-m-s", type=float, default=0.025)
    parser.add_argument("--angular-speed-rad-s", type=float, default=0.30)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.5)
    parser.add_argument("--position-tolerance-m", type=float, default=0.012)
    parser.add_argument("--rotation-tolerance-rad", type=float, default=0.08)
    parser.add_argument(
        "--no-abort-on-contact",
        dest="abort_on_contact",
        action="store_false",
        help="Do not abort when Franka reports contact (not recommended).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print transforms and workspace checks without connecting hardware.",
    )
    parser.set_defaults(abort_on_contact=True)
    args = parser.parse_args()
    if args.linear_speed_m_s <= 0.0 or args.linear_speed_m_s > MAX_LINEAR_SPEED_M_S:
        parser.error(f"--linear-speed-m-s must be in (0, {MAX_LINEAR_SPEED_M_S}].")
    if args.angular_speed_rad_s <= 0.0:
        parser.error("--angular-speed-rad-s must be positive.")
    for name in (
        "settle_seconds",
        "startup_timeout_s",
        "state_timeout_s",
        "position_tolerance_m",
        "rotation_tolerance_rad",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    return args


def main() -> int:
    """Run the policy-frame axis verification CLI."""
    args = _parse_args()
    config = load_axis_test_config(args.config)
    targets = build_policy_axis_targets(config, args.axes, args.distance_m)
    _print_plan(config, targets)
    if args.dry_run:
        print("Dry run complete: all configured targets are inside the workspace.")
        return 0
    _confirm_exact(
        "此测试将控制真实 Franka，但不会加载模型或控制夹爪。",
        "START",
    )
    _run_hardware_test(args, config, targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
