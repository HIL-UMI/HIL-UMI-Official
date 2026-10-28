# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Single-Franka environment for absolute TCP rot6d policy actions.

The policy-facing layout is ``[xyz(3), rot6d(6), gripper(1)]``.  This UMI
dataset stores the Oculus controller frame ``C``, while the physical
acquisition TCP is ``U`` and the robot OmniPicker TCP is ``P``.  The fixed
action-frame transform is therefore

``T_E_C = T_E_P @ inverse(T_C_U)``.

Observations expose ``T_B_C = T_B_E @ T_E_C``.  Absolute policy targets are
converted back with ``T_B_E = T_B_C @ inverse(T_E_C)`` before dispatch.  A
fixed Oculus world transform is intentionally not needed because training
uses SE(3) body-frame action deltas, but the controller-to-tool extrinsic
``T_C_U`` does not cancel and must be configured.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from training.envs.realworld.common.scene_alignment import (
    ReferenceSceneAlignmentWebUI,
)
from training.utils.rot6d import matrix_to_rot6d, rot6d_to_matrix

from ..end_effectors.base import EndEffectorType
from ..franka_env import FrankaEnv, FrankaRobotConfig

if TYPE_CHECKING:
    from ..pose_trace import AsyncPoseTraceContext, FrankaPoseTraceRecorder

ACTION_DIM = 10
TCP_ROT6D_DIM = 9
STATE_ACTION_NAMES = (
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
)


def _identity_transform() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def policy_gripper_to_rlinf_action(
    value: float,
    *,
    closed_value: float,
    open_value: float,
) -> float:
    """Map the policy gripper scale to RLinf's ``[-1, 1]`` convention."""
    opening_fraction = (float(value) - closed_value) / (open_value - closed_value)
    return 2.0 * float(np.clip(opening_fraction, 0.0, 1.0)) - 1.0


class GripperCommandFilter:
    """Convert noisy absolute gripper predictions into stable open/close states.

    A Schmitt trigger supplies separate close/open thresholds. A transition is
    committed only after the candidate state persists for ``debounce_steps``
    consecutive control ticks.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        close_threshold: float,
        open_threshold: float,
        debounce_steps: int,
        closed_value: float,
        open_value: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.debounce_steps = int(debounce_steps)
        self.closed_value = float(closed_value)
        self.open_value = float(open_value)
        self.reset(open_state=True)

    def reset(self, *, open_state: bool) -> None:
        """Reset the latched state and discard an incomplete transition."""
        self._open_state = bool(open_state)
        self._pending_state: bool | None = None
        self._pending_steps = 0

    def apply(self, value: float) -> tuple[float, bool]:
        """Return the filtered policy-scale value and whether it toggled."""
        value = float(np.clip(value, 0.0, 1.0))
        if not self.enabled:
            return value, False

        candidate: bool | None = None
        if self._open_state and value <= self.close_threshold:
            candidate = False
        elif not self._open_state and value >= self.open_threshold:
            candidate = True

        if candidate is None:
            self._pending_state = None
            self._pending_steps = 0
        elif candidate == self._pending_state:
            self._pending_steps += 1
        else:
            self._pending_state = candidate
            self._pending_steps = 1

        changed = False
        if (
            self._pending_state is not None
            and self._pending_steps >= self.debounce_steps
        ):
            self._open_state = self._pending_state
            self._pending_state = None
            self._pending_steps = 0
            changed = True

        filtered = self.open_value if self._open_state else self.closed_value
        return filtered, changed


def validate_rigid_transform(
    transform: np.ndarray,
    name: str = "franka_tcp_to_policy_tool",
) -> np.ndarray:
    """Validate and return a homogeneous rigid transform.

    Args:
        transform: Array-like value reshapeable to ``(4, 4)``.

    Returns:
        A float64 ``(4, 4)`` transform.

    Raises:
        ValueError: If the value is non-finite or is not a proper SE(3)
            transform.
    """
    transform = np.asarray(transform, dtype=np.float64)
    if transform.size != 16:
        raise ValueError(f"{name} must contain 16 values, got shape {transform.shape}.")
    transform = transform.reshape(4, 4)
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be finite.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} must have last row [0, 0, 0, 1].")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must have determinant +1.")
    return transform


def pose_xyzw_to_transform(pose: np.ndarray) -> np.ndarray:
    """Convert ``[xyz, quat_xyzw]`` to a homogeneous transform."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Expected a finite 7D xyzw pose, got {pose!r}.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_pose_xyzw(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to ``[xyz, quat_xyzw]``."""
    transform = validate_rigid_transform(transform)
    return np.concatenate(
        [transform[:3, 3], R.from_matrix(transform[:3, :3]).as_quat()]
    ).astype(np.float64)


def clamp_tcp_step(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_rad: float,
) -> tuple[np.ndarray, bool, bool]:
    """Clamp one absolute TCP target relative to the live TCP pose.

    Translation is clipped by Euclidean norm.  Rotation follows the shortest
    SO(3) geodesic and is clipped by rotation-vector norm.
    """
    current = pose_xyzw_to_transform(current_pose)
    target = pose_xyzw_to_transform(target_pose)

    translation = target[:3, 3] - current[:3, 3]
    translation_norm = float(np.linalg.norm(translation))
    translation_clipped = translation_norm > max_translation_m
    if translation_clipped:
        translation *= max_translation_m / max(translation_norm, 1e-12)

    relative_rotation = current[:3, :3].T @ target[:3, :3]
    rotation_vector = R.from_matrix(relative_rotation).as_rotvec()
    rotation_norm = float(np.linalg.norm(rotation_vector))
    rotation_clipped = rotation_norm > max_rotation_rad
    if rotation_clipped:
        rotation_vector *= max_rotation_rad / max(rotation_norm, 1e-12)

    safe = np.eye(4, dtype=np.float64)
    safe[:3, 3] = current[:3, 3] + translation
    safe[:3, :3] = current[:3, :3] @ R.from_rotvec(rotation_vector).as_matrix()
    return (
        transform_to_pose_xyzw(safe),
        translation_clipped,
        rotation_clipped,
    )


def tcp_tracking_error(
    actual_pose: np.ndarray,
    commanded_pose: np.ndarray,
) -> tuple[float, float]:
    """Compute Cartesian controller tracking error.

    Args:
        actual_pose: Measured ``[xyz, quat_xyzw]`` endpoint pose.
        commanded_pose: Commanded ``[xyz, quat_xyzw]`` endpoint pose.

    Returns:
        Position error in metres and shortest-path rotation error in radians.
    """
    actual = pose_xyzw_to_transform(actual_pose)
    commanded = pose_xyzw_to_transform(commanded_pose)
    position_error_m = float(np.linalg.norm(actual[:3, 3] - commanded[:3, 3]))
    rotation_error_rad = float(
        R.from_matrix(actual[:3, :3].T @ commanded[:3, :3]).magnitude()
    )
    return position_error_m, rotation_error_rad


def render_state_action_curve(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    step_frequency: float,
    title: str,
) -> np.ndarray:
    """Render ten aligned state/action channels into one OpenCV image."""
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1:] != (ACTION_DIM,):
        raise ValueError(
            f"states must have shape [N, {ACTION_DIM}], got {states.shape}."
        )
    expected_shape = (states.shape[0], ACTION_DIM)
    if actions.shape != expected_shape:
        raise ValueError(
            f"actions must match states shape {expected_shape}, got {actions.shape}."
        )
    if states.shape[0] < 1:
        raise ValueError("At least one state/action sample is required.")
    if step_frequency <= 0:
        raise ValueError("step_frequency must be positive.")

    rows, columns = 5, 2
    panel_width, panel_height = 780, 300
    header_height = 100
    canvas = np.full(
        (header_height + rows * panel_height, columns * panel_width, 3),
        255,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        title,
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    duration_s = max(0.0, (states.shape[0] - 1) / step_frequency)
    cv2.putText(
        canvas,
        f"samples={states.shape[0]}  nominal_frequency={step_frequency:g} Hz  "
        f"duration={duration_s:.3f} s",
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    state_color = (210, 90, 20)
    action_color = (20, 120, 235)
    for channel, name in enumerate(STATE_ACTION_NAMES):
        row, column = divmod(channel, columns)
        origin_x = column * panel_width
        origin_y = header_height + row * panel_height
        plot_left = origin_x + 90
        plot_right = origin_x + panel_width - 25
        plot_top = origin_y + 45
        plot_bottom = origin_y + panel_height - 55

        combined = np.concatenate([states[:, channel], actions[:, channel]])
        finite_values = combined[np.isfinite(combined)]
        if finite_values.size:
            value_min = float(finite_values.min())
            value_max = float(finite_values.max())
        else:
            value_min, value_max = -1.0, 1.0
        value_range = value_max - value_min
        padding = max(0.05 * value_range, 1e-4)
        if value_range < 1e-12:
            padding = max(abs(value_min) * 0.05, 1e-3)
        value_min -= padding
        value_max += padding

        cv2.rectangle(
            canvas,
            (plot_left, plot_top),
            (plot_right, plot_bottom),
            (90, 90, 90),
            1,
        )
        cv2.line(
            canvas,
            (plot_left, (plot_top + plot_bottom) // 2),
            (plot_right, (plot_top + plot_bottom) // 2),
            (225, 225, 225),
            1,
        )
        cv2.putText(
            canvas,
            name,
            (origin_x + 20, origin_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        def draw_series(values: np.ndarray, color: tuple[int, int, int]) -> None:
            finite = np.isfinite(values)
            indices = np.flatnonzero(finite)
            if not indices.size:
                return
            split_points = np.flatnonzero(np.diff(indices) > 1) + 1
            for segment in np.split(indices, split_points):
                x = plot_left + np.rint(
                    segment * (plot_right - plot_left) / max(states.shape[0] - 1, 1)
                ).astype(np.int32)
                y = plot_bottom - np.rint(
                    (values[segment] - value_min)
                    * (plot_bottom - plot_top)
                    / (value_max - value_min)
                ).astype(np.int32)
                points = np.column_stack([x, y]).reshape(-1, 1, 2)
                if len(points) == 1:
                    cv2.circle(canvas, tuple(points[0, 0]), 2, color, -1)
                else:
                    cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)

        draw_series(states[:, channel], state_color)
        draw_series(actions[:, channel], action_color)
        cv2.putText(
            canvas,
            f"{value_max:.4g}",
            (origin_x + 8, plot_top + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{value_min:.4g}",
            (origin_x + 8, plot_bottom + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "0 s",
            (plot_left, plot_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{duration_s:.2f} s",
            (plot_right - 65, plot_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        legend_x = plot_right - 205
        cv2.line(
            canvas,
            (legend_x, origin_y + 25),
            (legend_x + 28, origin_y + 25),
            state_color,
            3,
        )
        cv2.putText(
            canvas,
            "state",
            (legend_x + 35, origin_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (legend_x + 100, origin_y + 25),
            (legend_x + 128, origin_y + 25),
            action_color,
            3,
        )
        cv2.putText(
            canvas,
            "action",
            (legend_x + 135, origin_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    return canvas


@dataclass
class SingleFrankaTCPRobotConfig(FrankaRobotConfig):
    """Configuration for :class:`SingleFrankaTCPEnv`."""

    # Also consumed by RealWorldEnv to preserve keyboard-owned episode ends.
    manual_episode_control_only: bool = False
    rotation_repr: str = "rot6d"
    image_width: int = 640
    image_height: int = 480
    camera_fps: int = 30

    # Physical pose ^E T_P of the robot OmniPicker TCP P in Franka endpoint E.
    franka_tcp_to_policy_tool: np.ndarray = field(default_factory=_identity_transform)

    # Physical pose ^C T_U of the acquisition UMI TCP U in the saved Oculus
    # controller/action frame C. Identity preserves compatibility for datasets
    # whose saved pose is already expressed at the physical tool TCP.
    oculus_controller_to_umi_tool: np.ndarray = field(
        default_factory=_identity_transform
    )

    # Per-policy-step safety limits.  The workspace box inherited from
    # FrankaRobotConfig is applied after these local step limits.
    max_translation_per_step_m: float = 0.005
    max_rotation_per_step_rad: float = 0.05
    unsafe_action_behavior: str = "hold"
    open_gripper_on_reset: bool = False
    save_state_action_curve: bool = False
    state_action_curve_dir: str | None = None
    save_pose_trace: bool = False
    pose_trace_dir: str | None = None
    scene_alignment_enabled: bool = False
    scene_alignment_reference_dir: str | None = None
    scene_alignment_reference_set_dir: str | None = None
    scene_alignment_cache_dir: str | None = None
    scene_alignment_epoch_offset: int = 0
    scene_alignment_view_name: str = "wrist_2"
    scene_alignment_video_suffix: str = "wrist_2"
    scene_alignment_host: str = "127.0.0.1"
    scene_alignment_port: int = 8765
    scene_alignment_reference_opacity: float = 0.35

    # Franka reports jaw width in metres.  The UMI state distribution uses
    # approximately 0.2 for closed and 1.0 for open.
    gripper_width_min_m: float = 0.01
    gripper_width_max_m: float = 0.09
    policy_gripper_closed_value: float = 0.2
    policy_gripper_open_value: float = 1.0
    policy_gripper_action_threshold: float = 0.5
    gripper_hysteresis_enabled: bool = False
    policy_gripper_close_threshold: float = 0.4
    policy_gripper_open_threshold: float = 0.6
    gripper_debounce_steps: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rotation_repr != "rot6d":
            raise ValueError("SingleFrankaTCPEnv only supports rotation_repr='rot6d'.")
        self.franka_tcp_to_policy_tool = validate_rigid_transform(
            self.franka_tcp_to_policy_tool,
            "franka_tcp_to_policy_tool",
        )
        self.oculus_controller_to_umi_tool = validate_rigid_transform(
            self.oculus_controller_to_umi_tool,
            "oculus_controller_to_umi_tool",
        )
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive.")
        if self.camera_fps <= 0:
            raise ValueError("camera_fps must be positive.")
        if self.step_frequency <= 0:
            raise ValueError("step_frequency must be positive.")
        if self.max_num_steps <= 0:
            raise ValueError("max_num_steps must be positive.")
        if self.max_translation_per_step_m <= 0:
            raise ValueError("max_translation_per_step_m must be positive.")
        if self.max_rotation_per_step_rad <= 0:
            raise ValueError("max_rotation_per_step_rad must be positive.")
        if self.unsafe_action_behavior not in {"hold", "raise"}:
            raise ValueError("unsafe_action_behavior must be either 'hold' or 'raise'.")
        if self.save_state_action_curve and not (
            isinstance(self.state_action_curve_dir, str)
            and self.state_action_curve_dir.strip()
        ):
            raise ValueError(
                "state_action_curve_dir must be set when "
                "save_state_action_curve is enabled."
            )
        if self.save_pose_trace and not (
            isinstance(self.pose_trace_dir, str) and self.pose_trace_dir.strip()
        ):
            raise ValueError(
                "pose_trace_dir must be set when save_pose_trace is enabled."
            )
        has_reference_video_dir = bool(
            isinstance(self.scene_alignment_reference_dir, str)
            and self.scene_alignment_reference_dir.strip()
        )
        has_reference_set_dir = bool(
            isinstance(self.scene_alignment_reference_set_dir, str)
            and self.scene_alignment_reference_set_dir.strip()
        )
        if self.scene_alignment_enabled and not (
            has_reference_video_dir or has_reference_set_dir
        ):
            raise ValueError(
                "scene_alignment_reference_dir or "
                "scene_alignment_reference_set_dir must be set when scene "
                "alignment is enabled."
            )
        if has_reference_video_dir and has_reference_set_dir:
            raise ValueError(
                "Configure only one of scene_alignment_reference_dir and "
                "scene_alignment_reference_set_dir."
            )
        if self.scene_alignment_cache_dir is not None and not (
            isinstance(self.scene_alignment_cache_dir, str)
            and self.scene_alignment_cache_dir.strip()
        ):
            raise ValueError(
                "scene_alignment_cache_dir must be null or a non-empty path."
            )
        if not self.scene_alignment_view_name.strip():
            raise ValueError("scene_alignment_view_name must not be empty.")
        if self.scene_alignment_epoch_offset < 0:
            raise ValueError("scene_alignment_epoch_offset must be non-negative.")
        if not self.scene_alignment_video_suffix.strip():
            raise ValueError("scene_alignment_video_suffix must not be empty.")
        if not self.scene_alignment_host.strip():
            raise ValueError("scene_alignment_host must not be empty.")
        if not 1 <= self.scene_alignment_port <= 65535:
            raise ValueError("scene_alignment_port must be in [1, 65535].")
        if not 0.0 <= self.scene_alignment_reference_opacity <= 1.0:
            raise ValueError("scene_alignment_reference_opacity must be in [0, 1].")
        if self.gripper_width_max_m <= self.gripper_width_min_m:
            raise ValueError(
                "gripper_width_max_m must be greater than gripper_width_min_m."
            )
        if not (
            0.0
            <= self.policy_gripper_closed_value
            < self.policy_gripper_open_value
            <= 1.0
        ):
            raise ValueError(
                "Policy gripper values must satisfy 0 <= closed < open <= 1."
            )
        if not 0.0 <= self.policy_gripper_action_threshold <= 1.0:
            raise ValueError("policy_gripper_action_threshold must be in [0, 1].")
        if not (
            self.policy_gripper_closed_value
            <= self.policy_gripper_close_threshold
            < self.policy_gripper_open_threshold
            <= self.policy_gripper_open_value
        ):
            raise ValueError(
                "Gripper hysteresis thresholds must satisfy closed_value <= "
                "close_threshold < open_threshold <= open_value."
            )
        if self.gripper_debounce_steps < 1:
            raise ValueError("gripper_debounce_steps must be at least one.")
        for name, value, shape in (
            ("reset_ee_pose", self.reset_ee_pose, (6,)),
            ("target_ee_pose", self.target_ee_pose, (6,)),
            ("ee_pose_limit_min", self.ee_pose_limit_min, (6,)),
            ("ee_pose_limit_max", self.ee_pose_limit_max, (6,)),
            ("action_scale", self.action_scale, (3,)),
        ):
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{name} must be a finite array with shape {shape}, got {value!r}."
                )
        if np.any(self.ee_pose_limit_min > self.ee_pose_limit_max):
            raise ValueError("ee_pose_limit_min must not exceed ee_pose_limit_max.")
        if not np.all(
            (self.reset_ee_pose[:3] >= self.ee_pose_limit_min[:3])
            & (self.reset_ee_pose[:3] <= self.ee_pose_limit_max[:3])
        ):
            raise ValueError(
                "reset_ee_pose xyz must be inside the configured workspace."
            )


class SingleFrankaTCPEnv(FrankaEnv):
    """Single Franka driven by absolute policy-tool TCP rot6d waypoints."""

    CONFIG_CLS: type[SingleFrankaTCPRobotConfig] = SingleFrankaTCPRobotConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._scene_alignment_ui: ReferenceSceneAlignmentWebUI | None = None
        super().__init__(*args, **kwargs)
        self._franka_to_umi_tool = self.config.franka_tcp_to_policy_tool.copy()
        self._controller_to_umi_tool = self.config.oculus_controller_to_umi_tool.copy()
        # The checkpoint predicts body-frame deltas D_C in the saved Oculus
        # controller frame, not in the physical UMI/OmniPicker TCP frame.
        # ^E T_C = ^E T_P @ inverse(^C T_U).
        self._franka_to_policy = self._franka_to_umi_tool @ np.linalg.inv(
            self._controller_to_umi_tool
        )
        self._policy_to_franka = np.linalg.inv(self._franka_to_policy)
        self._last_commanded_quat: np.ndarray | None = None
        self._episode_states: list[np.ndarray] = []
        self._episode_actions: list[np.ndarray] = []
        self._state_action_curve_episode_index = 0
        self._pose_trace_recorder: FrankaPoseTraceRecorder | None = None
        self._pending_async_pose_trace_context: AsyncPoseTraceContext | None = None
        if self.config.save_pose_trace:
            from ..pose_trace import FrankaPoseTraceRecorder

            self._pose_trace_recorder = FrankaPoseTraceRecorder(
                output_dir=self.config.pose_trace_dir,
                capacity=self.config.max_num_steps,
                step_frequency=self.config.step_frequency,
                franka_tcp_to_policy_tool=self._franka_to_umi_tool,
                oculus_controller_to_umi_tool=self._controller_to_umi_tool,
                policy_gripper_closed_value=(self.config.policy_gripper_closed_value),
                policy_gripper_open_value=self.config.policy_gripper_open_value,
                gripper_width_min_m=self.config.gripper_width_min_m,
                gripper_width_max_m=self.config.gripper_width_max_m,
            )

        if self.config.scene_alignment_enabled:
            if self.config.is_dummy:
                raise ValueError(
                    "Scene alignment requires real camera frames and cannot be "
                    "enabled in dummy mode."
                )
            self._scene_alignment_ui = ReferenceSceneAlignmentWebUI(
                self.config.scene_alignment_reference_dir,
                reference_set_dir=self.config.scene_alignment_reference_set_dir,
                video_suffix=self.config.scene_alignment_video_suffix,
                epoch_offset=self.config.scene_alignment_epoch_offset,
                cache_dir=self.config.scene_alignment_cache_dir,
                host=self.config.scene_alignment_host,
                port=self.config.scene_alignment_port,
                default_reference_opacity=(
                    self.config.scene_alignment_reference_opacity
                ),
            )
            reference_set = self._scene_alignment_ui.reference_set
            if (
                reference_set is not None
                and reference_set.view_name != self.config.scene_alignment_view_name
            ):
                raise ValueError(
                    f"Scene-reference preset view {reference_set.view_name!r} does "
                    "not match scene_alignment_view_name "
                    f"{self.config.scene_alignment_view_name!r}."
                )

        if self.config.is_dummy:
            reset_pose = np.concatenate(
                [
                    self.config.reset_ee_pose[:3],
                    R.from_euler("xyz", self.config.reset_ee_pose[3:]).as_quat(),
                ]
            )
            self._franka_state.tcp_pose = reset_pose.astype(np.float64)
            self._franka_state.gripper_position = self.config.gripper_width_max_m
            self._franka_state.gripper_open = True

        self._gripper_command_filter = GripperCommandFilter(
            enabled=self.config.gripper_hysteresis_enabled,
            close_threshold=self.config.policy_gripper_close_threshold,
            open_threshold=self.config.policy_gripper_open_threshold,
            debounce_steps=self.config.gripper_debounce_steps,
            closed_value=self.config.policy_gripper_closed_value,
            open_value=self.config.policy_gripper_open_value,
        )
        self._gripper_command_filter.reset(
            open_state=bool(self._franka_state.gripper_open)
        )

    def _build_camera_infos(self):
        camera_infos = super()._build_camera_infos()
        for camera_info in camera_infos:
            camera_info.resolution = (
                int(self.config.image_width),
                int(self.config.image_height),
            )
            camera_info.fps = int(self.config.camera_fps)
        return camera_infos

    def _crop_frame(
        self,
        frame: np.ndarray,
        reshape_size: tuple[int, int],
        crop_region: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Preserve the full 4:3 UMI training view unless the user explicitly
        # configures a crop for a particular deployment camera.
        if crop_region is not None:
            return super()._crop_frame(frame, reshape_size, crop_region)
        return frame, cv2.resize(frame, reshape_size)

    def _init_action_obs_spaces(self) -> None:
        self._xyz_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[:3],
            high=self.config.ee_pose_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[3:],
            high=self.config.ee_pose_limit_max[3:],
            dtype=np.float64,
        )

        action_low = np.concatenate(
            [
                np.full(3, -np.inf),
                np.full(6, -1.5),
                np.array([0.0]),
            ]
        ).astype(np.float32)
        action_high = np.concatenate(
            [
                np.full(3, np.inf),
                np.full(6, 1.5),
                np.array([1.0]),
            ]
        ).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose_rot6d": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(TCP_ROT6D_DIM,),
                            dtype=np.float32,
                        ),
                        "gripper_position": gym.spaces.Box(
                            0.0, 1.0, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        camera_info.name: gym.spaces.Box(
                            0,
                            255,
                            shape=(
                                self.config.image_height,
                                self.config.image_width,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for camera_info in self._camera_infos
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def reset(self, *, seed=None, options=None):
        self.save_state_action_curve(result="reset")
        self.save_pose_trace(result="reset")
        self._pending_async_pose_trace_context = None
        self._last_commanded_quat = None
        if self.config.is_dummy:
            self._num_steps = 0
            self._success_hold_counter = 0
            self._franka_state.tcp_pose = np.concatenate(
                [
                    self.config.reset_ee_pose[:3],
                    R.from_euler("xyz", self.config.reset_ee_pose[3:]).as_quat(),
                ]
            ).astype(np.float64)
            self._franka_state.gripper_position = self.config.gripper_width_max_m
            self._franka_state.gripper_open = True
        elif self.config.open_gripper_on_reset:
            # Open before returning the arm to reset so an object is released
            # at the episode-end pose rather than carried through the reset.
            self._controller.open_gripper().wait()
            time.sleep(0.6)
        result = super().reset(seed=seed, options=options)
        self._gripper_command_filter.reset(
            open_state=bool(self._franka_state.gripper_open)
        )
        return result

    def _policy_tcp_pose_rot6d(self) -> np.ndarray:
        franka_pose = np.asarray(self._franka_state.tcp_pose, dtype=np.float64)
        base_to_franka = pose_xyzw_to_transform(franka_pose)
        base_to_policy = base_to_franka @ self._franka_to_policy
        xyz = base_to_policy[:3, 3]
        rot6d = matrix_to_rot6d(base_to_policy[:3, :3])
        return np.concatenate([xyz, rot6d]).astype(np.float32)

    def _policy_gripper_state(self) -> float:
        width = float(self._franka_state.gripper_position)
        ratio = (width - self.config.gripper_width_min_m) / (
            self.config.gripper_width_max_m - self.config.gripper_width_min_m
        )
        ratio = float(np.clip(ratio, 0.0, 1.0))
        closed = self.config.policy_gripper_closed_value
        opened = self.config.policy_gripper_open_value
        return float(closed + ratio * (opened - closed))

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            frames = {
                camera_info.name: np.zeros(
                    (
                        self.config.image_height,
                        self.config.image_width,
                        3,
                    ),
                    dtype=np.uint8,
                )
                for camera_info in self._camera_infos
            }
        else:
            frames = self._get_camera_frames()
        state = {
            "tcp_pose_rot6d": self._policy_tcp_pose_rot6d(),
            "gripper_position": np.array(
                [self._policy_gripper_state()], dtype=np.float32
            ),
        }
        return copy.deepcopy({"state": state, "frames": frames})

    def begin_scene_alignment(self, epoch_index: int) -> bool:
        """Prepare the reference first frame for an evaluation epoch."""
        if self._scene_alignment_ui is None:
            return False
        reference_path = self._scene_alignment_ui.start_epoch(epoch_index)
        self._logger.info(
            "Scene alignment ready for epoch %d. Reference=%s UI=%s",
            epoch_index,
            reference_path,
            self._scene_alignment_ui.url,
        )
        return True

    @property
    def scene_alignment_epoch_offset(self) -> int:
        """Return the global epoch number assigned to this reference set."""
        return int(self.config.scene_alignment_epoch_offset)

    def refresh_scene_alignment(self) -> None:
        """Publish the latest configured camera frame to the alignment UI."""
        if self._scene_alignment_ui is None:
            return
        frames = self._get_camera_frames()
        view_name = self.config.scene_alignment_view_name
        if view_name not in frames:
            raise KeyError(
                f"Scene-alignment view {view_name!r} is unavailable; "
                f"configured camera views are {list(frames)}."
            )
        # Franka observations are RGB, while cv2.imencode expects BGR.
        frame_bgr = np.ascontiguousarray(frames[view_name][..., ::-1])
        self._scene_alignment_ui.update_live_frame(frame_bgr)

    def capture_scene_alignment_observation(self) -> dict:
        """Capture a fresh observation after the operator accepts alignment."""
        if self._scene_alignment_ui is None:
            raise RuntimeError("Scene alignment is not enabled.")
        self._franka_state = self._controller.get_state().wait()[0]
        return self._get_observation()

    def finish_scene_alignment(self) -> None:
        """Mark the current alignment as accepted in the browser UI."""
        if self._scene_alignment_ui is not None:
            self._scene_alignment_ui.mark_rollout_started()

    def _policy_action_to_franka_pose(self, action: np.ndarray) -> np.ndarray:
        base_to_policy = np.eye(4, dtype=np.float64)
        base_to_policy[:3, 3] = action[:3]
        base_to_policy[:3, :3] = rot6d_to_matrix(action[3:9])
        base_to_franka = base_to_policy @ self._policy_to_franka
        return transform_to_pose_xyzw(base_to_franka)

    def _policy_action_to_physical_tool_pose(self, action: np.ndarray) -> np.ndarray:
        """Convert an absolute controller-frame action into ``B -> P``."""
        base_to_controller = np.eye(4, dtype=np.float64)
        base_to_controller[:3, 3] = action[:3]
        base_to_controller[:3, :3] = rot6d_to_matrix(action[3:9])
        return transform_to_pose_xyzw(base_to_controller @ self._controller_to_umi_tool)

    def _franka_pose_to_physical_tool_pose(self, pose: np.ndarray) -> np.ndarray:
        """Convert a Franka endpoint pose ``B -> E`` into ``B -> P``."""
        return transform_to_pose_xyzw(
            pose_xyzw_to_transform(pose) @ self._franka_to_umi_tool
        )

    def set_async_pose_trace_context(
        self,
        context: AsyncPoseTraceContext,
    ) -> None:
        """Set metadata consumed by exactly one asynchronous environment step."""
        if self._pose_trace_recorder is None:
            raise RuntimeError(
                "Asynchronous pose tracing was not enabled for this environment."
            )
        if self._pending_async_pose_trace_context is not None:
            raise RuntimeError(
                "The previous asynchronous pose trace context was not consumed."
            )
        self._pending_async_pose_trace_context = context

    def _record_async_pose_trace_sample(
        self,
        *,
        context: AsyncPoseTraceContext | None,
        buffer_action: np.ndarray,
        commanded_franka_pose: np.ndarray,
        unsafe_action: bool,
        translation_clipped: bool,
        rotation_clipped: bool,
        workspace_clipped: bool,
        commanded_gripper_value: float,
        arm_command_dispatched: bool,
        gripper_command_effective: bool,
    ) -> None:
        """Append one post-state-read sample without performing any I/O."""
        if self._pose_trace_recorder is None:
            return
        if context is None:
            self._logger.warning(
                "Skipping pose trace sample because async metadata is missing."
            )
            return
        try:
            model_action = np.asarray(context.model_action, dtype=np.float64).reshape(
                -1
            )
            if model_action.shape != (ACTION_DIM,) or not np.all(
                np.isfinite(model_action)
            ):
                raise ValueError(
                    "Raw model trace action must be a finite 10D vector, got "
                    f"{model_action!r}."
                )
            actual_width_m = float(self._franka_state.gripper_position)
            actual_open_fraction = float(
                np.clip(
                    (actual_width_m - self.config.gripper_width_min_m)
                    / (
                        self.config.gripper_width_max_m
                        - self.config.gripper_width_min_m
                    ),
                    0.0,
                    1.0,
                )
            )
            self._pose_trace_recorder.append(
                control_step=context.control_step,
                episode_step=self._num_steps - 1,
                request_id=context.request_id,
                observation_step=context.observation_step,
                ready_step=context.ready_step,
                chunk_index=context.chunk_index,
                candidate_count=context.candidate_count,
                inference_seconds=context.inference_seconds,
                blended=context.blended,
                exhausted=context.exhausted,
                model_action=model_action.astype(np.float32),
                buffer_action=np.asarray(buffer_action, dtype=np.float32),
                model_tool_pose_xyzw=(
                    self._policy_action_to_physical_tool_pose(model_action)
                ),
                commanded_tool_pose_xyzw=(
                    self._franka_pose_to_physical_tool_pose(commanded_franka_pose)
                ),
                actual_tool_pose_xyzw=(
                    self._franka_pose_to_physical_tool_pose(self._franka_state.tcp_pose)
                ),
                model_gripper_value=float(model_action[9]),
                commanded_gripper_value=float(commanded_gripper_value),
                actual_gripper_width_m=actual_width_m,
                actual_gripper_open_fraction=actual_open_fraction,
                unsafe_action=unsafe_action,
                translation_clipped=translation_clipped,
                rotation_clipped=rotation_clipped,
                workspace_clipped=workspace_clipped,
                arm_command_dispatched=arm_command_dispatched,
                gripper_command_effective=gripper_command_effective,
            )
        except Exception as exc:
            self._logger.warning("Failed to append asynchronous pose trace: %s", exc)

    def save_pose_trace(self, *, result: str = "episode_end") -> str | None:
        """Atomically save and clear the opt-in asynchronous episode trace."""
        recorder = getattr(self, "_pose_trace_recorder", None)
        if recorder is None:
            return None
        try:
            output_path = recorder.save(result=result)
        except Exception as exc:
            self._logger.warning("Failed to save asynchronous pose trace: %s", exc)
            recorder.clear()
            return None
        if output_path is not None:
            self._logger.info("Saved asynchronous pose trace: %s", output_path)
        return output_path

    def _handle_unsafe_action(
        self, action: np.ndarray, reason: str
    ) -> tuple[np.ndarray, bool]:
        if self.config.unsafe_action_behavior == "raise":
            raise ValueError(f"Unsafe SingleFrankaTCPEnv action: {reason}")
        self._logger.warning(
            "Holding Franka TCP because policy action is unsafe: %s; action=%s",
            reason,
            action,
        )
        return np.asarray(self._franka_state.tcp_pose, dtype=np.float64).copy(), True

    def _record_state_action(self, action: np.ndarray) -> None:
        if not self.config.save_state_action_curve:
            return
        state = np.concatenate(
            [
                self._policy_tcp_pose_rot6d(),
                np.array([self._policy_gripper_state()], dtype=np.float32),
            ]
        ).astype(np.float32)
        recorded_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
        flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
        copied_values = min(ACTION_DIM, flat_action.shape[0])
        recorded_action[:copied_values] = flat_action[:copied_values]
        self._episode_states.append(state)
        self._episode_actions.append(recorded_action)

    def save_state_action_curve(self, *, result: str = "episode_end") -> str | None:
        """Save and clear the current episode's aligned state/action trace."""
        if not self.config.save_state_action_curve or not self._episode_actions:
            return None

        states = np.stack(self._episode_states)
        actions = np.stack(self._episode_actions)
        episode_index = self._state_action_curve_episode_index
        safe_result = (
            "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(result)
            ).strip("_")
            or "episode_end"
        )
        output_dir = Path(self.config.state_action_curve_dir).expanduser()
        stem = f"episode_{episode_index:06d}_{safe_result}_state_action"
        image_path = output_dir / f"{stem}.png"
        data_path = output_dir / f"{stem}.npz"
        time_s = np.arange(states.shape[0], dtype=np.float64) / float(
            self.config.step_frequency
        )

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            image = render_state_action_curve(
                states,
                actions,
                step_frequency=self.config.step_frequency,
                title=(
                    f"State-Action Curve | episode={episode_index:06d} | "
                    f"result={safe_result}"
                ),
            )
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"OpenCV failed to write {image_path}.")
            np.savez_compressed(
                data_path,
                states=states,
                actions=actions,
                time_s=time_s,
                channel_names=np.asarray(STATE_ACTION_NAMES),
                result=np.asarray(safe_result),
            )
        except Exception as exc:
            self._logger.warning("Failed to save state-action curve: %s", exc)
            return None
        finally:
            self._episode_states.clear()
            self._episode_actions.clear()

        self._state_action_curve_episode_index += 1
        self._logger.info(
            "Saved episode state-action curve: %s (raw data: %s)",
            image_path,
            data_path,
        )
        return str(image_path)

    def step(self, action: np.ndarray):
        start_time = time.time()
        post_action_observation_wait_s = self._consume_post_action_observation_wait_s()
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        unsafe_action = False
        reason = None

        if action.shape != (ACTION_DIM,):
            reason = f"expected shape ({ACTION_DIM},), got {action.shape}"
        elif not np.all(np.isfinite(action)):
            reason = "action contains NaN or Inf"

        if reason is None:
            try:
                target_pose = self._policy_action_to_franka_pose(action)
            except (ValueError, FloatingPointError) as exc:
                reason = str(exc)

        raw_target_pose = target_pose.copy() if reason is None else None
        if reason is not None:
            target_pose, unsafe_action = self._handle_unsafe_action(action, reason)

        if unsafe_action:
            # A malformed action must never pull an already-stationary robot
            # toward a (possibly misconfigured) workspace boundary.
            translation_clipped = False
            rotation_clipped = False
            workspace_clipped = False
        else:
            target_pose, translation_clipped, rotation_clipped = clamp_tcp_step(
                self._franka_state.tcp_pose,
                target_pose,
                max_translation_m=self.config.max_translation_per_step_m,
                max_rotation_rad=self.config.max_rotation_per_step_rad,
            )
            before_workspace_clip = target_pose.copy()
            target_pose = self._clip_position_to_safety_box(target_pose)
            workspace_clipped = not np.allclose(
                target_pose, before_workspace_clip, atol=1e-7
            )

        if self._last_commanded_quat is not None and (
            np.dot(target_pose[3:], self._last_commanded_quat) < 0.0
        ):
            target_pose[3:] *= -1.0
        self._last_commanded_quat = target_pose[3:].copy()

        gripper_effective = False
        gripper_value = (
            float(np.clip(action[9], 0.0, 1.0))
            if action.shape == (ACTION_DIM,) and np.isfinite(action[9])
            else self._policy_gripper_state()
        )
        gripper_transition = False
        if not unsafe_action:
            gripper_value, gripper_transition = self._gripper_command_filter.apply(
                gripper_value
            )
            if gripper_transition:
                self._logger.info(
                    "Gripper command filter committed %s after %d stable steps.",
                    "open"
                    if gripper_value == self.config.policy_gripper_open_value
                    else "closed",
                    self.config.gripper_debounce_steps,
                )
        recorded_action = action.copy()
        if recorded_action.shape == (ACTION_DIM,):
            recorded_action[9] = gripper_value
        self._record_state_action(recorded_action)
        if not self.config.is_dummy:
            if not unsafe_action:
                if self._ee_type == EndEffectorType.OMNIPICKER_GRIPPER:
                    gripper_command = policy_gripper_to_rlinf_action(
                        gripper_value,
                        closed_value=self.config.policy_gripper_closed_value,
                        open_value=self.config.policy_gripper_open_value,
                    )
                else:
                    gripper_command = (
                        1.0
                        if gripper_value >= self.config.policy_gripper_action_threshold
                        else -1.0
                    )
                gripper_effective = self._end_effector_action(
                    np.array([gripper_command], dtype=np.float64)
                )
            if not unsafe_action and not self.stream_owns_arm:
                self._move_action(target_pose.astype(np.float32))
        else:
            self._franka_state.tcp_pose = target_pose.copy()
            self._franka_state.gripper_open = (
                gripper_value >= self.config.policy_gripper_action_threshold
            )
            self._franka_state.gripper_position = (
                self.config.gripper_width_max_m
                if self._franka_state.gripper_open
                else self.config.gripper_width_min_m
            )

        self._num_steps += 1
        observation_wait_s = post_action_observation_wait_s
        if not self.config.is_dummy:
            step_time = time.time() - start_time
            observation_wait_s += max(
                0.0,
                (1.0 / self.config.step_frequency) - step_time,
            )
        if observation_wait_s > 0.0:
            time.sleep(observation_wait_s)
        if not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]
        position_error_m, rotation_error_rad = tcp_tracking_error(
            self._franka_state.tcp_pose,
            target_pose,
        )
        self._logger.info(
            "Franka controller_error env=%d step=%d "
            "position_error=%.2fmm rotation_error=%.4frad gripper_action=%.4f",
            self.env_idx,
            self._num_steps,
            1000.0 * position_error_m,
            rotation_error_rad,
            (
                float(action[9])
                if action.shape == (ACTION_DIM,) and np.isfinite(action[9])
                else float("nan")
            ),
        )
        if self._pose_trace_recorder is not None:
            pose_trace_context = self._pending_async_pose_trace_context
            self._pending_async_pose_trace_context = None
            self._record_async_pose_trace_sample(
                context=pose_trace_context,
                buffer_action=action,
                commanded_franka_pose=target_pose,
                unsafe_action=unsafe_action,
                translation_clipped=translation_clipped,
                rotation_clipped=rotation_clipped,
                workspace_clipped=workspace_clipped,
                commanded_gripper_value=gripper_value,
                arm_command_dispatched=bool(
                    not unsafe_action
                    and (self.config.is_dummy or not self.stream_owns_arm)
                ),
                gripper_command_effective=gripper_effective,
            )

        observation = self._get_observation()
        reward = self._calc_step_reward(observation, gripper_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "controller_error": {
                "position_error_m": position_error_m,
                "rotation_error_rad": rotation_error_rad,
            },
            "safety": {
                "unsafe_action": unsafe_action,
                "translation_clipped": translation_clipped,
                "rotation_clipped": rotation_clipped,
                "workspace_clipped": workspace_clipped,
                "commanded_tcp_pose": target_pose.astype(np.float32),
                "gripper_raw_value": (
                    float(np.clip(action[9], 0.0, 1.0))
                    if action.shape == (ACTION_DIM,) and np.isfinite(action[9])
                    else None
                ),
                "gripper_filtered_value": gripper_value,
                "gripper_transition": gripper_transition,
            },
        }
        if os.environ.get("TRAINING_DEBUG_TCP_ACTIONS", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            log_every = max(
                1, int(os.environ.get("TRAINING_DEBUG_TCP_ACTIONS_EVERY_N", "1"))
            )
            if self._num_steps % log_every == 0:
                self._logger.info(
                    "[SingleFrankaTCPEnv debug] step=%d "
                    "policy_state_xyz=%s policy_action_xyz=%s "
                    "raw_target_tcp_xyz=%s pre_workspace_tcp_xyz=%s "
                    "commanded_tcp_xyz=%s live_tcp_xyz=%s z_min=%.5f "
                    "workspace_clipped=%s translation_clipped=%s "
                    "rotation_clipped=%s unsafe_action=%s",
                    self._num_steps,
                    np.array2string(
                        observation["state"]["tcp_pose_rot6d"][:3],
                        precision=5,
                        suppress_small=False,
                    ),
                    np.array2string(action[:3], precision=5, suppress_small=False),
                    np.array2string(
                        raw_target_pose[:3]
                        if raw_target_pose is not None
                        else np.full(3, np.nan),
                        precision=5,
                        suppress_small=False,
                    ),
                    np.array2string(
                        before_workspace_clip[:3]
                        if not unsafe_action
                        else np.full(3, np.nan),
                        precision=5,
                        suppress_small=False,
                    ),
                    np.array2string(
                        info["safety"]["commanded_tcp_pose"][:3],
                        precision=5,
                        suppress_small=False,
                    ),
                    np.array2string(
                        np.asarray(self._franka_state.tcp_pose[:3], dtype=np.float64),
                        precision=5,
                        suppress_small=False,
                    ),
                    self.config.ee_pose_limit_min[2],
                    workspace_clipped,
                    translation_clipped,
                    rotation_clipped,
                    unsafe_action,
                )
        return observation, reward, terminated, truncated, info

    def close(self):
        """Flush a partial trace before releasing Franka resources."""
        self.save_state_action_curve(result="shutdown")
        self.save_pose_trace(result="shutdown")
        if self._scene_alignment_ui is not None:
            self._scene_alignment_ui.close()
        return super().close()
