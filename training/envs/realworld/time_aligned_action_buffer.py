# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0
"""Time-aligned action buffering for real-world VLA evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from training.utils.rot6d import matrix_to_rot6d, rot6d_to_matrix


@dataclass(frozen=True)
class TimeAlignedActionBufferConfig:
    """Configuration for asynchronous real-world action execution."""

    enabled: bool = False
    replan_interval_steps: int = 10
    handover_blend_steps: int = 3
    temporal_ensemble_decay: float = 0.05
    motion_continuity_enabled: bool = True
    motion_tracking_gain: float = 0.08
    max_translation_velocity_m_per_step: float = 0.005
    max_translation_acceleration_m_per_step2: float = 0.001
    max_translation_jerk_m_per_step3: float = 0.00025
    max_rotation_velocity_rad_per_step: float = 0.03
    max_rotation_acceleration_rad_per_step2: float = 0.004
    max_rotation_jerk_rad_per_step3: float = 0.001
    translation_correction_activate_m: float = 0.0015
    translation_correction_release_m: float = 0.00075
    rotation_correction_activate_rad: float = 0.01
    rotation_correction_release_rad: float = 0.005


@dataclass(frozen=True)
class ActionBufferInstallResult:
    """Outcome of installing an asynchronously predicted action chunk."""

    accepted: bool
    aligned_index: int
    response_age_steps: int
    reason: str | None = None


@dataclass(frozen=True)
class ActionBufferSelection:
    """One action selected for a particular control step."""

    action: np.ndarray
    chunk_index: int
    observation_step: int
    blended: bool
    exhausted: bool


@dataclass
class _ActionChunk:
    actions: np.ndarray
    translation_velocities: np.ndarray
    rotation_velocities: np.ndarray
    observation_step: int
    ready_step: int


def resolve_time_aligned_action_buffer_config(
    env_cfg,
    prediction_horizon: int,
) -> TimeAlignedActionBufferConfig:
    """Resolve and validate the asynchronous action-buffer configuration."""
    raw = env_cfg.get("async_action_buffer", {})
    enabled = bool(raw.get("enabled", False))
    replan_interval_steps = int(
        raw.get(
            "replan_interval_steps",
            env_cfg.get("action_execution_horizon", prediction_horizon),
        )
    )
    handover_blend_steps = int(raw.get("handover_blend_steps", 3))
    temporal_ensemble_decay = float(raw.get("temporal_ensemble_decay", 0.05))
    motion_continuity_enabled = bool(raw.get("motion_continuity_enabled", True))
    motion_tracking_gain = float(raw.get("motion_tracking_gain", 0.08))
    max_translation_velocity_m_per_step = float(
        raw.get("max_translation_velocity_m_per_step", 0.005)
    )
    max_translation_acceleration_m_per_step2 = float(
        raw.get("max_translation_acceleration_m_per_step2", 0.001)
    )
    max_translation_jerk_m_per_step3 = float(
        raw.get("max_translation_jerk_m_per_step3", 0.00025)
    )
    max_rotation_velocity_rad_per_step = float(
        raw.get("max_rotation_velocity_rad_per_step", 0.03)
    )
    max_rotation_acceleration_rad_per_step2 = float(
        raw.get("max_rotation_acceleration_rad_per_step2", 0.004)
    )
    max_rotation_jerk_rad_per_step3 = float(
        raw.get("max_rotation_jerk_rad_per_step3", 0.001)
    )
    translation_correction_activate_m = float(
        raw.get("translation_correction_activate_m", 0.0015)
    )
    translation_correction_release_m = float(
        raw.get("translation_correction_release_m", 0.00075)
    )
    rotation_correction_activate_rad = float(
        raw.get("rotation_correction_activate_rad", 0.01)
    )
    rotation_correction_release_rad = float(
        raw.get("rotation_correction_release_rad", 0.005)
    )
    interval_is_valid = 1 <= replan_interval_steps <= prediction_horizon
    if enabled:
        interval_is_valid = 1 <= replan_interval_steps < prediction_horizon
    if not interval_is_valid:
        raise ValueError(
            "async_action_buffer.replan_interval_steps must satisfy "
            f"1 <= interval {'<' if enabled else '<='} prediction_horizon "
            f"({prediction_horizon}), got "
            f"{replan_interval_steps}."
        )
    if handover_blend_steps < 0:
        raise ValueError(
            "async_action_buffer.handover_blend_steps must be non-negative, "
            f"got {handover_blend_steps}."
        )
    if not np.isfinite(temporal_ensemble_decay) or temporal_ensemble_decay < 0.0:
        raise ValueError(
            "async_action_buffer.temporal_ensemble_decay must be a finite "
            f"non-negative number, got {temporal_ensemble_decay}."
        )
    if not np.isfinite(motion_tracking_gain) or not 0.0 <= motion_tracking_gain <= 1.0:
        raise ValueError(
            "async_action_buffer.motion_tracking_gain must be a finite number "
            f"in [0, 1], got {motion_tracking_gain}."
        )
    positive_motion_limits = {
        "max_translation_velocity_m_per_step": max_translation_velocity_m_per_step,
        "max_translation_acceleration_m_per_step2": (
            max_translation_acceleration_m_per_step2
        ),
        "max_translation_jerk_m_per_step3": max_translation_jerk_m_per_step3,
        "max_rotation_velocity_rad_per_step": max_rotation_velocity_rad_per_step,
        "max_rotation_acceleration_rad_per_step2": (
            max_rotation_acceleration_rad_per_step2
        ),
        "max_rotation_jerk_rad_per_step3": max_rotation_jerk_rad_per_step3,
    }
    for name, value in positive_motion_limits.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"async_action_buffer.{name} must be a finite positive number, "
                f"got {value}."
            )
    correction_thresholds = {
        "translation": (
            translation_correction_activate_m,
            translation_correction_release_m,
        ),
        "rotation": (
            rotation_correction_activate_rad,
            rotation_correction_release_rad,
        ),
    }
    for name, (activate, release) in correction_thresholds.items():
        if (
            not np.isfinite(activate)
            or not np.isfinite(release)
            or activate < 0.0
            or release < 0.0
            or release > activate
        ):
            raise ValueError(
                f"async_action_buffer.{name}_correction thresholds must satisfy "
                f"0 <= release <= activate, got release={release}, "
                f"activate={activate}."
            )
    return TimeAlignedActionBufferConfig(
        enabled=enabled,
        replan_interval_steps=replan_interval_steps,
        handover_blend_steps=handover_blend_steps,
        temporal_ensemble_decay=temporal_ensemble_decay,
        motion_continuity_enabled=motion_continuity_enabled,
        motion_tracking_gain=motion_tracking_gain,
        max_translation_velocity_m_per_step=max_translation_velocity_m_per_step,
        max_translation_acceleration_m_per_step2=(
            max_translation_acceleration_m_per_step2
        ),
        max_translation_jerk_m_per_step3=max_translation_jerk_m_per_step3,
        max_rotation_velocity_rad_per_step=max_rotation_velocity_rad_per_step,
        max_rotation_acceleration_rad_per_step2=(
            max_rotation_acceleration_rad_per_step2
        ),
        max_rotation_jerk_rad_per_step3=max_rotation_jerk_rad_per_step3,
        translation_correction_activate_m=translation_correction_activate_m,
        translation_correction_release_m=translation_correction_release_m,
        rotation_correction_activate_rad=rotation_correction_activate_rad,
        rotation_correction_release_rad=rotation_correction_release_rad,
    )


def _slerp_quaternions(
    first_xyzw: np.ndarray,
    second_xyzw: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Spherically interpolate batched xyzw quaternions."""
    first = np.asarray(first_xyzw, dtype=np.float64)
    second = np.asarray(second_xyzw, dtype=np.float64).copy()
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0.0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    linear = dot > 0.9995
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe_denominator = np.where(linear, 1.0, sin_theta)
    first_scale = np.sin((1.0 - weight) * theta) / safe_denominator
    second_scale = np.sin(weight * theta) / safe_denominator
    spherical = first_scale * first + second_scale * second
    interpolated = np.where(
        linear,
        (1.0 - weight) * first + weight * second,
        spherical,
    )
    return interpolated / np.linalg.norm(interpolated, axis=-1, keepdims=True)


def blend_policy_actions(
    previous: np.ndarray,
    current: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend absolute 10D policy actions, with rotation blended on SO(3)."""
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError(
            "Policy actions must have matching [batch, action_dim] shapes, got "
            f"{previous.shape} and {current.shape}."
        )
    if previous.shape[-1] != 10:
        raise ValueError(
            "Time-aligned real-world blending expects 10D policy actions, got "
            f"{previous.shape[-1]}D."
        )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"Blend weight must be in [0, 1], got {weight}.")

    blended = (1.0 - weight) * previous + weight * current
    previous_quat = R.from_matrix(rot6d_to_matrix(previous[:, 3:9])).as_quat()
    current_quat = R.from_matrix(rot6d_to_matrix(current[:, 3:9])).as_quat()
    blended_quat = _slerp_quaternions(previous_quat, current_quat, weight)
    blended[:, 3:9] = matrix_to_rot6d(R.from_quat(blended_quat).as_matrix())
    return blended.astype(np.float32)


def ensemble_policy_actions(
    actions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Fuse aligned absolute 10D actions using a weighted temporal ensemble.

    Translation and gripper values use a weighted arithmetic mean. Rotation is
    averaged as unit quaternions after mapping all quaternions to a common
    hemisphere, which avoids invalid direct averaging of rot6d coordinates.

    Args:
        actions: Candidate actions shaped ``[chunks, batch, 10]``.
        weights: Non-negative weight for each candidate chunk.

    Returns:
        The fused actions shaped ``[batch, 10]``.
    """
    actions = np.asarray(actions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[-1] != 10:
        raise ValueError(
            "Temporal ensemble actions must have shape [chunks, batch, 10], "
            f"got {actions.shape}."
        )
    if weights.shape != (actions.shape[0],):
        raise ValueError(
            "Temporal ensemble weights must have one value per chunk, got "
            f"{weights.shape} for {actions.shape[0]} chunks."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Temporal ensemble weights must be finite and non-negative.")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("Temporal ensemble weights must have a positive sum.")
    normalized_weights = weights / weight_sum

    fused = np.tensordot(normalized_weights, actions, axes=(0, 0))
    rotation_matrices = rot6d_to_matrix(actions[..., 3:9])
    quaternions = (
        R.from_matrix(rotation_matrices.reshape(-1, 3, 3))
        .as_quat()
        .reshape(actions.shape[0], actions.shape[1], 4)
    )
    reference = quaternions[-1]
    signs = np.where(
        np.sum(quaternions * reference[None], axis=-1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    aligned_quaternions = quaternions * signs
    scatter = np.einsum(
        "n,nbi,nbj->bij",
        normalized_weights,
        aligned_quaternions,
        aligned_quaternions,
    )
    _, eigenvectors = np.linalg.eigh(scatter)
    mean_quaternions = eigenvectors[..., -1]
    mean_quaternions = np.where(
        np.sum(mean_quaternions * reference, axis=-1, keepdims=True) < 0.0,
        -mean_quaternions,
        mean_quaternions,
    )
    fused[..., 3:9] = matrix_to_rot6d(R.from_quat(mean_quaternions).as_matrix())
    return fused.astype(np.float32)


def _compute_chunk_motion(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-step spatial motion from one absolute action chunk.

    Motion at index ``i`` advances the plan from ``action[i - 1]`` to
    ``action[i]``. Index zero reuses the first forward difference so a chunk
    that is returned without inference latency still has a meaningful motion
    reference. The initial buffer selection is held explicitly, so this
    forward value is not used to jump the robot at episode startup.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[-1] != 10:
        raise ValueError(
            "Chunk motion expects actions shaped [batch, horizon, 10], "
            f"got {actions.shape}."
        )

    translation_velocities = np.zeros(
        (*actions.shape[:2], 3),
        dtype=np.float64,
    )
    rotation_velocities = np.zeros_like(translation_velocities)
    if actions.shape[1] <= 1:
        return (
            translation_velocities.astype(np.float32),
            rotation_velocities.astype(np.float32),
        )

    translation_velocities[:, 1:] = np.diff(actions[..., :3], axis=1)
    translation_velocities[:, 0] = translation_velocities[:, 1]

    rotation_matrices = rot6d_to_matrix(actions[..., 3:9])
    rotation_deltas = np.einsum(
        "bhij,bhkj->bhik",
        rotation_matrices[:, 1:],
        rotation_matrices[:, :-1],
    )
    rotation_velocities[:, 1:] = (
        R.from_matrix(rotation_deltas.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(actions.shape[0], actions.shape[1] - 1, 3)
    )
    rotation_velocities[:, 0] = rotation_velocities[:, 1]
    return (
        translation_velocities.astype(np.float32),
        rotation_velocities.astype(np.float32),
    )


def _ensemble_motion_vectors(
    vectors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Fuse aligned Cartesian or spatial-angular velocity references."""
    vectors = np.asarray(vectors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[-1] != 3:
        raise ValueError(
            f"Motion vectors must have shape [chunks, batch, 3], got {vectors.shape}."
        )
    if weights.shape != (vectors.shape[0],):
        raise ValueError(
            "Motion weights must have one value per chunk, got "
            f"{weights.shape} for {vectors.shape[0]} chunks."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Motion weights must be finite and non-negative.")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("Motion weights must have a positive sum.")
    return np.tensordot(weights / weight_sum, vectors, axes=(0, 0)).astype(np.float32)


def _smoothstep(value: float) -> float:
    """Return a cubic zero-slope ramp over the closed unit interval."""
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _clip_vector_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip each row of a vector batch without changing its direction."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scales = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return vectors * scales


class _SE3MotionContinuityFilter:
    """Track absolute targets with bounded velocity, acceleration, and jerk.

    The filter follows the velocity of the fused target trajectory instead of
    repeatedly stepping directly to each absolute target. A proportional
    correction prevents long-term drift. Translation is processed in Cartesian
    coordinates; orientation velocity and correction are processed as spatial
    rotation vectors on SO(3). The gripper command bypasses this filter.
    """

    def __init__(
        self,
        *,
        tracking_gain: float,
        max_translation_velocity_m_per_step: float,
        max_translation_acceleration_m_per_step2: float,
        max_translation_jerk_m_per_step3: float,
        max_rotation_velocity_rad_per_step: float,
        max_rotation_acceleration_rad_per_step2: float,
        max_rotation_jerk_rad_per_step3: float,
        translation_correction_activate_m: float,
        translation_correction_release_m: float,
        rotation_correction_activate_rad: float,
        rotation_correction_release_rad: float,
    ):
        self.tracking_gain = float(tracking_gain)
        self.max_translation_velocity_m_per_step = float(
            max_translation_velocity_m_per_step
        )
        self.max_translation_acceleration_m_per_step2 = float(
            max_translation_acceleration_m_per_step2
        )
        self.max_translation_jerk_m_per_step3 = float(max_translation_jerk_m_per_step3)
        self.max_rotation_velocity_rad_per_step = float(
            max_rotation_velocity_rad_per_step
        )
        self.max_rotation_acceleration_rad_per_step2 = float(
            max_rotation_acceleration_rad_per_step2
        )
        self.max_rotation_jerk_rad_per_step3 = float(max_rotation_jerk_rad_per_step3)
        self.translation_correction_activate_m = float(
            translation_correction_activate_m
        )
        self.translation_correction_release_m = float(translation_correction_release_m)
        self.rotation_correction_activate_rad = float(rotation_correction_activate_rad)
        self.rotation_correction_release_rad = float(rotation_correction_release_rad)
        self.reset()

    def reset(self) -> None:
        """Forget the previous target, output, and velocities."""
        self._last_target: np.ndarray | None = None
        self._last_output: np.ndarray | None = None
        self._translation_velocity: np.ndarray | None = None
        self._translation_acceleration: np.ndarray | None = None
        self._rotation_velocity: np.ndarray | None = None
        self._rotation_acceleration: np.ndarray | None = None
        self._translation_correction_active: np.ndarray | None = None
        self._rotation_correction_active: np.ndarray | None = None

    @staticmethod
    def _validate_action(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 2 or action.shape[-1] != 10:
            raise ValueError(
                "SE(3) motion continuity expects actions shaped [batch, 10], "
                f"got {action.shape}."
            )
        return action

    def hold(self, action: np.ndarray) -> np.ndarray:
        """Hold an action and reset velocity before execution resumes."""
        action = self._validate_action(action).copy()
        batch_size = action.shape[0]
        self._last_target = action.copy()
        self._last_output = action.copy()
        self._translation_velocity = np.zeros((batch_size, 3), dtype=np.float64)
        self._translation_acceleration = np.zeros((batch_size, 3), dtype=np.float64)
        self._rotation_velocity = np.zeros((batch_size, 3), dtype=np.float64)
        self._rotation_acceleration = np.zeros((batch_size, 3), dtype=np.float64)
        self._translation_correction_active = np.zeros(batch_size, dtype=bool)
        self._rotation_correction_active = np.zeros(batch_size, dtype=bool)
        return action.astype(np.float32)

    @staticmethod
    def _apply_correction_hysteresis(
        correction: np.ndarray,
        active: np.ndarray,
        *,
        activate_threshold: float,
        release_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a radial deadband with stateful activation hysteresis."""
        norms = np.linalg.norm(correction, axis=-1)
        next_active = np.where(
            active,
            norms > release_threshold,
            norms > activate_threshold,
        )
        scale = np.maximum(norms - release_threshold, 0.0) / np.maximum(norms, 1e-12)
        filtered = correction * scale[:, None]
        filtered[~next_active] = 0.0
        return filtered, next_active

    @staticmethod
    def _jerk_limited_acceleration(
        requested_velocity: np.ndarray,
        previous_velocity: np.ndarray,
        previous_acceleration: np.ndarray,
        *,
        max_acceleration: float,
        max_jerk: float,
    ) -> np.ndarray:
        """Approach requested velocity without discontinuous acceleration."""
        requested_acceleration = _clip_vector_norm(
            requested_velocity - previous_velocity,
            max_acceleration,
        )
        jerk = _clip_vector_norm(
            requested_acceleration - previous_acceleration,
            max_jerk,
        )
        return _clip_vector_norm(
            previous_acceleration + jerk,
            max_acceleration,
        )

    def apply(
        self,
        target: np.ndarray,
        *,
        translation_reference_velocity: np.ndarray,
        rotation_reference_velocity: np.ndarray,
    ) -> np.ndarray:
        """Follow a fused pose using motion derived inside model chunks.

        The feed-forward velocities are computed before temporal-ensemble
        weights change. This deliberately avoids differentiating the fused
        absolute pose, which otherwise injects a periodic velocity term at the
        action-buffer replanning frequency.
        """
        target = self._validate_action(target)
        if self._last_target is None:
            return self.hold(target)
        if target.shape != self._last_target.shape:
            raise ValueError(
                "SE(3) motion-continuity batch shape changed from "
                f"{self._last_target.shape} to {target.shape}."
            )
        assert self._last_output is not None
        assert self._translation_velocity is not None
        assert self._translation_acceleration is not None
        assert self._rotation_velocity is not None
        assert self._rotation_acceleration is not None
        assert self._translation_correction_active is not None
        assert self._rotation_correction_active is not None

        translation_reference_velocity = np.asarray(
            translation_reference_velocity,
            dtype=np.float64,
        )
        rotation_reference_velocity = np.asarray(
            rotation_reference_velocity,
            dtype=np.float64,
        )
        expected_motion_shape = (target.shape[0], 3)
        if translation_reference_velocity.shape != expected_motion_shape:
            raise ValueError(
                "Translation reference velocity must have shape "
                f"{expected_motion_shape}, got "
                f"{translation_reference_velocity.shape}."
            )
        if rotation_reference_velocity.shape != expected_motion_shape:
            raise ValueError(
                "Rotation reference velocity must have shape "
                f"{expected_motion_shape}, got {rotation_reference_velocity.shape}."
            )

        output = target.copy()

        translation_reference_velocity = _clip_vector_norm(
            translation_reference_velocity,
            self.max_translation_velocity_m_per_step,
        )
        predicted_translation = (
            self._last_output[:, :3] + translation_reference_velocity
        )
        translation_correction = target[:, :3] - predicted_translation
        translation_correction, self._translation_correction_active = (
            self._apply_correction_hysteresis(
                translation_correction,
                self._translation_correction_active,
                activate_threshold=self.translation_correction_activate_m,
                release_threshold=self.translation_correction_release_m,
            )
        )
        requested_translation_velocity = (
            translation_reference_velocity + self.tracking_gain * translation_correction
        )
        requested_translation_velocity = _clip_vector_norm(
            requested_translation_velocity,
            self.max_translation_velocity_m_per_step,
        )
        translation_acceleration = self._jerk_limited_acceleration(
            requested_translation_velocity,
            self._translation_velocity,
            self._translation_acceleration,
            max_acceleration=self.max_translation_acceleration_m_per_step2,
            max_jerk=self.max_translation_jerk_m_per_step3,
        )
        translation_velocity = _clip_vector_norm(
            self._translation_velocity + translation_acceleration,
            self.max_translation_velocity_m_per_step,
        )
        output[:, :3] = self._last_output[:, :3] + translation_velocity

        target_rotation = rot6d_to_matrix(target[:, 3:9])
        previous_output_rotation = rot6d_to_matrix(self._last_output[:, 3:9])
        rotation_reference_velocity = _clip_vector_norm(
            rotation_reference_velocity,
            self.max_rotation_velocity_rad_per_step,
        )
        predicted_rotation = np.einsum(
            "bij,bjk->bik",
            R.from_rotvec(rotation_reference_velocity).as_matrix(),
            previous_output_rotation,
        )
        rotation_correction_delta = np.einsum(
            "bij,bkj->bik",
            target_rotation,
            predicted_rotation,
        )
        rotation_correction = R.from_matrix(rotation_correction_delta).as_rotvec()
        rotation_correction, self._rotation_correction_active = (
            self._apply_correction_hysteresis(
                rotation_correction,
                self._rotation_correction_active,
                activate_threshold=self.rotation_correction_activate_rad,
                release_threshold=self.rotation_correction_release_rad,
            )
        )
        requested_rotation_velocity = (
            rotation_reference_velocity + self.tracking_gain * rotation_correction
        )
        requested_rotation_velocity = _clip_vector_norm(
            requested_rotation_velocity,
            self.max_rotation_velocity_rad_per_step,
        )
        rotation_acceleration = self._jerk_limited_acceleration(
            requested_rotation_velocity,
            self._rotation_velocity,
            self._rotation_acceleration,
            max_acceleration=self.max_rotation_acceleration_rad_per_step2,
            max_jerk=self.max_rotation_jerk_rad_per_step3,
        )
        rotation_velocity = _clip_vector_norm(
            self._rotation_velocity + rotation_acceleration,
            self.max_rotation_velocity_rad_per_step,
        )
        output_rotation = np.einsum(
            "bij,bjk->bik",
            R.from_rotvec(rotation_velocity).as_matrix(),
            previous_output_rotation,
        )
        output[:, 3:9] = matrix_to_rot6d(output_rotation)

        self._last_target = target.copy()
        self._last_output = output.copy()
        self._translation_acceleration = (
            translation_velocity - self._translation_velocity
        )
        self._rotation_acceleration = rotation_velocity - self._rotation_velocity
        self._translation_velocity = translation_velocity
        self._rotation_velocity = rotation_velocity
        return output.astype(np.float32)


class TimeAlignedActionBuffer:
    """Select actions by control-step timestamp while inference runs in parallel.

    Each predicted chunk is stamped with the control step at which its observation
    was captured. If inference takes ``L`` control steps, the first action used from
    that chunk is index ``L`` rather than index zero. All predictions that still
    cover the same target control step are fused as a temporal ensemble. New chunks
    ramp in and expiring chunks ramp out. Absolute poses and within-chunk spatial
    velocities are fused with the same weights, so changing ensemble weights
    cannot create the artificial motion produced by differentiating blended poses.
    """

    def __init__(
        self,
        prediction_horizon: int,
        handover_blend_steps: int = 3,
        temporal_ensemble_decay: float = 0.05,
        motion_continuity_enabled: bool = False,
        motion_tracking_gain: float = 0.08,
        max_translation_velocity_m_per_step: float = 0.005,
        max_translation_acceleration_m_per_step2: float = 0.001,
        max_translation_jerk_m_per_step3: float = 0.00025,
        max_rotation_velocity_rad_per_step: float = 0.03,
        max_rotation_acceleration_rad_per_step2: float = 0.004,
        max_rotation_jerk_rad_per_step3: float = 0.001,
        translation_correction_activate_m: float = 0.0015,
        translation_correction_release_m: float = 0.00075,
        rotation_correction_activate_rad: float = 0.01,
        rotation_correction_release_rad: float = 0.005,
    ):
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive.")
        if handover_blend_steps < 0:
            raise ValueError("handover_blend_steps must be non-negative.")
        if not np.isfinite(temporal_ensemble_decay) or temporal_ensemble_decay < 0:
            raise ValueError("temporal_ensemble_decay must be finite and non-negative.")
        if (
            not np.isfinite(motion_tracking_gain)
            or not 0.0 <= motion_tracking_gain <= 1.0
        ):
            raise ValueError("motion_tracking_gain must be finite and in [0, 1].")
        motion_limits = (
            max_translation_velocity_m_per_step,
            max_translation_acceleration_m_per_step2,
            max_translation_jerk_m_per_step3,
            max_rotation_velocity_rad_per_step,
            max_rotation_acceleration_rad_per_step2,
            max_rotation_jerk_rad_per_step3,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in motion_limits):
            raise ValueError(
                "Motion velocity, acceleration, and jerk limits must be positive."
            )
        correction_thresholds = (
            (
                "translation",
                translation_correction_activate_m,
                translation_correction_release_m,
            ),
            (
                "rotation",
                rotation_correction_activate_rad,
                rotation_correction_release_rad,
            ),
        )
        for name, activate, release in correction_thresholds:
            if (
                not np.isfinite(activate)
                or not np.isfinite(release)
                or activate < 0.0
                or release < 0.0
                or release > activate
            ):
                raise ValueError(
                    f"{name} correction thresholds must satisfy "
                    f"0 <= release <= activate, got release={release}, "
                    f"activate={activate}."
                )
        self.prediction_horizon = int(prediction_horizon)
        self.handover_blend_steps = int(handover_blend_steps)
        self.temporal_ensemble_decay = float(temporal_ensemble_decay)
        self.motion_continuity_enabled = bool(motion_continuity_enabled)
        self._motion_continuity_filter = (
            _SE3MotionContinuityFilter(
                tracking_gain=motion_tracking_gain,
                max_translation_velocity_m_per_step=(
                    max_translation_velocity_m_per_step
                ),
                max_translation_acceleration_m_per_step2=(
                    max_translation_acceleration_m_per_step2
                ),
                max_translation_jerk_m_per_step3=max_translation_jerk_m_per_step3,
                max_rotation_velocity_rad_per_step=max_rotation_velocity_rad_per_step,
                max_rotation_acceleration_rad_per_step2=(
                    max_rotation_acceleration_rad_per_step2
                ),
                max_rotation_jerk_rad_per_step3=max_rotation_jerk_rad_per_step3,
                translation_correction_activate_m=(translation_correction_activate_m),
                translation_correction_release_m=(translation_correction_release_m),
                rotation_correction_activate_rad=rotation_correction_activate_rad,
                rotation_correction_release_rad=rotation_correction_release_rad,
            )
            if self.motion_continuity_enabled
            else None
        )
        self.clear()

    def clear(self) -> None:
        """Discard all predictions and the last held action."""
        self._chunks: list[_ActionChunk] = []
        self._last_action: np.ndarray | None = None
        if self._motion_continuity_filter is not None:
            self._motion_continuity_filter.reset()

    def install(
        self,
        actions: np.ndarray,
        *,
        observation_step: int,
        ready_step: int,
    ) -> ActionBufferInstallResult:
        """Install a prediction and align it to the current control step."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError(
                "Predicted actions must have shape [batch, horizon, action_dim], "
                f"got {actions.shape}."
            )
        if actions.shape[1] != self.prediction_horizon:
            raise ValueError(
                f"Expected prediction horizon {self.prediction_horizon}, got "
                f"{actions.shape[1]}."
            )
        if observation_step < 0 or ready_step < observation_step:
            raise ValueError(
                "Control-step timestamps must satisfy "
                f"0 <= observation_step <= ready_step, got {observation_step} and "
                f"{ready_step}."
            )

        response_age_steps = ready_step - observation_step
        if response_age_steps >= self.prediction_horizon:
            return ActionBufferInstallResult(
                accepted=False,
                aligned_index=response_age_steps,
                response_age_steps=response_age_steps,
                reason="prediction arrived after its action horizon expired",
            )

        self._chunks = [
            chunk
            for chunk in self._chunks
            if ready_step - chunk.observation_step < self.prediction_horizon
        ]
        translation_velocities, rotation_velocities = _compute_chunk_motion(actions)
        self._chunks.append(
            _ActionChunk(
                actions=np.ascontiguousarray(actions),
                translation_velocities=np.ascontiguousarray(translation_velocities),
                rotation_velocities=np.ascontiguousarray(rotation_velocities),
                observation_step=int(observation_step),
                ready_step=int(ready_step),
            )
        )
        self._chunks.sort(key=lambda chunk: (chunk.observation_step, chunk.ready_step))
        return ActionBufferInstallResult(
            accepted=True,
            aligned_index=response_age_steps,
            response_age_steps=response_age_steps,
        )

    def select(self, control_step: int) -> ActionBufferSelection:
        """Return the time-aligned action for ``control_step``."""
        if not self._chunks:
            raise RuntimeError("No action chunk has been installed.")

        candidates = [
            chunk
            for chunk in self._chunks
            if chunk.ready_step <= control_step
            and 0 <= control_step - chunk.observation_step < self.prediction_horizon
        ]
        if not candidates:
            newest = self._chunks[-1]
            if self._last_action is None:
                selected = newest.actions[:, -1]
            else:
                selected = self._last_action.copy()
            if self._motion_continuity_filter is not None:
                selected = self._motion_continuity_filter.hold(selected)
            self._last_action = np.ascontiguousarray(selected)
            return ActionBufferSelection(
                action=self._last_action.copy(),
                chunk_index=self.prediction_horizon - 1,
                observation_step=newest.observation_step,
                blended=False,
                exhausted=True,
            )

        newest = candidates[-1]
        current_index = control_step - newest.observation_step
        if self.handover_blend_steps == 0 or len(candidates) == 1:
            selected = newest.actions[:, current_index]
            translation_reference_velocity = newest.translation_velocities[
                :, current_index
            ]
            rotation_reference_velocity = newest.rotation_velocities[:, current_index]
            blended = False
        else:
            ramp_span = self.handover_blend_steps + 1
            newest_observation_step = newest.observation_step
            candidate_actions = []
            candidate_translation_velocities = []
            candidate_rotation_velocities = []
            candidate_weights = []
            for chunk in candidates:
                chunk_index = control_step - chunk.observation_step
                steps_since_ready = control_step - chunk.ready_step
                remaining_steps = self.prediction_horizon - chunk_index
                ramp_in = _smoothstep((steps_since_ready + 1) / ramp_span)
                ramp_out = _smoothstep(remaining_steps / ramp_span)
                recency = np.exp(
                    -self.temporal_ensemble_decay
                    * (newest_observation_step - chunk.observation_step)
                )
                candidate_actions.append(chunk.actions[:, chunk_index])
                candidate_translation_velocities.append(
                    chunk.translation_velocities[:, chunk_index]
                )
                candidate_rotation_velocities.append(
                    chunk.rotation_velocities[:, chunk_index]
                )
                candidate_weights.append(ramp_in * ramp_out * recency)
            candidate_weights = np.asarray(candidate_weights)
            selected = ensemble_policy_actions(
                np.stack(candidate_actions),
                candidate_weights,
            )
            translation_reference_velocity = _ensemble_motion_vectors(
                np.stack(candidate_translation_velocities),
                candidate_weights,
            )
            rotation_reference_velocity = _ensemble_motion_vectors(
                np.stack(candidate_rotation_velocities),
                candidate_weights,
            )
            blended = True

        if self._motion_continuity_filter is not None:
            selected = self._motion_continuity_filter.apply(
                selected,
                translation_reference_velocity=translation_reference_velocity,
                rotation_reference_velocity=rotation_reference_velocity,
            )
        self._last_action = np.ascontiguousarray(selected)
        return ActionBufferSelection(
            action=self._last_action.copy(),
            chunk_index=current_index,
            observation_step=newest.observation_step,
            blended=blended,
            exhausted=False,
        )
