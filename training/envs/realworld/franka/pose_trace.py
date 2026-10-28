# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.
"""Opt-in, numeric-only pose traces for asynchronous Franka evaluation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AsyncPoseTraceContext:
    """Raw model action and timing metadata for one asynchronous control step."""

    model_action: np.ndarray
    control_step: int
    request_id: int
    observation_step: int
    ready_step: int
    chunk_index: int
    candidate_count: int
    inference_seconds: float
    blended: bool
    exhausted: bool


@dataclass(frozen=True)
class _PredictedActionChunk:
    actions: np.ndarray
    request_id: int
    observation_step: int
    ready_step: int
    inference_seconds: float


class AsyncPoseTraceTracker:
    """Track raw chunks only when pose tracing is explicitly enabled.

    This object is deliberately separate from ``TimeAlignedActionBuffer``. The
    normal asynchronous selection implementation therefore remains untouched
    when tracing is disabled.
    """

    def __init__(self, prediction_horizon: int) -> None:
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive.")
        self.prediction_horizon = int(prediction_horizon)
        self.clear()

    def clear(self) -> None:
        """Discard all raw prediction chunks."""
        self._chunks: list[_PredictedActionChunk] = []

    def install(
        self,
        actions: np.ndarray,
        *,
        request_id: int,
        observation_step: int,
        ready_step: int,
        inference_seconds: float,
    ) -> None:
        """Install one accepted action-buffer response without copying it."""
        actions = np.asarray(actions, dtype=np.float32)
        expected_prefix = (1, self.prediction_horizon)
        if actions.ndim != 3 or actions.shape[:2] != expected_prefix:
            raise ValueError(
                "Pose tracing requires one action chunk with shape "
                f"[1, {self.prediction_horizon}, action_dim], got {actions.shape}."
            )
        self._chunks = [
            chunk
            for chunk in self._chunks
            if ready_step - chunk.observation_step < self.prediction_horizon
        ]
        self._chunks.append(
            _PredictedActionChunk(
                actions=actions,
                request_id=int(request_id),
                observation_step=int(observation_step),
                ready_step=int(ready_step),
                inference_seconds=float(inference_seconds),
            )
        )
        self._chunks.sort(key=lambda chunk: (chunk.observation_step, chunk.ready_step))

    def select(
        self,
        control_step: int,
        *,
        blended: bool,
    ) -> AsyncPoseTraceContext:
        """Return the newest unfiltered model target aligned to a control step."""
        if not self._chunks:
            raise RuntimeError("No raw action chunk has been installed.")
        candidates = [
            chunk
            for chunk in self._chunks
            if chunk.ready_step <= control_step
            and 0 <= control_step - chunk.observation_step < self.prediction_horizon
        ]
        exhausted = not candidates
        newest = candidates[-1] if candidates else self._chunks[-1]
        chunk_index = (
            control_step - newest.observation_step
            if candidates
            else self.prediction_horizon - 1
        )
        return AsyncPoseTraceContext(
            model_action=newest.actions[0, chunk_index].copy(),
            control_step=int(control_step),
            request_id=newest.request_id,
            observation_step=newest.observation_step,
            ready_step=newest.ready_step,
            chunk_index=chunk_index,
            candidate_count=len(candidates),
            inference_seconds=newest.inference_seconds,
            blended=bool(blended),
            exhausted=exhausted,
        )


_FIELD_SPECS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "control_step": (np.dtype(np.int64), ()),
    "episode_step": (np.dtype(np.int64), ()),
    "elapsed_time_s": (np.dtype(np.float64), ()),
    "sample_wall_time_s": (np.dtype(np.float64), ()),
    "request_id": (np.dtype(np.int64), ()),
    "observation_step": (np.dtype(np.int64), ()),
    "ready_step": (np.dtype(np.int64), ()),
    "chunk_index": (np.dtype(np.int64), ()),
    "candidate_count": (np.dtype(np.int64), ()),
    "inference_seconds": (np.dtype(np.float64), ()),
    "blended": (np.dtype(np.bool_), ()),
    "exhausted": (np.dtype(np.bool_), ()),
    "model_action": (np.dtype(np.float32), (10,)),
    "buffer_action": (np.dtype(np.float32), (10,)),
    "model_tool_pose_xyzw": (np.dtype(np.float64), (7,)),
    "commanded_tool_pose_xyzw": (np.dtype(np.float64), (7,)),
    "actual_tool_pose_xyzw": (np.dtype(np.float64), (7,)),
    "model_gripper_value": (np.dtype(np.float64), ()),
    "commanded_gripper_value": (np.dtype(np.float64), ()),
    "actual_gripper_width_m": (np.dtype(np.float64), ()),
    "actual_gripper_open_fraction": (np.dtype(np.float64), ()),
    "unsafe_action": (np.dtype(np.bool_), ()),
    "translation_clipped": (np.dtype(np.bool_), ()),
    "rotation_clipped": (np.dtype(np.bool_), ()),
    "workspace_clipped": (np.dtype(np.bool_), ()),
    "arm_command_dispatched": (np.dtype(np.bool_), ()),
    "gripper_command_effective": (np.dtype(np.bool_), ()),
}


class FrankaPoseTraceRecorder:
    """Preallocated in-memory episode trace with atomic end-of-episode saving."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        output_dir: str,
        capacity: int,
        step_frequency: float,
        franka_tcp_to_policy_tool: np.ndarray,
        oculus_controller_to_umi_tool: np.ndarray,
        policy_gripper_closed_value: float,
        policy_gripper_open_value: float,
        gripper_width_min_m: float,
        gripper_width_max_m: float,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Pose trace capacity must be positive.")
        self.output_dir = Path(output_dir).expanduser()
        self.capacity = int(capacity)
        self._arrays = {
            name: np.empty((self.capacity, *shape), dtype=dtype)
            for name, (dtype, shape) in _FIELD_SPECS.items()
        }
        self._metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "coordinate_frame": "Franka base B",
            "tool_frame": "physical OmniPicker TCP P",
            "pose_layout": "[x, y, z, qx, qy, qz, qw]",
            "model_target_definition": (
                "newest time-aligned model target before temporal ensemble, "
                "handover blending, motion-continuity filtering, and safety clamps"
            ),
            "commanded_target_definition": (
                "effective Franka target after action-buffer filtering and "
                "environment safety clamps"
            ),
            "actual_pose_definition": (
                "Franka state read after command dispatch and the control-period wait"
            ),
            "step_frequency_hz": float(step_frequency),
            "franka_tcp_to_policy_tool": np.asarray(
                franka_tcp_to_policy_tool, dtype=np.float64
            ).tolist(),
            "oculus_controller_to_umi_tool": np.asarray(
                oculus_controller_to_umi_tool, dtype=np.float64
            ).tolist(),
            "policy_gripper_closed_value": float(policy_gripper_closed_value),
            "policy_gripper_open_value": float(policy_gripper_open_value),
            "gripper_width_min_m": float(gripper_width_min_m),
            "gripper_width_max_m": float(gripper_width_max_m),
        }
        self._episode_index = 0
        self.clear()

    @property
    def sample_count(self) -> int:
        """Number of samples currently buffered in memory."""
        return self._count

    def clear(self) -> None:
        """Discard the current in-memory episode without touching the filesystem."""
        self._count = 0
        self._episode_start_monotonic_s: float | None = None

    def append(self, **sample: object) -> None:
        """Append one post-dispatch control sample to preallocated arrays."""
        if self._count >= self.capacity:
            raise RuntimeError(
                f"Pose trace exceeded its preallocated capacity of {self.capacity}."
            )
        now_monotonic_s = time.monotonic()
        if self._episode_start_monotonic_s is None:
            self._episode_start_monotonic_s = now_monotonic_s
        sample["elapsed_time_s"] = now_monotonic_s - self._episode_start_monotonic_s
        sample["sample_wall_time_s"] = time.time()
        missing = set(_FIELD_SPECS) - set(sample)
        extra = set(sample) - set(_FIELD_SPECS)
        if missing or extra:
            raise ValueError(
                f"Pose trace sample fields mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}."
            )
        for name, value in sample.items():
            self._arrays[name][self._count] = value
        self._count += 1

    def save(self, *, result: str = "episode_end") -> str | None:
        """Atomically save and clear the current episode as an uncompressed NPZ."""
        if self._count == 0:
            return None
        safe_result = (
            "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(result)
            ).strip("_")
            or "episode_end"
        )
        episode_index = self._episode_index
        output_path = self.output_dir / (
            f"episode_{episode_index:06d}_{safe_result}_pose_trace.npz"
        )
        while output_path.exists():
            episode_index += 1
            output_path = self.output_dir / (
                f"episode_{episode_index:06d}_{safe_result}_pose_trace.npz"
            )
        metadata = dict(self._metadata)
        metadata.update(
            {
                "episode_index": episode_index,
                "result": safe_result,
                "sample_count": self._count,
            }
        )
        payload = {name: values[: self._count] for name, values in self._arrays.items()}
        payload.update(
            {
                "schema_version": np.asarray(self.SCHEMA_VERSION, dtype=np.int64),
                "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                "result": np.asarray(safe_result),
            }
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                prefix=f".{output_path.stem}.",
                dir=self.output_dir,
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                np.savez(temporary_file, **payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, output_path)
        except Exception:
            if temporary_path is not None:
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except OSError:
                    pass
            raise

        self._episode_index = episode_index + 1
        self.clear()
        return str(output_path)
