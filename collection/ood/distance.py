# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Geometry-aware distances for 10-D single-Franka action chunks.

Actions use the layout ``xyz(3) + rotation6d(6) + gripper(1)``.  Model
samples have shape ``[K, T, 10]`` and the aligned human chunk has shape
``[T, 10]``.  The final OOD score is the empirical Energy Distance described
in the retained implementation; it does not fit a
Gaussian or collapse multi-modal model samples to their mean.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometricDistanceConfig:
    """Physical/statistical scales and semantic weights for action distance."""

    position_scales: np.ndarray
    rotation_scale_rad: float
    gripper_scale: float
    position_weight: float = 1.0 / 3.0
    rotation_weight: float = 1.0 / 3.0
    gripper_weight: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        position_scales = np.asarray(self.position_scales, dtype=np.float64)
        if position_scales.shape != (3,):
            raise ValueError(
                "position_scales must contain exactly xyz scales, "
                f"got shape {position_scales.shape}"
            )
        if not np.all(np.isfinite(position_scales)) or np.any(position_scales <= 0):
            raise ValueError("position_scales must be finite and strictly positive")
        object.__setattr__(self, "position_scales", position_scales)

        rotation_scale_rad = float(self.rotation_scale_rad)
        if not math.isfinite(rotation_scale_rad) or not (0.0 < rotation_scale_rad <= math.pi):
            raise ValueError("rotation_scale_rad must be finite and in (0, pi]")
        object.__setattr__(self, "rotation_scale_rad", rotation_scale_rad)

        gripper_scale = float(self.gripper_scale)
        if not math.isfinite(gripper_scale) or gripper_scale <= 0:
            raise ValueError("gripper_scale must be finite and strictly positive")
        object.__setattr__(self, "gripper_scale", gripper_scale)

        weights = np.asarray(
            [self.position_weight, self.rotation_weight, self.gripper_weight],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("component weights must be finite and non-negative")
        if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                "position, rotation, and gripper weights must sum to 1, "
                f"got {weights.sum():.8f}"
            )


def _load_action_std(norm_stats_path: str | Path) -> np.ndarray:
    """Read checkpoint ``actions.std`` for statistics-backed components."""
    path = Path(norm_stats_path).expanduser()
    if path.is_dir():
        path = path / "norm_stats.json"
    if not path.is_file():
        raise ValueError(f"norm stats file does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read norm stats from {path}: {exc}") from exc

    try:
        std = np.asarray(payload["norm_stats"]["actions"]["std"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"norm stats {path} must contain norm_stats.actions.std"
        ) from exc
    if std.ndim != 1 or std.size < 10:
        raise ValueError(
            f"norm_stats.actions.std must contain at least 10 values, got {std.shape}"
        )

    return std


def load_gripper_scale(norm_stats_path: str | Path) -> float:
    """Load only the gripper denominator from checkpoint ``actions.std[9]``."""
    gripper_scale = float(_load_action_std(norm_stats_path)[9])
    if not math.isfinite(gripper_scale) or gripper_scale <= 0:
        raise ValueError(
            "checkpoint action std for gripper must be finite and positive, "
            f"got {gripper_scale}"
        )
    return gripper_scale


def load_action_scales(norm_stats_path: str | Path) -> tuple[np.ndarray, float]:
    """Backward-compatible loader for checkpoint xyz and gripper statistics.

    New OOD monitor configurations intentionally do not use the returned xyz
    values; their ``s_x/s_y/s_z`` denominators come from ``ood.yaml``.
    """
    std = _load_action_std(norm_stats_path)
    position_scales = std[:3].copy()
    gripper_scale = float(std[9])
    selected = np.concatenate([position_scales, [gripper_scale]])
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0):
        raise ValueError(
            "checkpoint action std for xyz and gripper must be finite and positive, "
            f"got {selected.tolist()}"
        )
    return position_scales, gripper_scale


def rotation6d_to_matrix(rotation6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert Zhou-style rotation-6D columns to an SO(3) matrix."""
    rotation6d = np.asarray(rotation6d, dtype=np.float64)
    if rotation6d.shape[-1:] != (6,):
        raise ValueError(
            f"rotation6d must have trailing dimension 6, got {rotation6d.shape}"
        )

    first = rotation6d[..., :3]
    second = rotation6d[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < eps):
        raise ValueError("degenerate rotation6d first column")
    basis1 = first / first_norm

    second_orthogonal = second - np.sum(basis1 * second, axis=-1, keepdims=True) * basis1
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm < eps):
        raise ValueError("degenerate rotation6d second column")
    basis2 = second_orthogonal / second_norm
    basis3 = np.cross(basis1, basis2, axis=-1)
    return np.stack([basis1, basis2, basis3], axis=-1)


def _validate_action_array(actions: np.ndarray, name: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim < 2 or actions.shape[-1] != 10:
        raise ValueError(
            f"{name} must have shape [..., T, 10], got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"{name} contains non-finite values")
    return actions


def geometric_step_squared_distance(
    action_a: np.ndarray,
    action_b: np.ndarray,
    config: GeometricDistanceConfig,
) -> np.ndarray:
    """Return broadcast step-distance squared for 10-D actions."""
    action_a = np.asarray(action_a, dtype=np.float64)
    action_b = np.asarray(action_b, dtype=np.float64)
    if action_a.shape[-1:] != (10,) or action_b.shape[-1:] != (10,):
        raise ValueError(
            "action inputs must have trailing dimension 10, "
            f"got {action_a.shape} and {action_b.shape}"
        )
    if not np.all(np.isfinite(action_a)) or not np.all(np.isfinite(action_b)):
        raise ValueError("action inputs contain non-finite values")

    position_delta = (action_a[..., :3] - action_b[..., :3]) / config.position_scales
    position_squared = np.mean(np.square(position_delta), axis=-1)

    rotation_a = rotation6d_to_matrix(action_a[..., 3:9])
    rotation_b = rotation6d_to_matrix(action_b[..., 3:9])
    chordal = np.linalg.norm(rotation_a - rotation_b, axis=(-2, -1))
    rotation_denominator = 2.0 * math.sqrt(2.0) * math.sin(
        config.rotation_scale_rad / 2.0
    )
    rotation_squared = np.square(chordal / rotation_denominator)

    gripper_squared = np.square(
        (action_a[..., 9] - action_b[..., 9]) / config.gripper_scale
    )
    return (
        config.position_weight * position_squared
        + config.rotation_weight * rotation_squared
        + config.gripper_weight * gripper_squared
    )


def geometric_chunk_distance(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    config: GeometricDistanceConfig,
    time_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the geometry-aware distance ``rho`` between action chunks.

    Leading dimensions are broadcast, allowing both human-to-sample and
    sample-to-sample distance matrices to be calculated without Python loops.
    """
    chunk_a = _validate_action_array(chunk_a, "chunk_a")
    chunk_b = _validate_action_array(chunk_b, "chunk_b")
    if chunk_a.shape[-2] != chunk_b.shape[-2]:
        raise ValueError(
            "action chunks must have the same number of timesteps, "
            f"got {chunk_a.shape[-2]} and {chunk_b.shape[-2]}"
        )
    timesteps = chunk_a.shape[-2]
    if time_weights is None:
        weights = np.full(timesteps, 1.0 / timesteps, dtype=np.float64)
    else:
        weights = np.asarray(time_weights, dtype=np.float64)
        if weights.shape != (timesteps,):
            raise ValueError(
                f"time_weights must have shape ({timesteps},), got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("time_weights must be finite and non-negative")
        if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(f"time_weights must sum to 1, got {weights.sum():.8f}")

    per_step_squared = geometric_step_squared_distance(chunk_a, chunk_b, config)
    return np.sqrt(np.sum(per_step_squared * weights, axis=-1))


def geometric_energy_distance(
    model_samples: np.ndarray,
    human_chunk: np.ndarray,
    config: GeometricDistanceConfig,
) -> dict[str, float | np.ndarray]:
    """Compute empirical Energy Distance using geometry-aware chunk distance.

    ``D² = 2 E[rho(H, A)] - E[rho(A, A')]`` and the returned score is
    ``sqrt(max(D², 0))`` without changing the retained metric definition.
    """
    model_samples = _validate_action_array(model_samples, "model_samples")
    human_chunk = _validate_action_array(human_chunk, "human_chunk")
    if model_samples.ndim != 3:
        raise ValueError(
            f"model_samples must be 3-D [K,T,10], got {model_samples.shape}"
        )
    if human_chunk.ndim == 3:
        if human_chunk.shape[0] != 1:
            raise ValueError(
                "human_chunk batch dimension must be 1 when present, "
                f"got {human_chunk.shape}"
            )
        human_chunk = human_chunk[0]
    if human_chunk.ndim != 2:
        raise ValueError(f"human_chunk must be [T,10], got {human_chunk.shape}")
    if human_chunk.shape != model_samples.shape[1:]:
        raise ValueError(
            "human_chunk shape must match model sample [T,10], "
            f"got {human_chunk.shape} vs {model_samples.shape[1:]}"
        )

    human_to_samples = geometric_chunk_distance(
        model_samples,
        human_chunk[None, ...],
        config,
    )
    sample_to_sample = geometric_chunk_distance(
        model_samples[:, None, :, :],
        model_samples[None, :, :, :],
        config,
    )
    human_term = float(human_to_samples.mean())
    model_spread = float(sample_to_sample.mean())
    squared_score = 2.0 * human_term - model_spread
    score = math.sqrt(max(squared_score, 0.0))
    return {
        "score": score,
        "squared_score": squared_score,
        "human_to_model": human_term,
        "model_spread": model_spread,
        "human_to_samples": human_to_samples,
        "sample_to_sample": sample_to_sample,
    }


def compute_ood_metrics(
    model_samples: np.ndarray,
    human_chunk: np.ndarray,
    config: GeometricDistanceConfig,
) -> dict[str, Any]:
    """Return the Energy score plus descriptive model statistics for the UI."""
    energy = geometric_energy_distance(model_samples, human_chunk, config)
    component_configs = {
        "position": GeometricDistanceConfig(
            position_scales=config.position_scales,
            rotation_scale_rad=config.rotation_scale_rad,
            gripper_scale=config.gripper_scale,
            position_weight=1.0,
            rotation_weight=0.0,
            gripper_weight=0.0,
        ),
        "rotation": GeometricDistanceConfig(
            position_scales=config.position_scales,
            rotation_scale_rad=config.rotation_scale_rad,
            gripper_scale=config.gripper_scale,
            position_weight=0.0,
            rotation_weight=1.0,
            gripper_weight=0.0,
        ),
        "gripper": GeometricDistanceConfig(
            position_scales=config.position_scales,
            rotation_scale_rad=config.rotation_scale_rad,
            gripper_scale=config.gripper_scale,
            position_weight=0.0,
            rotation_weight=0.0,
            gripper_weight=1.0,
        ),
    }
    component_scores = {
        name: float(
            geometric_energy_distance(
                model_samples,
                human_chunk,
                component_config,
            )["score"]
        )
        for name, component_config in component_configs.items()
    }
    component_total = sum(component_scores.values())
    components = {
        name: {
            "score": score,
            "share": score / component_total if component_total > 0.0 else 0.0,
        }
        for name, score in component_scores.items()
    }
    return {
        "energy_distance": energy,
        "components": components,
        # These statistics are descriptive only; they never enter OOD scoring.
        "model_mean": np.asarray(model_samples).mean(axis=0),
        "model_std": np.asarray(model_samples).std(axis=0),
    }


def is_ood(metrics: dict[str, Any], threshold: float) -> bool:
    """Return whether the geometry-aware Energy Distance exceeds threshold."""
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("OOD threshold must be finite and non-negative")
    return float(metrics["energy_distance"]["score"]) > threshold
