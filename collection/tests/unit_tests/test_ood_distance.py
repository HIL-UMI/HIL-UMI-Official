# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from ood.distance import (
    GeometricDistanceConfig,
    compute_ood_metrics,
    geometric_chunk_distance,
    geometric_energy_distance,
    geometric_step_squared_distance,
    is_ood,
    load_action_scales,
    rotation6d_to_matrix,
)


def _identity_action() -> np.ndarray:
    return np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        dtype=np.float64,
    )


def _config(
    *,
    position_weight: float = 1.0 / 3.0,
    rotation_weight: float = 1.0 / 3.0,
    gripper_weight: float = 1.0 / 3.0,
) -> GeometricDistanceConfig:
    return GeometricDistanceConfig(
        position_scales=np.ones(3),
        rotation_scale_rad=math.radians(10.0),
        gripper_scale=1.0,
        position_weight=position_weight,
        rotation_weight=rotation_weight,
        gripper_weight=gripper_weight,
    )


def _rotation_action(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    action = _identity_action()
    action[3:9] = [
        math.cos(angle),
        math.sin(angle),
        0.0,
        -math.sin(angle),
        math.cos(angle),
        0.0,
    ]
    return action


def test_rotation6d_orthogonalization_is_parameterization_invariant():
    canonical = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    scaled_and_skewed = np.array([2.0, 0.0, 0.0, 3.0, 4.0, 0.0])

    np.testing.assert_allclose(
        rotation6d_to_matrix(canonical),
        rotation6d_to_matrix(scaled_and_skewed),
        atol=1e-12,
    )


def test_rotation_at_configured_angle_has_unit_normalized_distance():
    config = _config(position_weight=0.0, rotation_weight=1.0, gripper_weight=0.0)

    distance_squared = geometric_step_squared_distance(
        _identity_action(), _rotation_action(10.0), config
    )

    assert distance_squared == pytest.approx(1.0, abs=1e-12)


def test_position_and_gripper_use_configured_scales():
    position_config = GeometricDistanceConfig(
        position_scales=np.array([0.01, 0.02, 0.03]),
        rotation_scale_rad=math.radians(10.0),
        gripper_scale=0.5,
        position_weight=1.0,
        rotation_weight=0.0,
        gripper_weight=0.0,
    )
    action = _identity_action()
    action[:3] = [0.01, 0.02, 0.03]
    assert geometric_step_squared_distance(
        _identity_action(), action, position_config
    ) == pytest.approx(1.0)

    gripper_config = GeometricDistanceConfig(
        position_scales=np.ones(3),
        rotation_scale_rad=math.radians(10.0),
        gripper_scale=0.5,
        position_weight=0.0,
        rotation_weight=0.0,
        gripper_weight=1.0,
    )
    action = _identity_action()
    action[9] = 0.5
    assert geometric_step_squared_distance(
        _identity_action(), action, gripper_config
    ) == pytest.approx(1.0)


def test_chunk_distance_uses_uniform_timestep_weights():
    config = _config(position_weight=1.0, rotation_weight=0.0, gripper_weight=0.0)
    first = np.broadcast_to(_identity_action(), (4, 10)).copy()
    second = first.copy()
    second[0, :3] = 1.0

    distance = geometric_chunk_distance(first, second, config)

    assert distance == pytest.approx(0.5)


def test_energy_is_zero_for_identical_human_and_model_chunks():
    human = np.broadcast_to(_identity_action(), (3, 10)).copy()
    samples = np.broadcast_to(human, (10, 3, 10)).copy()

    result = geometric_energy_distance(samples, human, _config())

    assert result["score"] == pytest.approx(0.0)
    assert result["squared_score"] == pytest.approx(0.0)


def test_energy_applies_outer_square_root_from_reference_formula():
    config = _config(position_weight=1.0, rotation_weight=0.0, gripper_weight=0.0)
    human = _identity_action()[None, :]
    model_action = _identity_action()
    model_action[:3] = 2.0
    samples = np.broadcast_to(model_action, (10, 1, 10)).copy()

    result = geometric_energy_distance(samples, human, config)

    assert result["human_to_model"] == pytest.approx(2.0)
    assert result["model_spread"] == pytest.approx(0.0)
    assert result["squared_score"] == pytest.approx(4.0)
    assert result["score"] == pytest.approx(2.0)


def test_energy_preserves_multimodal_samples_and_subtracts_model_spread():
    config = _config(position_weight=1.0, rotation_weight=0.0, gripper_weight=0.0)
    left = _identity_action()
    left[:3] = -1.0
    right = _identity_action()
    right[:3] = 1.0
    samples = np.stack([left, right], axis=0)[:, None, :]
    human = right[None, :]

    result = geometric_energy_distance(samples, human, config)

    assert result["human_to_model"] == pytest.approx(1.0)
    assert result["model_spread"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(1.0)


def test_compute_metrics_mean_is_descriptive_only():
    human = _identity_action()[None, :]
    samples = np.broadcast_to(human, (4, 1, 10)).copy()

    metrics = compute_ood_metrics(samples, human, _config())

    assert metrics["energy_distance"]["score"] == pytest.approx(0.0)
    assert metrics["model_mean"].shape == (1, 10)
    assert set(metrics["components"]) == {"position", "rotation", "gripper"}
    assert all(
        component["score"] == pytest.approx(0.0)
        and component["share"] == pytest.approx(0.0)
        for component in metrics["components"].values()
    )
    assert not is_ood(metrics, threshold=0.1)
    assert is_ood(
        {
            "energy_distance": {"score": 0.2},
        },
        threshold=0.1,
    )


def test_component_energy_scores_are_independent_and_shares_are_normalized():
    human = _identity_action()[None, :]
    model = _identity_action()
    model[:3] = 1.0
    model[9] = 2.0
    samples = np.broadcast_to(model, (4, 1, 10)).copy()

    metrics = compute_ood_metrics(samples, human, _config())

    assert metrics["components"]["position"]["score"] > 0.0
    assert metrics["components"]["gripper"]["score"] > 0.0
    assert metrics["components"]["rotation"]["score"] == pytest.approx(0.0)
    assert sum(
        component["share"] for component in metrics["components"].values()
    ) == pytest.approx(1.0)


def test_load_action_scales_reads_checkpoint_layout(tmp_path):
    stats = np.arange(1.0, 33.0)
    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps({"norm_stats": {"actions": {"std": stats.tolist()}}}),
        encoding="utf-8",
    )

    position_scales, gripper_scale = load_action_scales(tmp_path)

    np.testing.assert_array_equal(position_scales, [1.0, 2.0, 3.0])
    assert gripper_scale == pytest.approx(10.0)


@pytest.mark.parametrize(
    "rotation6d",
    [
        np.zeros(6),
        np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
    ],
)
def test_degenerate_rotation6d_raises(rotation6d):
    with pytest.raises(ValueError, match="degenerate"):
        rotation6d_to_matrix(rotation6d)


def test_bad_shapes_and_invalid_scales_raise(tmp_path):
    with pytest.raises(ValueError, match="position_scales"):
        GeometricDistanceConfig(
            position_scales=np.ones(2),
            rotation_scale_rad=0.1,
            gripper_scale=1.0,
        )
    with pytest.raises(ValueError, match="model_samples"):
        geometric_energy_distance(
            np.zeros((2, 10)),
            np.zeros((1, 10)),
            _config(),
        )

    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "actions": {"std": [1.0, 1.0, 1.0] + [0.0] * 7}
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite and positive"):
        load_action_scales(path)
