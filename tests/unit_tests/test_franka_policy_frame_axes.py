# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation as R

from toolkits.realworld_check.check_franka_policy_frame_axes import (
    DEFAULT_CONFIG,
    PolicyFrameAxisConfig,
    build_policy_axis_targets,
    interpolate_pose_xyzw,
    load_axis_test_config,
    pose_xyzw_to_transform,
    validate_workspace_target,
)


def _config(**overrides) -> PolicyFrameAxisConfig:
    values = {
        "reset_base_to_ee": np.eye(4),
        "ee_to_policy": np.eye(4),
        "workspace_min": np.array([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
        "workspace_max": np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
        "orientation_center": np.zeros(3),
        "frequency_hz": 30.0,
        "compliance_params": {},
    }
    values.update(overrides)
    return PolicyFrameAxisConfig(**values)


def test_axis_targets_are_local_policy_translations() -> None:
    ee_to_policy = np.eye(4)
    ee_to_policy[:3, :3] = R.from_euler("z", 90, degrees=True).as_matrix()
    ee_to_policy[:3, 3] = [0.01, 0.02, 0.03]
    config = _config(ee_to_policy=ee_to_policy)

    targets = build_policy_axis_targets(config, ["x+", "y-", "z+"], 0.05)
    base_to_policy_reset = config.reset_base_to_ee @ config.ee_to_policy

    expected = ([0.05, 0.0, 0.0], [0.0, -0.05, 0.0], [0.0, 0.0, 0.05])
    for target, expected_delta in zip(targets, expected, strict=True):
        base_to_policy_target = target.base_to_ee @ config.ee_to_policy
        relative = np.linalg.inv(base_to_policy_reset) @ base_to_policy_target
        assert np.allclose(relative[:3, 3], expected_delta, atol=1e-9)
        assert np.allclose(relative[:3, :3], np.eye(3), atol=1e-9)


def test_axis_distance_has_hard_five_centimeter_cap() -> None:
    with pytest.raises(ValueError, match=r"distance_m"):
        build_policy_axis_targets(_config(), ["x+"], 0.051)


def test_workspace_rejects_out_of_bounds_target() -> None:
    target = np.eye(4)
    target[0, 3] = 0.21
    with pytest.raises(ValueError, match=r"outside the configured workspace"):
        validate_workspace_target(target, _config(), label="outside")


def test_interpolation_preserves_endpoints_and_short_rotation() -> None:
    start = np.r_[[0.0, 0.0, 0.0], R.identity().as_quat()]
    target = np.r_[[0.1, -0.2, 0.3], R.from_euler("z", 0.4).as_quat()]
    path = interpolate_pose_xyzw(start, target, np.array([0.0, 0.5, 1.0]))

    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], target)
    halfway = pose_xyzw_to_transform(path[1])
    assert np.allclose(halfway[:3, 3], [0.05, -0.1, 0.15])
    assert R.from_matrix(halfway[:3, :3]).magnitude() == pytest.approx(0.2)


def test_deployment_config_all_six_targets_are_inside_workspace() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_axis_test_config(
        repo_root
        / DEFAULT_CONFIG
    )
    targets = build_policy_axis_targets(
        config,
        ["x+", "x-", "y+", "y-", "z+", "z-"],
        0.05,
    )

    assert len(targets) == 6
    for target in targets:
        assert np.linalg.norm(target.base_delta_m) == pytest.approx(0.05)


def test_deployment_config_composes_saved_controller_action_frame() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / DEFAULT_CONFIG
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))["override_cfg"]
    ee_to_policy_tool = np.asarray(raw["franka_tcp_to_policy_tool"], dtype=np.float64)
    controller_to_umi_tool = np.asarray(
        raw["oculus_controller_to_umi_tool"], dtype=np.float64
    )

    ee_to_controller = ee_to_policy_tool @ np.linalg.inv(controller_to_umi_tool)
    expected_ee_to_controller = np.array(
        [
            [0.549848417, 0.565977749, -0.614276728, 0.008374356],
            [0.795424049, -0.130386596, 0.591861417, 0.001559043],
            [0.254886944, -0.814044537, -0.521885182, -0.061800428],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(ee_to_controller, expected_ee_to_controller, atol=1e-8)
