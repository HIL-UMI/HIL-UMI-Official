# Copyright 2026 HIL-UMI contributors.
# SPDX-License-Identifier: Apache-2.0

"""Source launchers must preview safely without a robot, ROS, models or Ray."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run(
    script: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    settings = dict(os.environ)
    for name in (
        "TRAINING_NODE_RANK",
        "RAY_ADDRESS",
        "RAY_PORT",
        "EVAL_PYTHON",
        "TRAINING_EVAL_PYTHON",
        "POLICY_PYTHON",
        "OPENPI_PYTHON",
        "FRANKA_PYTHON",
    ):
        settings.pop(name, None)
    settings.update(env or {})
    return subprocess.run(
        ["bash", str(ROOT / script), *args],
        text=True,
        capture_output=True,
        env=settings,
        cwd=ROOT,
    )


def command_from_preview(output: str) -> list[str]:
    return shlex.split(output.split("Dry run (command only): ")[-1].strip())


def test_all_retained_shell_entry_points_have_valid_syntax() -> None:
    for folder in ("scripts", "examples", "evaluations", "ray_utils"):
        for path in (ROOT / folder).rglob("*.sh"):
            result = subprocess.run(
                ["bash", "-n", str(path)], text=True, capture_output=True
            )
            assert result.returncode == 0, f"{path}: {result.stderr}"


def test_evaluator_previews_with_missing_interpreter_and_preserves_arguments(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "must-not-exist"
    override = (
        "env.eval.override_cfg.task_description='fold the towel, Advantage: positive'"
    )
    result = run(
        "evaluations/run_eval.sh",
        "realworld",
        "umi_left_franka_tcp_rot6d_eval",
        "--dry-run",
        override,
        env={"EVAL_PYTHON": "/not/installed/python", "EVAL_LOG_DIR": str(logs)},
    )
    assert result.returncode == 0, result.stderr
    command = command_from_preview(result.stdout)
    assert command[0] == "/not/installed/python"
    assert override in command
    assert not logs.exists()


def test_evaluator_rejects_removed_benchmarks() -> None:
    result = run("evaluations/run_eval.sh", "libero", "removed", "--dry-run")
    assert result.returncode == 2
    assert "realworld" in result.stderr


def test_franka_preview_needs_no_ros_model_or_devices(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    result = run(
        "evaluations/run_umi_left_franka_eval.sh",
        "--model_path",
        "/not/installed/model",
        "--norm_stats_path",
        "/not/installed/stats",
        "--dry-run",
        env={
            "EVAL_LOG_DIR": str(logs),
            "ROS_NOETIC_SETUP": "/not/installed/ros",
            "FRANKA_CATKIN_SETUP": "/not/installed/catkin",
        },
    )
    assert result.returncode == 0, result.stderr
    command = command_from_preview(result.stdout)
    assert "rollout.model.model_path=/not/installed/model" in command
    assert "rollout.model.openpi_data.norm_stats_path=/not/installed/stats" in command
    assert not logs.exists()


def test_franka_help_and_missing_option_value_are_actionable() -> None:
    assert run("evaluations/run_umi_left_franka_eval.sh", "--help").returncode == 0
    result = run("evaluations/run_umi_left_franka_eval.sh", "--model_path")
    assert result.returncode == 2
    assert "Missing value for --model_path" in result.stderr


@pytest.mark.parametrize(
    ("args", "env", "expected"),
    [
        ([], {}, str(ROOT / ".venv-policy/bin/python")),
        ([], {"POLICY_PYTHON": "/custom/policy/python"}, "/custom/policy/python"),
        ([], {"OPENPI_PYTHON": "/legacy/python"}, "/legacy/python"),
        (
            [],
            {"POLICY_PYTHON": "/custom/policy/python", "OPENPI_PYTHON": "/legacy/python"},
            "/custom/policy/python",
        ),
        (["--policy_python", "/custom policy/python"], {}, "/custom policy/python"),
        (["--openpi_python=/legacy/python"], {}, "/legacy/python"),
    ],
)
def test_franka_policy_environment_selection(args, env, expected) -> None:
    result = run(
        "evaluations/run_umi_left_franka_eval.sh",
        "--model_path",
        "/not/installed/model",
        "--norm_stats_path",
        "/not/installed/stats",
        "--dry-run",
        *args,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert command_from_preview(result.stdout)[0] == expected


def test_installer_defaults_to_distinct_policy_and_robot_environments() -> None:
    for target, name in (("embodied", ".venv-policy"), ("franka", ".venv-franka")):
        result = run("requirements/install.sh", target, "--dry-run")
        assert result.returncode == 0, result.stderr
        assert str(ROOT / name) in result.stdout


def test_ray_preview_has_explicit_node_rank_and_no_machine_memory_override() -> None:
    head = run(
        "ray_utils/start_ray.sh",
        "--dry-run",
        env={"TRAINING_NODE_RANK": "0", "RAY_PORT": "6380"},
    )
    worker = run(
        "ray_utils/start_ray.sh",
        "--dry-run",
        env={"TRAINING_NODE_RANK": "1", "RAY_ADDRESS": "ray-head.example:6380"},
    )
    assert head.returncode == worker.returncode == 0
    assert "--head" in head.stdout and "--port=6380" in head.stdout
    assert "--address=ray-head.example:6380" in worker.stdout
    assert "--memory" not in head.stdout + worker.stdout
    assert not (ROOT / "ray_utils/ray_head_ip.txt").exists()


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"TRAINING_NODE_RANK": "-1"},
        {"TRAINING_NODE_RANK": "1"},
        {"TRAINING_NODE_RANK": "0", "RAY_PORT": "99999"},
        {"TRAINING_NODE_RANK": "1", "RAY_ADDRESS": "head:invalid"},
    ],
)
def test_ray_rejects_incomplete_configuration_without_starting(
    env: dict[str, str],
) -> None:
    result = run("ray_utils/start_ray.sh", "--dry-run", env=env)
    assert result.returncode == 2


def test_sft_and_evaluation_configs_share_the_single_franka_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    for key, value in {
        "HIL_UMI_ROOT": str(ROOT),
        "EMBODIED_PATH": str(ROOT / "examples/sft"),
        "PI05_DEPLOY_CKPT": "/path/to/model",
        "UMI_LEFT_FRANKA_NORM_STATS": "/path/to/stats",
        "FRANKA_ROBOT_IP": "robot.example",
        "FRANKA_WRIST_CAMERA_SERIAL": "WRIST_SERIAL",
        "FRANKA_FRONT_CAMERA_SERIAL": "FRONT_SERIAL",
        "POLICY_PYTHON": "/custom/policy/bin/python",
        "FRANKA_PYTHON": "/custom/franka/bin/python",
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "evaluations/realworld")
    ):
        evaluation = compose(config_name="umi_left_franka_tcp_rot6d_eval")
    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "examples/sft/config")
    ):
        sft = compose(config_name="umi_left_franka_sft_openpi_pytorch_tcp_rot6d")
    for model in (sft.actor.model, evaluation.rollout.model):
        assert model.model_type == "openpi_pytorch"
        assert model.action_dim == 10
        assert model.num_action_chunks == 20
        assert model.openpi.config_name == "pi05_singlefranka_tcp_rot6d"
    assert sft.actor.model.openpi.task == "sft"
    assert evaluation.rollout.model.openpi.task == "eval"
    assert evaluation.env.eval.init_params.id == "SingleFrankaTCPEnv-v1"
    assert evaluation.env.eval.use_vive is False
    assert evaluation.cluster.node_groups[0].env_configs[0].python_interpreter_path == (
        "/custom/policy/bin/python"
    )
    assert evaluation.cluster.node_groups[1].env_configs[0].python_interpreter_path == (
        "/custom/franka/bin/python"
    )
    assert sft.data.acbc is False
    assert sft.data.mix_acbc is False
    OmegaConf.to_container(evaluation, resolve=True)
    OmegaConf.to_container(sft, resolve=True)
