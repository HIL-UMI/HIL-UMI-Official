# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the weights-only collection post-train schedule."""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_LAUNCHER = REPO_ROOT / "scripts" / "run_vla_sft.sh"


def test_safe_post_train_cosine_schedule_is_bounded():
    """The fresh post-train schedule must never exceed the stage-1 final LR."""
    torch = pytest.importorskip("torch")
    from training.hybrid_engines.fsdp.utils import get_lr_scheduler

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=2.5e-6)
    scheduler = get_lr_scheduler(
        "openpi_cosine",
        optimizer,
        num_warmup_steps=500,
        num_training_steps=20_000,
        min_lr=2.5e-7,
    )

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(20_000):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert max(learning_rates) == pytest.approx(2.5e-6)
    assert learning_rates[500] == pytest.approx(2.5e-6)
    assert learning_rates[-1] == pytest.approx(2.5e-7)


def test_weights_only_post_train_dry_run_uses_safe_defaults_and_ratio_name(tmp_path):
    """The launcher should resolve a fresh schedule without resume state."""
    model_dir = tmp_path / "stage1-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").touch()

    result = subprocess.run(
        [
            "bash",
            str(SFT_LAUNCHER),
            "umi_left_franka_sft_openpi_pytorch_tcp_rot6d",
            "--task_name",
            "stack_cubes_umi",
            "--post-train-model-path",
            str(model_dir),
            "--post-train-data-path",
            "/data/corrections",
            "--post-train-demo-ratio",
            "0.7",
            "--post-train-correction-ratio",
            "0.3",
            "--norm_stats_path",
            "/stats/post_train_70_30",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    command = result.stdout
    assert "data.post_train=true" in command
    assert "data.post_train_demo_ratio=0.7" in command
    assert "data.post_train_correction_ratio=0.3" in command
    assert (
        "runner.logger.experiment_name="
        "umi_left_franka_tcp_rot6d_openpi_pytorch_post-train-weights-only7-3"
    ) in command
    assert f"actor.model.model_path={model_dir}" in command
    assert "runner.resume_dir=null" in command
    assert "runner.max_steps=20000" in command
    assert "actor.optim.total_training_steps=20000" in command
    assert "actor.optim.lr=2.5e-6" in command
    assert "actor.optim.min_lr=2.5e-7" in command
    assert "actor.optim.lr_warmup_steps=500" in command


def test_weights_only_post_train_rejects_full_resume(tmp_path):
    """Weights-only initialization and full-state resume are mutually exclusive."""
    model_dir = tmp_path / "stage1-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").touch()

    result = subprocess.run(
        [
            "bash",
            str(SFT_LAUNCHER),
            "umi_left_franka_sft_openpi_pytorch_tcp_rot6d",
            "--post-train-model-path",
            str(model_dir),
            "runner.resume_dir=/checkpoints/global_step_20000",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode != 0
    assert "cannot be combined with runner.resume_dir" in result.stdout


def test_launcher_selects_fast_no_padding_norm_stats(tmp_path):
    """The training launcher should resolve the dedicated GPU stats directory."""
    assets_dir = tmp_path / "assets"
    stats_dir = (
        assets_dir
        / "umi_left_franka_tcp_rot6d"
        / "stack_cubes_round2_no_padding_gpu"
    )
    stats_dir.mkdir(parents=True)
    (stats_dir / "norm_stats.json").write_text("{}", encoding="utf-8")
    env = os.environ.copy()
    env["UMI_LEFT_FRANKA_ASSETS_DIR"] = str(assets_dir)

    result = subprocess.run(
        [
            "bash",
            str(SFT_LAUNCHER),
            "umi_left_franka_sft_openpi_pytorch_tcp_rot6d",
            "--task_name",
            "stack_cubes_round2",
            "--data_path",
            "/data/demo-plus-round1",
            "--post-train",
            "true",
            "--post-train-data-path",
            "/data/round2",
            "--fast-norm-stats",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )

    assert f"Using fast no-padding norm stats: {stats_dir}" in result.stdout
    assert f"actor.model.openpi.assets_dir={stats_dir.parent}" in result.stdout
    assert (
        "actor.model.openpi.asset_id=stack_cubes_round2_no_padding_gpu"
        in result.stdout
    )
    assert f"python {REPO_ROOT}/examples/sft/train_vla_sft.py" in result.stdout


def test_launcher_rejects_conflicting_norm_stats_modes():
    """Fast naming and an explicit stats path must not silently compete."""
    result = subprocess.run(
        [
            "bash",
            str(SFT_LAUNCHER),
            "umi_left_franka_sft_openpi_pytorch_tcp_rot6d",
            "--task_name",
            "stack_cubes_round2",
            "--fast-norm-stats",
            "--norm_stats_path",
            "/stats/manual",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode != 0
    assert "cannot be combined with --norm_stats_path" in result.stdout
