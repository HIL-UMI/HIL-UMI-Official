"""The launcher selects an existing environment without rebuilding one."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_works_from_another_working_directory(tmp_path):
    env = os.environ.copy()
    env["ADVANTAGE_PYTHON"] = sys.executable
    env["ADVANTAGE_DATA_ROOT"] = str(tmp_path / "external_data")
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "advantage"),
            "python",
            "-c",
            (
                "import json, os; from hil_umi_advantage.shared import paths; "
                "print(json.dumps([str(paths.repo_root()), str(paths.data_root()), os.getenv('WANDB_MODE')]))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == [str(ROOT), str(tmp_path / "external_data"), env.get("WANDB_MODE", "offline")]
    assert not (ROOT / ".runtime/venv").exists()


def test_missing_interpreter_has_actionable_error_and_does_not_create_it(tmp_path):
    missing = tmp_path / "missing_env/bin/python"
    env = os.environ.copy()
    env["ADVANTAGE_PYTHON"] = str(missing)
    result = subprocess.run(
        ["bash", str(ROOT / "advantage"), "doctor"], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "ADVANTAGE_PYTHON" in result.stderr
    assert "uv sync" in result.stderr
    assert not missing.parent.exists()


def test_help_does_not_require_a_runtime(tmp_path):
    env = os.environ.copy()
    env["ADVANTAGE_PYTHON"] = str(tmp_path / "missing")
    result = subprocess.run(
        ["bash", str(ROOT / "advantage"), "--help"], env=env, capture_output=True, text=True, check=True
    )
    assert "train-mix" in result.stdout


def test_default_environment_is_distinct_from_policy_and_old_component_venv(tmp_path):
    component = tmp_path / "release/advantage"
    component.mkdir(parents=True)
    canonical_python = component.parent / ".venv-advantage/bin/python"
    canonical_python.parent.mkdir(parents=True)
    canonical_python.symlink_to(sys.executable)
    for other in (component / ".venv", component.parent / ".venv-policy"):
        (other / "bin").mkdir(parents=True)
        wrong_python = other / "bin/python"
        wrong_python.write_text("#!/usr/bin/env bash\nexit 99\n")
        wrong_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("ADVANTAGE_PYTHON", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; advantage_init_runtime "$2"; printf "%s\\n" "$ADVANTAGE_PYTHON"',
            "bash",
            str(ROOT / "scripts/source_runtime.sh"),
            str(component),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(canonical_python)


def test_missing_default_never_falls_back_to_old_environment(tmp_path):
    component = tmp_path / "release/advantage"
    old_python = component / ".venv/bin/python"
    old_python.parent.mkdir(parents=True)
    old_python.symlink_to(sys.executable)
    env = os.environ.copy()
    env.pop("ADVANTAGE_PYTHON", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; advantage_init_runtime "$2"',
            "bash",
            str(ROOT / "scripts/source_runtime.sh"),
            str(component),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert str(component.parent / ".venv-advantage/bin/python") in result.stderr
    assert "UV_PROJECT_ENVIRONMENT=" in result.stderr
    assert old_python.is_symlink()
    assert not (component.parent / ".venv-advantage").exists()
