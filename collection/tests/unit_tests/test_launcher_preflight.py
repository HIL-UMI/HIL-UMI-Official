# Copyright 2026 The HIL-UMI Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate launch behavior without importing or running private hardware code."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from ood.preflight import validate_recorder_config

ROOT = Path(__file__).resolve().parents[2]


def configured_pair():
    with (ROOT / "config/recorder.example.yaml").open() as file:
        recorder = yaml.safe_load(file)
    with (ROOT / "config/ood.yaml").open() as file:
        monitor = yaml.safe_load(file)
    recorder["collection_metadata"]["collection_device_id"] = "synthetic-test-rig"
    for index, camera in enumerate(recorder["cameras"]):
        camera["serial"] = f"synthetic-camera-{index}"
    return recorder, monitor


def test_example_must_be_configured_before_any_devices_start():
    recorder, monitor = configured_pair()
    recorder["cameras"][0]["serial"] = ""
    with pytest.raises(ValueError, match="Set serial"):
        validate_recorder_config(recorder, monitor)


@pytest.mark.parametrize("field,value", [("width", 320), ("ring_slots", 20)])
def test_shared_memory_layout_must_match_recorder(field, value):
    recorder, monitor = configured_pair()
    recorder["cameras"][0][field] = value
    with pytest.raises(ValueError, match=f"disagree on {field}"):
        validate_recorder_config(recorder, monitor)


def test_prompts_must_match():
    recorder, monitor = configured_pair()
    recorder["task_description"] = "different task"
    with pytest.raises(ValueError, match="prompt"):
        validate_recorder_config(recorder, monitor)


def test_temporal_reference_fits_within_ring():
    recorder, monitor = configured_pair()
    monitor["detection"] = {"frame_offset": 60}
    with pytest.raises(ValueError, match="frame_offset"):
        validate_recorder_config(recorder, monitor)


def test_public_launcher_reports_missing_external_dependency(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "RECORDER_PYTHON",
            "RECORD_BIN",
            "OOD_PYTHON",
            "POLICY_PYTHON",
            "ADVANTAGE_PYTHON",
        }
    }
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "run_ood_teleop.sh"),
            str(ROOT / "config/recorder.example.yaml"),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "set RECORDER_PYTHON and RECORD_BIN" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["pi05", "advantage"])
def test_check_mode_does_not_start_recorder_and_works_outside_repo(tmp_path, mode):
    trace = tmp_path / "trace"
    fake_python = tmp_path / "fake python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TRACE"\n')
    fake_python.chmod(0o755)
    record = tmp_path / "record"
    record.write_text('#!/bin/sh\nprintf "RECORDER_STARTED\\n" >> "$TRACE"\n')
    record.chmod(0o755)
    env = dict(
        os.environ,
        RECORDER_PYTHON=str(fake_python),
        RECORD_BIN=str(record),
        OOD_PYTHON=str(fake_python),
        TRACE=str(trace),
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "run_ood_teleop.sh"),
            "--mode",
            mode,
            "--check",
            str(ROOT / "config/recorder.example.yaml"),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    logged = trace.read_text()
    assert "ood.preflight" in logged
    assert f"--mode {mode}" in logged
    assert "RECORDER_STARTED" not in logged
    assert "Preflight passed" in result.stdout


@pytest.mark.parametrize(
    "mode,variable", [("pi05", "POLICY_PYTHON"), ("advantage", "ADVANTAGE_PYTHON")]
)
@pytest.mark.parametrize("override", [False, True])
def test_launcher_selects_mode_environment_and_explicit_override(
    tmp_path, mode, variable, override
):
    trace = tmp_path / "trace"
    interpreters = {}
    for name in ("recorder", "policy", "advantage", "override"):
        path = tmp_path / f"{name}-python"
        path.write_text(f'#!/bin/sh\nprintf "{name} %s\\n" "$*" >> "$TRACE"\n')
        path.chmod(0o755)
        interpreters[name] = str(path)
    record = tmp_path / "record"
    record.write_text('#!/bin/sh\nprintf "RECORDER_STARTED\\n" >> "$TRACE"\n')
    record.chmod(0o755)
    env = dict(
        os.environ,
        RECORDER_PYTHON=interpreters["recorder"],
        RECORD_BIN=str(record),
        POLICY_PYTHON=interpreters["policy"],
        ADVANTAGE_PYTHON=interpreters["advantage"],
        TRACE=str(trace),
    )
    env.pop("OOD_PYTHON", None)
    if override:
        env["OOD_PYTHON"] = interpreters["override"]
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "run_ood_teleop.sh"),
            "--mode",
            mode,
            "--check",
            str(ROOT / "config/recorder.example.yaml"),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    monitor_calls = [
        line for line in trace.read_text().splitlines() if "-m ood.preflight" in line
    ]
    selected = (
        "override"
        if override
        else {"POLICY_PYTHON": "policy", "ADVANTAGE_PYTHON": "advantage"}[variable]
    )
    assert len(monitor_calls) == 1
    assert monitor_calls[0].startswith(f"{selected} ")
    assert "RECORDER_STARTED" not in trace.read_text()


@pytest.mark.parametrize(
    "mode,required,other",
    [
        ("pi05", "POLICY_PYTHON", "ADVANTAGE_PYTHON"),
        ("advantage", "ADVANTAGE_PYTHON", "POLICY_PYTHON"),
    ],
)
def test_launcher_does_not_fall_back_to_other_model_environment(
    tmp_path, mode, required, other
):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OOD_PYTHON", "POLICY_PYTHON", "ADVANTAGE_PYTHON"}
    }
    env.update(
        RECORDER_PYTHON="/not-used", RECORD_BIN="/not-used", **{other: "/other-model"}
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "run_ood_teleop.sh"),
            "--mode",
            mode,
            "--check",
            str(ROOT / "config/recorder.example.yaml"),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"Set {required}" in result.stderr
    assert not list(tmp_path.iterdir())


def test_socket_override_is_translated_only_for_external_recorder(tmp_path):
    trace = tmp_path / "trace"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/bin/sh\nexit 0\n")
    fake_python.chmod(0o755)
    record = tmp_path / "record"
    record.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$EGODAGGER_OOD_SOCKET_PATH" "$COLLECTION_OOD_SOCKET_PATH" "$*" > "$TRACE"\n'
    )
    record.chmod(0o755)
    env = dict(
        os.environ,
        RECORDER_PYTHON=str(fake_python),
        RECORD_BIN=str(record),
        OOD_PYTHON=str(fake_python),
        TRACE=str(trace),
        COLLECTION_OOD_SOCKET_PATH="/tmp/synthetic-collection-test.sock",
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "run_ood_teleop.sh"),
            str(ROOT / "config/recorder.example.yaml"),
            "--debug",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = trace.read_text().splitlines()
    assert lines[0] == lines[1] == env["COLLECTION_OOD_SOCKET_PATH"]
    assert lines[2].endswith(" --debug")


def test_action_frequency_matches_recorder_time_axis():
    recorder, monitor = configured_pair()
    recorder["fps"] = 15
    with pytest.raises(ValueError, match="action_frequency_hz"):
        validate_recorder_config(recorder, monitor)


def test_both_monitors_use_public_socket_override(tmp_path, monkeypatch):
    from ood.advantage_monitor import _load_config as load_advantage
    from ood.monitor import _load_config as load_policy

    config = tmp_path / "monitor.yaml"
    config.write_text("source:\n  socket_path: /tmp/default-synthetic.sock\n")
    monkeypatch.setenv("COLLECTION_OOD_SOCKET_PATH", "/tmp/overridden-synthetic.sock")
    assert (
        load_policy(str(config))["source"]["socket_path"]
        == "/tmp/overridden-synthetic.sock"
    )
    assert (
        load_advantage(str(config))["source"]["socket_path"]
        == "/tmp/overridden-synthetic.sock"
    )
