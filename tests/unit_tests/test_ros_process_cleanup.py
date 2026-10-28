# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stopping complete ROS process trees."""

import signal
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training.envs.realworld import realworld_env as realworld_env_module
from training.envs.realworld.common import process_utils
from training.envs.realworld.realworld_env import RealWorldEnv


def test_realworld_setup_preserves_a_responsive_master(monkeypatch):
    process_iter = Mock()
    rosgraph = SimpleNamespace(is_master_online=lambda: True)
    monkeypatch.setitem(sys.modules, "rosgraph", rosgraph)
    monkeypatch.setattr(realworld_env_module.psutil, "process_iter", process_iter)

    RealWorldEnv.realworld_setup()

    process_iter.assert_not_called()


def test_start_process_group_creates_a_new_session(monkeypatch):
    process = SimpleNamespace(pid=1234)
    popen = Mock(return_value=process)
    monkeypatch.setattr(process_utils.psutil, "Popen", popen)

    result = process_utils.start_process_group(
        ("roslaunch", "package", "controller.launch")
    )

    assert result is process
    popen.assert_called_once_with(
        ["roslaunch", "package", "controller.launch"],
        stdout=process_utils.sys.stdout,
        stderr=process_utils.sys.stderr,
        start_new_session=True,
    )


def test_stop_process_group_escalates_until_the_group_exits(monkeypatch):
    process = SimpleNamespace(pid=4321)
    logger = Mock()
    sent_signals = []
    wait_results = iter((False, False, True))
    monkeypatch.setattr(process_utils.os, "getpgrp", lambda: 9999)
    monkeypatch.setattr(
        process_utils.os,
        "killpg",
        lambda process_group_id, sig: sent_signals.append((process_group_id, sig)),
    )
    monkeypatch.setattr(
        process_utils,
        "_wait_process_groups",
        lambda _process, _process_group_ids, _timeout_s: next(wait_results),
    )

    process_utils.stop_process_group(process, name="test ROS", logger=logger)

    assert sent_signals == [
        (4321, signal.SIGINT),
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]
    assert logger.warning.call_count == 2
    logger.debug.assert_called_once()
    logger.error.assert_not_called()


def test_stop_process_group_signals_descendant_process_groups(monkeypatch):
    descendants = [SimpleNamespace(pid=5001), SimpleNamespace(pid=5002)]
    process = SimpleNamespace(pid=4321, children=lambda recursive: descendants)
    logger = Mock()
    sent_signals = []
    process_group_ids = {5001: 5001, 5002: 5001}
    monkeypatch.setattr(process_utils.os, "getpgrp", lambda: 9999)
    monkeypatch.setattr(
        process_utils.os,
        "getpgid",
        lambda pid: process_group_ids[pid],
    )
    monkeypatch.setattr(
        process_utils.os,
        "killpg",
        lambda process_group_id, sig: sent_signals.append((process_group_id, sig)),
    )
    monkeypatch.setattr(
        process_utils,
        "_wait_process_groups",
        lambda _process, _process_group_ids, _timeout_s: True,
    )

    process_utils.stop_process_group(process, name="test ROS", logger=logger)

    assert set(sent_signals) == {
        (4321, signal.SIGINT),
        (5001, signal.SIGINT),
    }


def test_stop_process_group_refuses_to_signal_its_own_group(monkeypatch):
    process = SimpleNamespace(pid=4321)
    monkeypatch.setattr(process_utils.os, "getpgrp", lambda: 4321)
    killpg = Mock()
    monkeypatch.setattr(process_utils.os, "killpg", killpg)

    with pytest.raises(RuntimeError, match="shares process group"):
        process_utils.stop_process_group(process, name="test ROS", logger=Mock())

    killpg.assert_not_called()
