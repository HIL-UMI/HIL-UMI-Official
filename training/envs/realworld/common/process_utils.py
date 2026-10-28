# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Helpers for managing child process trees as isolated process groups."""

import os
import signal
import sys
import time
from collections.abc import Sequence
from typing import Any

import psutil


def start_process_group(
    command: Sequence[str],
    *,
    stderr: Any = sys.stderr,
    stdout: Any = sys.stdout,
) -> psutil.Popen:
    """Start a command in a new session so its descendants can be stopped.

    ROS launchers spawn several child processes. Terminating only the launcher
    leaves those children alive, so each launcher must own a process group.

    Args:
        command: Command and arguments to execute.
        stderr: File object used for standard error.
        stdout: File object used for standard output.

    Returns:
        The started process. Its PID is also the new process-group ID.
    """
    process = psutil.Popen(
        list(command),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    return process


def stop_process_group(
    process: psutil.Popen | None,
    *,
    name: str,
    logger: Any,
    interrupt_timeout_s: float = 5.0,
    terminate_timeout_s: float = 2.0,
    kill_timeout_s: float = 1.0,
) -> None:
    """Stop an owned process group with SIGINT, SIGTERM, then SIGKILL.

    Args:
        process: Process returned by :func:`start_process_group`.
        name: Human-readable process name used in log messages.
        logger: Logger with ``debug``, ``warning``, and ``error`` methods.
        interrupt_timeout_s: Seconds to wait after SIGINT.
        terminate_timeout_s: Seconds to wait after SIGTERM.
        kill_timeout_s: Seconds to wait after SIGKILL.
    """
    if process is None:
        return

    # ``start_new_session`` makes the child PID its process-group ID. Use the
    # known PID instead of querying the launcher because it may already have
    # exited while descendants are still running.
    process_group_id = process.pid
    if process_group_id == os.getpgrp():
        raise RuntimeError(
            f"Refusing to stop {name}: it shares process group "
            f"{process_group_id} with the current process."
        )

    process_group_ids = _get_process_group_ids(process, process_group_id)
    stages = (
        (signal.SIGINT, interrupt_timeout_s, "SIGINT"),
        (signal.SIGTERM, terminate_timeout_s, "SIGTERM"),
        (signal.SIGKILL, kill_timeout_s, "SIGKILL"),
    )
    for index, (sig, timeout_s, signal_name) in enumerate(stages):
        process_group_ids.update(_get_process_group_ids(process, process_group_id))
        for owned_process_group_id in process_group_ids:
            try:
                os.killpg(owned_process_group_id, sig)
            except ProcessLookupError:
                continue

        if _wait_process_groups(process, process_group_ids, timeout_s):
            logger.debug("Stopped %s process tree with %s", name, signal_name)
            return

        if index < len(stages) - 1:
            logger.warning(
                "%s process tree did not exit after %s; escalating cleanup",
                name,
                signal_name,
            )

    logger.error(
        "%s process groups %s are still present after SIGKILL",
        name,
        sorted(process_group_ids),
    )


def _get_process_group_ids(
    process: psutil.Popen,
    root_process_group_id: int,
) -> set[int]:
    """Return process groups currently owned by a launcher tree."""
    process_group_ids = {root_process_group_id}
    children = getattr(process, "children", None)
    if children is None:
        return process_group_ids
    try:
        descendants = children(recursive=True)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return process_group_ids

    current_process_group_id = os.getpgrp()
    for descendant in descendants:
        try:
            descendant_process_group_id = os.getpgid(descendant.pid)
        except (PermissionError, ProcessLookupError):
            continue
        if descendant_process_group_id != current_process_group_id:
            process_group_ids.add(descendant_process_group_id)
    return process_group_ids


def _wait_process_groups(
    process: psutil.Popen,
    process_group_ids: set[int],
    timeout_s: float,
) -> bool:
    """Return whether all owned process groups disappeared before timeout."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        _reap_parent(process)
        live_process_group_ids = set()
        for process_group_id in process_group_ids:
            try:
                os.killpg(process_group_id, 0)
            except ProcessLookupError:
                continue
            live_process_group_ids.add(process_group_id)
        if not live_process_group_ids:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _reap_parent(process: psutil.Popen) -> None:
    """Reap a completed launcher without waiting for a running process."""
    try:
        process.wait(timeout=0)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        pass
