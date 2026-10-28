# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for environment worker cleanup."""

from unittest.mock import Mock

from training.workers.env.env_worker import EnvWorker


class _FakeEnv:
    def __init__(self, *, fail: bool = False):
        self.close_calls = 0
        self._fail = fail

    def close(self) -> None:
        self.close_calls += 1
        if self._fail:
            raise RuntimeError("close failed")


def test_env_worker_cleanup_closes_unique_environments_once():
    shared_env = _FakeEnv()
    failing_env = _FakeEnv(fail=True)
    worker = object.__new__(EnvWorker)
    worker.env_list = [shared_env]
    worker.eval_env_list = [shared_env, failing_env]
    worker._envs_closed = False
    worker.log_warning = Mock()
    worker.log_error = Mock()

    worker.cleanup()
    worker.cleanup()

    assert shared_env.close_calls == 1
    assert failing_env.close_calls == 1
    assert worker.env_list == []
    assert worker.eval_env_list == []
    worker.log_warning.assert_called_once()
    worker.log_error.assert_called_once()
