# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for predicting a longer action chunk than evaluation executes."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from training.envs.action_utils import (
    compute_chunk_steps,
    resolve_action_execution_horizon,
    resolve_sync_action_chunk_final_wait_s,
    slice_action_chunk,
)
from training.workers.env.env_worker import EnvWorker


def test_execution_horizon_defaults_to_prediction_horizon():
    assert resolve_action_execution_horizon({}, prediction_horizon=20) == 20


def test_execution_horizon_configures_ten_of_twenty_actions():
    env_cfg = {"action_execution_horizon": 10}

    execution_horizon = resolve_action_execution_horizon(
        env_cfg,
        prediction_horizon=20,
    )

    assert execution_horizon == 10
    assert compute_chunk_steps(300, execution_horizon) == 30


def test_sync_action_chunk_final_wait_defaults_to_zero_and_accepts_seconds():
    assert resolve_sync_action_chunk_final_wait_s({}) == 0.0
    assert resolve_sync_action_chunk_final_wait_s(
        {"sync_action_chunk_final_wait_s": 0.5}
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("configured", [-0.1, float("nan"), float("inf"), True])
def test_sync_action_chunk_final_wait_rejects_invalid_values(configured):
    with pytest.raises(ValueError, match="sync_action_chunk_final_wait_s"):
        resolve_sync_action_chunk_final_wait_s(
            {"sync_action_chunk_final_wait_s": configured}
        )


@pytest.mark.parametrize("configured", [0, -1, 21, "invalid"])
def test_execution_horizon_rejects_invalid_values(configured):
    with pytest.raises(ValueError, match="action_execution_horizon"):
        resolve_action_execution_horizon(
            {"action_execution_horizon": configured},
            prediction_horizon=20,
        )


def test_slice_action_chunk_keeps_strict_torch_prefix():
    actions = torch.arange(2 * 20 * 10).reshape(2, 20, 10)

    executed = slice_action_chunk(actions, action_execution_horizon=10)

    assert isinstance(executed, torch.Tensor)
    assert executed.shape == (2, 10, 10)
    assert executed.is_contiguous()
    assert torch.equal(executed, actions[:, :10])


def test_slice_action_chunk_keeps_strict_numpy_prefix():
    actions = np.arange(2 * 20 * 10).reshape(2, 20, 10)

    executed = slice_action_chunk(actions, action_execution_horizon=10)

    assert isinstance(executed, np.ndarray)
    assert executed.shape == (2, 10, 10)
    assert executed.flags.c_contiguous
    assert np.array_equal(executed, actions[:, :10])


def test_slice_action_chunk_rejects_short_prediction():
    with pytest.raises(ValueError, match="Cannot execute 10 actions"):
        slice_action_chunk(np.zeros((1, 8, 10)), action_execution_horizon=10)


def test_env_worker_executes_only_first_ten_actions_from_each_prediction():
    class RecordingChunkEnv:
        def __init__(self):
            self.received_chunks = []
            self.received_final_waits = []

        def chunk_step(
            self,
            chunk_actions,
            *,
            sync_action_chunk_final_wait_s=0.0,
        ):
            self.received_chunks.append(chunk_actions.copy())
            self.received_final_waits.append(sync_action_chunk_final_wait_s)
            chunk_size = chunk_actions.shape[1]
            observations = [
                {"states": torch.full((1, 10), float(step))}
                for step in range(chunk_size)
            ]
            rewards = torch.zeros((1, chunk_size))
            terminations = torch.zeros((1, chunk_size), dtype=torch.bool)
            truncations = torch.zeros((1, chunk_size), dtype=torch.bool)
            infos = [{} for _ in range(chunk_size)]
            return observations, rewards, terminations, truncations, infos

    env = RecordingChunkEnv()
    worker = object.__new__(EnvWorker)
    worker.truncate_eval_action_chunk = True
    worker.eval_action_execution_horizon = 10
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "env_type": "realworld",
                    "auto_reset": True,
                }
            }
        }
    )
    worker.model_cfg = OmegaConf.create(
        {
            "model_type": "openpi_pytorch",
            "num_action_chunks": 20,
            "action_dim": 10,
        }
    )
    worker.eval_env_list = [env]
    worker.use_external_reward_model = False

    first_prediction = torch.arange(20 * 10).reshape(1, 20, 10).float()
    second_prediction = first_prediction + 10_000
    env_output, _ = EnvWorker.env_evaluate_step(
        worker,
        first_prediction,
        stage_id=0,
        sync_action_chunk_final_wait_s=0.5,
    )
    EnvWorker.env_evaluate_step(worker, second_prediction, stage_id=0)

    assert len(env.received_chunks) == 2
    assert env.received_chunks[0].shape == (1, 10, 10)
    assert env.received_chunks[1].shape == (1, 10, 10)
    assert np.array_equal(
        env.received_chunks[0],
        first_prediction[:, :10].numpy(),
    )
    assert np.array_equal(
        env.received_chunks[1],
        second_prediction[:, :10].numpy(),
    )
    assert env.received_final_waits == pytest.approx([0.5, 0.0])
    assert env_output.dones.shape == (1, 10)
    assert not env_output.dones.any()
