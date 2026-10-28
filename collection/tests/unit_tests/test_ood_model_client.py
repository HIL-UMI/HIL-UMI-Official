# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
from ood.model_client import Pi05ModelClient


class _FakeModel:
    def __init__(self, output_shape: tuple[int, ...]) -> None:
        self.output_shape = output_shape
        self.observation = None
        self.mode = None
        self.rng = None
        self.num_samples = None
        self.profile = None
        self.use_cuda_graph = None
        self.profiles: list[bool] = []

    def predict_action_batch_samples(
        self, observation, *, num_samples, rng, profile, use_cuda_graph
    ):
        self.observation = observation
        self.rng = rng
        self.num_samples = num_samples
        self.profile = profile
        self.use_cuda_graph = use_cuda_graph
        self.profiles.append(profile)
        return torch.zeros(self.output_shape, dtype=torch.float32), {
            "timings_ms": {"prefix_ms": 12.5}
        }

    def predict_action_batch(self, *args, **kwargs):
        raise AssertionError("legacy repeated-observation path must not be used")


def _client(fake_model: _FakeModel) -> Pi05ModelClient:
    client = Pi05ModelClient({"num_action_chunks": 20, "action_dim": 10})
    client._model = fake_model
    client._device = torch.device("cpu")
    return client


def test_predict_action_batch_k_uses_shared_sampler_without_repeating_observation():
    fake_model = _FakeModel((1, 4, 20, 10))
    client = _client(fake_model)
    rng = torch.Generator().manual_seed(7)
    observation = {
        "states": np.arange(10, dtype=np.float32)[None],
        "main_images": np.ones((1, 2, 3, 3), dtype=np.uint8),
        "extra_view_images": torch.ones((1, 2, 3, 3), dtype=torch.uint8),
        "task_descriptions": ["stack the cubes"],
    }

    actions = client.predict_action_batch_k(observation, k=4, rng=rng)

    assert actions.shape == (4, 20, 10)
    assert fake_model.rng is rng
    assert fake_model.num_samples == 4
    assert fake_model.profile is True
    assert fake_model.use_cuda_graph is True
    assert fake_model.observation["states"].shape == (1, 10)
    assert fake_model.observation["main_images"].shape == (1, 2, 3, 3)
    assert fake_model.observation["extra_view_images"].shape == (1, 2, 3, 3)
    assert fake_model.observation["task_descriptions"] == ["stack the cubes"]
    assert client.last_inference_timings_ms["prefix_ms"] == pytest.approx(12.5)
    assert client.last_inference_timings_ms["client_total_ms"] >= 0.0


def test_predict_action_batch_k_rejects_wrong_model_shape():
    client = _client(_FakeModel((1, 2, 32, 10)))

    with pytest.raises(RuntimeError, match=r"expected \(1, 2, 20, 10\)"):
        client.predict_action_batch_k(
            {
                "states": np.zeros((1, 10), dtype=np.float32),
                "task_descriptions": ["stack the cubes"],
            },
            k=2,
        )


def test_predict_action_batch_k_rejects_non_single_input_batch():
    client = _client(_FakeModel((1, 2, 20, 10)))

    with pytest.raises(ValueError, match="batch size 1"):
        client.predict_action_batch_k(
            {"states": np.zeros((2, 10), dtype=np.float32)},
            k=2,
        )


def test_warmup_uses_unprofiled_calls_and_clears_diagnostic_timings():
    fake_model = _FakeModel((1, 4, 20, 10))
    client = _client(fake_model)
    observation = {
        "states": np.zeros((1, 10), dtype=np.float32),
        "task_descriptions": ["stack the cubes"],
    }

    elapsed = client.warmup(observation, k=4, runs=2, seed=123)

    assert len(elapsed) == 2
    assert all(value >= 0.0 for value in elapsed)
    assert fake_model.profiles == [False, False]
    assert client.last_inference_timings_ms == {}
