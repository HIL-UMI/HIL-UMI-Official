# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.models.embodiment.openpi_pytorch.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from training.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from training.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


def _observation(batch_size: int = 2) -> Observation:
    images = {
        key: torch.zeros(batch_size, 224, 224, 3, dtype=torch.float32)
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    masks = {key: torch.ones(batch_size, dtype=torch.bool) for key in images}
    return Observation(
        images=images,
        image_masks=masks,
        state=torch.zeros(batch_size, 10),
        tokenized_prompt=torch.ones(batch_size, 8, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch_size, 8, dtype=torch.bool),
    )


class _FakeSamplingPi0(Pi0):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.action_horizon = 2
        self.action_dim = 3
        self.prefix_batches: list[int] = []
        self.prefix_image_keys: list[tuple[str, ...] | None] = []
        self.suffix_batches: list[int] = []
        self.suffix_image_batches: list[int] = []
        self.cache_batch_rows: list[list[float]] = []

    def build_prefix_cache(self, observation, *, image_keys=None):
        batch_size = observation.state.shape[0]
        self.prefix_batches.append(batch_size)
        self.prefix_image_keys.append(image_keys)
        mask = torch.ones(batch_size, 1, dtype=torch.bool)
        rows = torch.arange(batch_size, dtype=torch.float32).reshape(
            batch_size, 1, 1, 1
        )
        return torch.zeros(batch_size, 1, 1), mask, ((rows, rows.clone()),)

    def run_suffix(self, observation, x_t, t_tensor, kv_cache, prefix_mask):
        del t_tensor, prefix_mask
        self.suffix_batches.append(observation.state.shape[0])
        self.suffix_image_batches.append(observation.images["base_0_rgb"].shape[0])
        self.cache_batch_rows.append(kv_cache[0][0][:, 0, 0, 0].tolist())
        return torch.zeros_like(x_t)

    def velocity_from_suffix(self, suffix_out_act):
        return suffix_out_act


def test_core_samples_k_noises_from_one_prefix_batch():
    core = _FakeSamplingPi0()
    observation = _observation(batch_size=2)
    noise = torch.arange(2 * 3 * 2 * 3, dtype=torch.float32).reshape(2, 3, 2, 3)

    actions, timings = core.sample_action_samples(
        observation,
        num_samples=3,
        num_steps=2,
        noise=noise,
    )

    assert timings == {}
    assert core.prefix_batches == [2]
    assert core.prefix_image_keys == [None]
    assert core.suffix_batches == [6, 6]
    assert core.suffix_image_batches == [2, 2]
    assert core.cache_batch_rows == [
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ]
    torch.testing.assert_close(actions, noise)


def test_core_rejects_wrong_explicit_noise_shape():
    core = _FakeSamplingPi0()
    with pytest.raises(ValueError, match="noise must have shape"):
        core.sample_action_samples(
            _observation(batch_size=1),
            num_samples=4,
            noise=torch.zeros(4, 2, 3),
        )


class _FakeVision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, images):
        self.batch_sizes.append(images.shape[0])
        return torch.zeros(images.shape[0], 4, 5), None


class _FakeLlm(torch.nn.Module):
    def embed(self, prompt):
        return torch.zeros(prompt.shape[0], prompt.shape[1], 5)


class _FakePrefixPi0(Pi0):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.img = _FakeVision()
        self.llm = _FakeLlm()
        self.pcd = False


def test_prefix_batches_only_selected_real_camera_views():
    core = _FakePrefixPi0()
    observation = _observation(batch_size=1)
    observation.image_masks["right_wrist_0_rgb"] = torch.zeros(1, dtype=torch.bool)

    tokens, mask, _ = core.embed_prefix(
        observation,
        image_keys=("base_0_rgb", "left_wrist_0_rgb"),
    )

    assert core.img.batch_sizes == [2]
    assert tokens.shape == (1, 16, 5)
    assert mask.shape == (1, 16)
    assert torch.all(mask)


class _FakeCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.observation = None
        self.num_samples = None

    def sample_action_samples(
        self,
        observation,
        *,
        num_samples,
        num_steps,
        noise,
        rng,
        profile,
        image_keys,
        use_cuda_graph,
    ):
        del num_steps, rng
        self.observation = observation
        self.num_samples = num_samples
        self.image_keys = image_keys
        self.use_cuda_graph = use_cuda_graph
        assert profile
        return noise, {"prefix_ms": 2.0, "denoise_ms": 3.0}


def test_eval_wrapper_keeps_input_batch_one_and_returns_b_k_actions():
    core = _FakeCore()
    wrapper = OpenPiPytorchEvalActionModel(
        core,
        num_steps=10,
        action_env_dim=3,
        action_chunk=2,
        config_name="test",
    )
    processed = {
        "image": {
            key: torch.full((1, 2, 2, 3), 255, dtype=torch.uint8)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "image_mask": {
            "base_0_rgb": torch.ones(1, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool),
        },
        "state": torch.zeros(1, 10, dtype=torch.float64),
        "tokenized_prompt": torch.ones(1, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    wrapper._repack_env_obs = lambda env_obs: env_obs
    wrapper.input_transform = lambda obs, transpose=False: processed
    wrapper.output_transform = lambda outputs: outputs
    noise = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)

    actions, result = wrapper.predict_action_batch_samples(
        {"unused": np.zeros(1)},
        num_samples=4,
        noise=noise,
        profile=True,
    )

    assert core.num_samples == 4
    assert core.image_keys == ("base_0_rgb", "left_wrist_0_rgb")
    assert core.use_cuda_graph is False
    assert core.observation.state.shape == (1, 10)
    assert core.observation.state.dtype == torch.float32
    assert core.observation.images["base_0_rgb"].dtype == torch.float32
    assert torch.all(core.observation.images["base_0_rgb"] == 1.0)
    assert actions.shape == (1, 4, 2, 3)
    torch.testing.assert_close(actions, noise)
    assert result["timings_ms"]["prefix_ms"] == pytest.approx(2.0)
    assert result["timings_ms"]["denoise_ms"] == pytest.approx(3.0)
    assert result["active_image_keys"] == (
        "base_0_rgb",
        "left_wrist_0_rgb",
    )
