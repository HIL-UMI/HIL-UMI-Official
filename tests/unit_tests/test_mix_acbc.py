# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Mix ACBC configuration and sampling."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

from training.config import validate_sft_cfg
from training.data.datasets.two_stream_batch_sampler import TwoStreamBatchSampler


def _build_sampler(
    *,
    primary_len: int,
    secondary_len: int,
    batch_size: int,
    primary_ratio: float,
    secondary_ratio: float,
) -> TwoStreamBatchSampler:
    primary = TensorDataset(torch.arange(primary_len))
    secondary = TensorDataset(torch.arange(secondary_len))
    primary_sampler = DistributedSampler(
        primary,
        num_replicas=1,
        rank=0,
        shuffle=False,
        drop_last=True,
    )
    secondary_sampler = DistributedSampler(
        secondary,
        num_replicas=1,
        rank=0,
        shuffle=False,
        drop_last=False,
    )
    return TwoStreamBatchSampler(
        primary_sampler=primary_sampler,
        secondary_sampler=secondary_sampler,
        secondary_offset=primary_len,
        batch_size=batch_size,
        primary_ratio=primary_ratio,
        secondary_ratio=secondary_ratio,
    )


def _batch_split(batch: list[int], secondary_offset: int) -> tuple[int, int]:
    primary = sum(index < secondary_offset for index in batch)
    return primary, len(batch) - primary


def _minimal_sft_cfg(**data_overrides):
    data = {
        "train_data_paths": "/data/group-a",
        "val_data_paths": None,
        "prompt_from_task": True,
        "acbc": True,
        "mix_acbc": True,
        "mix_acbc_data_path": "/data/group-b",
        **data_overrides,
    }
    return OmegaConf.create(
        {
            "actor": {
                "global_batch_size": 8,
                "micro_batch_size": 4,
                "model": {},
            },
            "runner": {},
            "data": data,
        }
    )


def test_mix_acbc_sampler_is_exact_half_per_even_batch_and_cycles_secondary():
    sampler = _build_sampler(
        primary_len=8,
        secondary_len=2,
        batch_size=4,
        primary_ratio=0.5,
        secondary_ratio=0.5,
    )

    batches = list(sampler)

    assert batches
    assert all(_batch_split(batch, 8) == (2, 2) for batch in batches)
    assert sampler.primary_samples_per_epoch == 8
    assert sampler.secondary_samples_per_epoch == 8
    assert sum(index >= 8 for batch in batches for index in batch) == 8


def test_mix_acbc_sampler_alternates_odd_batch_rounding():
    sampler = _build_sampler(
        primary_len=8,
        secondary_len=8,
        batch_size=3,
        primary_ratio=0.5,
        secondary_ratio=0.5,
    )

    splits = [_batch_split(batch, 8) for batch in sampler]

    assert splits[:4] == [(2, 1), (1, 2), (2, 1), (1, 2)]


def test_mix_acbc_sampler_honors_non_equal_relative_weights():
    sampler = _build_sampler(
        primary_len=14,
        secondary_len=20,
        batch_size=10,
        primary_ratio=0.7,
        secondary_ratio=0.3,
    )

    assert all(_batch_split(batch, 14) == (7, 3) for batch in sampler)


def test_mix_acbc_config_adds_default_ratios():
    cfg = validate_sft_cfg(_minimal_sft_cfg())

    assert cfg.data.mix_acbc_primary_ratio == 0.5
    assert cfg.data.mix_acbc_secondary_ratio == 0.5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"acbc": False}, "requires data.acbc=true"),
        ({"prompt_from_task": False}, "requires data.prompt_from_task=true"),
        ({"mix_acbc_data_path": None}, "requires data.mix_acbc_data_path"),
        ({"mix_acbc_primary_ratio": 0}, "primary_ratio"),
        ({"mix_acbc_secondary_ratio": float("inf")}, "secondary_ratio"),
        ({"post_train": True, "post_train_data_path": "/data/c"}, "cannot be combined"),
    ],
)
def test_mix_acbc_config_rejects_invalid_contract(overrides, message):
    with pytest.raises(AssertionError, match=message):
        validate_sft_cfg(_minimal_sft_cfg(**overrides))
