# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Deterministic fixed-ratio batches from two distributed data streams."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data.distributed import DistributedSampler


def normalize_two_stream_ratios(
    primary_ratio: float,
    secondary_ratio: float,
) -> tuple[float, float]:
    """Normalize two positive finite relative weights."""
    primary_ratio = float(primary_ratio)
    secondary_ratio = float(secondary_ratio)
    if not math.isfinite(primary_ratio) or primary_ratio <= 0:
        raise ValueError(
            "Two-stream primary ratio must be a positive finite number, "
            f"got {primary_ratio!r}."
        )
    if not math.isfinite(secondary_ratio) or secondary_ratio <= 0:
        raise ValueError(
            "Two-stream secondary ratio must be a positive finite number, "
            f"got {secondary_ratio!r}."
        )
    total = primary_ratio + secondary_ratio
    return primary_ratio / total, secondary_ratio / total


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _ratio_primary_count_for_batch(
    batch_index: int,
    batch_size: int,
    primary_fraction: float,
) -> int:
    previous_total = _round_half_up(batch_index * batch_size * primary_fraction)
    next_total = _round_half_up((batch_index + 1) * batch_size * primary_fraction)
    return max(0, min(batch_size, next_total - previous_total))


def _build_primary_batch_counts(
    *,
    primary_len: int,
    batch_size: int,
    primary_fraction: float,
) -> list[int]:
    if primary_len <= 0:
        raise ValueError("Two-stream primary dataset has no local samples.")
    if batch_size <= 0:
        raise ValueError(f"Two-stream batch_size must be positive, got {batch_size}.")

    counts: list[int] = []
    consumed = 0
    batch_index = 0
    expected_batches = max(
        1,
        int(math.ceil(primary_len / max(batch_size * primary_fraction, 1e-12))) + 2,
    )
    max_batches = expected_batches + batch_size + 8
    while consumed < primary_len and batch_index < max_batches:
        primary_count = _ratio_primary_count_for_batch(
            batch_index, batch_size, primary_fraction
        )
        if primary_count > primary_len - consumed:
            break
        counts.append(primary_count)
        consumed += primary_count
        batch_index += 1

    if not counts:
        raise ValueError(
            "Two-stream primary dataset is too small for the requested ratio: "
            f"primary local samples={primary_len}, batch_size={batch_size}, "
            f"primary_fraction={primary_fraction:.6f}."
        )
    return counts


class TwoStreamBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Build fixed-size batches from two distributed samplers."""

    def __init__(
        self,
        *,
        primary_sampler: DistributedSampler,
        secondary_sampler: DistributedSampler,
        secondary_offset: int,
        batch_size: int,
        primary_ratio: float,
        secondary_ratio: float,
    ) -> None:
        if len(secondary_sampler) == 0:
            raise ValueError("Two-stream secondary dataset has no local samples.")

        primary_fraction, secondary_fraction = normalize_two_stream_ratios(
            primary_ratio,
            secondary_ratio,
        )
        primary_batch_counts = _build_primary_batch_counts(
            primary_len=len(primary_sampler),
            batch_size=batch_size,
            primary_fraction=primary_fraction,
        )

        self._primary_sampler = primary_sampler
        self._secondary_sampler = secondary_sampler
        self._secondary_offset = secondary_offset
        self._batch_size = batch_size
        self._primary_batch_counts = primary_batch_counts
        self._primary_fraction = primary_fraction
        self._secondary_fraction = secondary_fraction
        self._primary_samples_per_epoch = sum(primary_batch_counts)
        self._secondary_samples_per_epoch = (
            len(primary_batch_counts) * batch_size - self._primary_samples_per_epoch
        )

    @property
    def primary_samples_per_epoch(self) -> int:
        """Return the number of primary samples emitted per epoch."""
        return self._primary_samples_per_epoch

    @property
    def secondary_samples_per_epoch(self) -> int:
        """Return the number of secondary samples emitted per epoch."""
        return self._secondary_samples_per_epoch

    @property
    def first_batch_split(self) -> tuple[int, int]:
        """Return primary and secondary counts for the first batch."""
        primary_count = self._primary_batch_counts[0]
        return primary_count, self._batch_size - primary_count

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch changes to both distributed samplers."""
        self._primary_sampler.set_epoch(epoch)
        self._secondary_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        primary_indices = list(iter(self._primary_sampler))
        secondary_iter = iter(self._secondary_sampler)
        primary_start = 0

        for primary_count in self._primary_batch_counts:
            primary_batch = primary_indices[
                primary_start : primary_start + primary_count
            ]
            primary_start += primary_count

            secondary_batch: list[int] = []
            for _ in range(self._batch_size - primary_count):
                try:
                    secondary_idx = next(secondary_iter)
                except StopIteration:
                    secondary_iter = iter(self._secondary_sampler)
                    secondary_idx = next(secondary_iter)
                secondary_batch.append(secondary_idx + self._secondary_offset)

            yield primary_batch + secondary_batch

    def __len__(self) -> int:
        return len(self._primary_batch_counts)
