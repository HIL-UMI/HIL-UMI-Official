# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.


import hashlib
from typing import Any

import torch


def safe_hash(input_tuple: tuple[Any, ...]) -> int:
    """Create a deterministic hash for seeding RNG."""
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)
    seed = int(sha256.hexdigest(), 16)
    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


def forward_set_epoch(data_loader: Any, epoch: int) -> None:
    """Forward ``set_epoch`` to the wrapped sampler and dataset when present."""
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)


def batch_pad_to_fixed_len(
    batch: list[torch.Tensor],
    max_batch_len: int,
    pad_token: int,
    left_pad: bool = False,
) -> torch.Tensor:
    if left_pad:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),  # pad on the left
                        seq,
                    ]
                )
                for seq in batch
            ]
        )
    else:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        seq,
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),
                    ]
                )
                for seq in batch
            ]
        )
    return batch_pad
