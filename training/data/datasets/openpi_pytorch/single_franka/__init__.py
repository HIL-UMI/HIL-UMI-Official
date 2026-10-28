# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Single-Franka tcp_rot6d SFT data pipeline for openpi_pytorch."""

from __future__ import annotations

import importlib
from typing import Any

_ACBC_EXPORTS = {
    "AcbcDatasetError",
    "AcbcDatasetSummary",
    "PromptFromLeRobotTask",
    "validate_acbc_dataset",
}
_LOADER_EXPORTS = {
    "SingleFrankaSftDataConfig",
    "SingleFrankaSftDataLoader",
    "build_single_franka_sft_dataloader",
    "collate_single_franka_sft_items",
    "create_single_franka_sft_data_loader",
}
_DATASET_EXPORTS = {"SingleFrankaSftDataset"}


def __getattr__(name: str) -> Any:
    """Load single-Franka dependencies only when their exports are requested."""
    if name in _ACBC_EXPORTS:
        module_name = f"{__name__}.acbc"
    elif name in _LOADER_EXPORTS:
        module_name = f"{__name__}.single_franka_sft_data_loader"
    elif name in _DATASET_EXPORTS:
        module_name = f"{__name__}.single_franka_sft_dataset"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = sorted(_ACBC_EXPORTS | _LOADER_EXPORTS | _DATASET_EXPORTS)
