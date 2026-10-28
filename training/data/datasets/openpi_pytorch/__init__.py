# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable

SftDataLoaderBuilder = Callable[..., tuple[Any, Any]]


def _load_single_franka_sft_dataloader() -> SftDataLoaderBuilder:
    from training.data.datasets.openpi_pytorch.single_franka import (
        build_single_franka_sft_dataloader,
    )

    return build_single_franka_sft_dataloader


# Environment name -> lazy SFT dataloader builder.
_SFT_DATALOADER_BUILDERS = {
    "singlefranka": _load_single_franka_sft_dataloader,
}


def _resolve_env(config_name: str) -> str:
    """Resolve the registered environment named by ``config_name``."""
    for env_type in _SFT_DATALOADER_BUILDERS:
        if env_type in config_name:
            return env_type
    raise ValueError(
        f"No openpi_pytorch SFT dataloader registered matching "
        f"config_name={config_name!r}; known envs: {list(_SFT_DATALOADER_BUILDERS)}."
    )


def build_openpi_pytorch_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the environment-specific openpi_pytorch SFT dataloader."""
    env_type = _resolve_env(str(cfg.actor.model.openpi.config_name))
    builder = _SFT_DATALOADER_BUILDERS[env_type]()
    return builder(cfg, world_size, rank, data_paths, eval_dataset)


__all__ = ["build_openpi_pytorch_sft_dataloader"]
