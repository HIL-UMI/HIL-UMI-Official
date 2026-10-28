# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Resolve LeRobot dataset paths without relying on a global HF_LEROBOT_HOME override."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HOME
from omegaconf import DictConfig, ListConfig

_DEFAULT_LEROBOT_HOME = Path(HF_HOME) / "lerobot"


def default_hf_lerobot_home() -> Path:
    return Path(os.getenv("HF_LEROBOT_HOME", _DEFAULT_LEROBOT_HOME)).expanduser()


def _mapping_get(mapping: Any, key: str, default: Any = None) -> Any:
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return default


def resolve_lerobot_repo_id(data_paths: Any) -> str | None:
    """Extract a LeRobot repo id or local dataset path from ``data.train_data_paths``."""
    if data_paths is None:
        return None
    if isinstance(data_paths, (str, Path)):
        return str(data_paths)
    if isinstance(data_paths, (dict, DictConfig)):
        path = _mapping_get(data_paths, "dataset_path") or _mapping_get(
            data_paths, "data_path"
        )
        return str(path) if path is not None else None
    if isinstance(data_paths, (list, tuple, ListConfig)):
        if len(data_paths) == 0:
            return None
        return resolve_lerobot_repo_id(data_paths[0])
    return str(data_paths)


def resolve_lerobot_dataset_root(data_path: str) -> Path:
    """Resolve the on-disk LeRobot dataset root for a path or Hugging Face repo id."""
    path = Path(data_path).expanduser()
    if (path / "meta" / "info.json").is_file():
        return path.resolve()

    cached = default_hf_lerobot_home() / data_path
    if (cached / "meta" / "info.json").is_file():
        return cached.resolve()

    return path.resolve()
