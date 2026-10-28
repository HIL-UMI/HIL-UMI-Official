# Modified for HIL-UMI.
"""Location-independent paths for advantage source checkouts."""

from __future__ import annotations

import os
from pathlib import Path


def _from_env(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def repo_root() -> Path:
    """Return the active advantage root, independent of the current working directory."""
    configured = _from_env("ADVANTAGE_ROOT")
    if configured is not None:
        return configured

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/hil_umi_advantage").is_dir():
            return parent
    # ``src/hil_umi_advantage/shared/paths.py`` is four levels below the repository root.
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    return _from_env("ADVANTAGE_DATA_ROOT") or repo_root() / "data"


def checkpoint_root() -> Path:
    return _from_env("ADVANTAGE_CHECKPOINT_ROOT") or repo_root() / "checkpoints"


def pi05_base() -> Path:
    return _from_env("ADVANTAGE_PI05_BASE") or checkpoint_root() / "pi05_base"


def output_root() -> Path:
    return _from_env("ADVANTAGE_OUTPUT_ROOT") or repo_root() / "outputs"


def cache_root() -> Path:
    return _from_env("ADVANTAGE_CACHE_ROOT") or repo_root() / ".cache"
