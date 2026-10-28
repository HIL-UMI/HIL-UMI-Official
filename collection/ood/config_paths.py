# Copyright 2026 The HIL-UMI Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Resolve paths shared by the integrated HIL-UMI configuration files."""

from __future__ import annotations

import os
from pathlib import Path


def integration_root() -> Path:
    """Return the monorepo root, with an environment override for deployments."""
    configured = os.environ.get("HIL_UMI_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_integration_path(value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to the monorepo root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (integration_root() / path).resolve()
