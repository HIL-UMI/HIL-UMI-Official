# Copyright 2026 The HIL-UMI Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check the selected model namespace without importing model dependencies."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def require_model_namespace(mode: str, *, advantage_root: Path | None = None) -> None:
    """Fail before hardware/model loading when the wrong environment is selected."""
    if mode == "advantage":
        packages = ("hil_umi_advantage",)
        environment = "ADVANTAGE_PYTHON (.venv-advantage)"
    elif mode == "pi05":
        packages = ("training", "openpi")
        environment = "POLICY_PYTHON (.venv-policy)"
    else:
        raise ValueError(f"Unsupported monitor mode: {mode}")

    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise RuntimeError(
                f"{mode} mode requires the {package!r} package. Select {environment} "
                "and install the corresponding component; OOD_PYTHON, when set, "
                "overrides the mode-specific interpreter."
            )
        if mode == "advantage" and advantage_root is not None:
            expected_source = advantage_root.resolve() / "src" / "hil_umi_advantage"
            if not expected_source.is_dir():
                raise ValueError(
                    "runtime.advantage_root must point to the Advantage source checkout "
                    "containing src/hil_umi_advantage"
                )
            if not Path(spec.origin).resolve().is_relative_to(expected_source):
                raise RuntimeError(
                    "The imported hil_umi_advantage belongs to another installation. "
                    "Install the selected Advantage checkout in ADVANTAGE_PYTHON "
                    "(.venv-advantage)."
                )
