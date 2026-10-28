# Copyright 2026 The HIL-UMI Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from ood.config_paths import integration_root, resolve_integration_path


def test_default_integration_root_is_parent_of_collection(monkeypatch):
    monkeypatch.delenv("HIL_UMI_ROOT", raising=False)
    assert integration_root() == Path(__file__).resolve().parents[3]


def test_relative_path_uses_environment_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HIL_UMI_ROOT", str(tmp_path))
    assert resolve_integration_path("checkpoints/model") == (
        tmp_path / "checkpoints" / "model"
    ).resolve()


def test_absolute_path_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("HIL_UMI_ROOT", "/should/not/be/used")
    assert resolve_integration_path(tmp_path) == tmp_path.resolve()
