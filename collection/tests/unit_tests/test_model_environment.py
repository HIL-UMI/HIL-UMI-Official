# Copyright 2026 The HIL-UMI Authors.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic namespace checks do not import either full model stack."""

from importlib.machinery import ModuleSpec

import ood.model_environment as environment
import pytest
from ood.advantage_model_client import AdvantageModelClient


def _install_specs(monkeypatch, origins):
    requested = []

    def find_spec(name):
        requested.append(name)
        origin = origins.get(name)
        return (
            None
            if origin is None
            else ModuleSpec(name, loader=None, origin=str(origin))
        )

    monkeypatch.setattr(environment.importlib.util, "find_spec", find_spec)
    return requested


def test_policy_openpi_does_not_satisfy_advantage_dependency(monkeypatch):
    requested = _install_specs(monkeypatch, {"openpi": "/policy/openpi/__init__.py"})

    with pytest.raises(RuntimeError, match="hil_umi_advantage.*ADVANTAGE_PYTHON"):
        environment.require_model_namespace("advantage")

    assert requested == ["hil_umi_advantage"]


def test_advantage_namespace_does_not_satisfy_policy_dependency(monkeypatch):
    requested = _install_specs(
        monkeypatch,
        {
            "training": "/policy/training/__init__.py",
            "hil_umi_advantage": "/advantage/hil_umi_advantage/__init__.py",
        },
    )

    with pytest.raises(RuntimeError, match="openpi.*POLICY_PYTHON"):
        environment.require_model_namespace("pi05")

    assert requested == ["training", "openpi"]


def test_policy_namespace_check_does_not_import_advantage(monkeypatch):
    requested = _install_specs(
        monkeypatch,
        {
            "training": "/policy/training/__init__.py",
            "openpi": "/policy/openpi/__init__.py",
        },
    )

    environment.require_model_namespace("pi05")

    assert requested == ["training", "openpi"]


def test_advantage_preflight_rejects_another_checkout(tmp_path, monkeypatch):
    source = tmp_path / "selected" / "src" / "hil_umi_advantage"
    source.mkdir(parents=True)
    _install_specs(
        monkeypatch,
        {"hil_umi_advantage": tmp_path / "other" / "hil_umi_advantage" / "__init__.py"},
    )

    with pytest.raises(RuntimeError, match="another installation"):
        environment.require_model_namespace(
            "advantage", advantage_root=source.parents[1]
        )


def test_advantage_preflight_accepts_selected_editable_checkout(tmp_path, monkeypatch):
    source = tmp_path / "selected" / "src" / "hil_umi_advantage"
    source.mkdir(parents=True)
    _install_specs(monkeypatch, {"hil_umi_advantage": source / "__init__.py"})

    environment.require_model_namespace("advantage", advantage_root=source.parents[1])


def test_advantage_client_checks_namespace_before_weights_or_cuda(monkeypatch):
    _install_specs(monkeypatch, {"openpi": "/policy/openpi/__init__.py"})
    client = AdvantageModelClient(
        {
            "config_name": "ADVANTAGE_TORCH_UMI",
            "ckpt_dir": "/nonexistent/synthetic-checkpoint",
            "device": "cuda:0",
            "prompt": "fold the towel",
        }
    )

    with pytest.raises(RuntimeError, match="hil_umi_advantage.*ADVANTAGE_PYTHON"):
        client.load()
