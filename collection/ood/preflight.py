# Copyright 2026 The HIL-UMI Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate installed monitor dependencies before starting external hardware."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import yaml

from ood.config_paths import resolve_integration_path
from ood.model_environment import require_model_namespace


def validate_recorder_config(config: dict[str, Any], monitor: dict[str, Any]) -> None:
    """Reject unconfigured rigs and mismatched shared-memory dimensions."""
    rig_id = str(
        config.get("collection_metadata", {}).get("collection_device_id") or ""
    ).strip()
    if not rig_id:
        raise ValueError(
            "Set collection_metadata.collection_device_id in your private recorder YAML"
        )
    enabled = {
        camera["name"]: camera
        for camera in config.get("cameras", [])
        if camera.get("enable", True)
    }
    if not enabled or any(
        not str(camera.get("serial") or "").strip() for camera in enabled.values()
    ):
        raise ValueError(
            "Set serial for every enabled recorder camera in your private YAML"
        )
    source = monitor.get("source", {})
    frequency = monitor.get("ood", {}).get("action_frequency_hz")
    if frequency is not None and float(frequency) != float(config.get("fps", 30)):
        raise ValueError("Monitor action_frequency_hz must match recorder fps")
    for key, camera in source.get("cameras", {}).items():
        recorder_camera = enabled.get(camera["name"])
        if recorder_camera is None:
            raise ValueError(
                f"Monitor camera {key!r} is not enabled in recorder config"
            )
        for field in ("width", "height", "ring_slots"):
            if int(camera[field]) != int(recorder_camera[field]):
                raise ValueError(f"Monitor/recorder camera {key!r} disagree on {field}")
        if int(monitor.get("detection", {}).get("frame_offset", 0)) >= int(
            camera["ring_slots"]
        ):
            raise ValueError("Advantage frame_offset must be smaller than ring_slots")
    prompt = monitor.get("model", {}).get("prompt") or source.get("task_description")
    if str(prompt) != str(config.get("task_description")):
        raise ValueError("Monitor prompt and recorder task_description must match")


def main() -> None:
    """Check local files and imports without opening devices or loading weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pi05", "advantage"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--recorder-config", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    try:
        with open(args.config, encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}
        with open(args.recorder_config, encoding="utf-8") as file:
            recorder = yaml.safe_load(file) or {}
        validate_recorder_config(recorder, cfg)
        module_name = (
            "ood.advantage_monitor" if args.mode == "advantage" else "ood.monitor"
        )
        module = importlib.import_module(module_name)
        if (
            Path(module.__file__).resolve().parent
            != Path(args.source_root).resolve() / "ood"
        ):
            raise RuntimeError(
                "Install this checkout with pip install -e collection in the monitor environment"
            )
        advantage_root = (
            resolve_integration_path(cfg["runtime"]["advantage_root"])
            if args.mode == "advantage"
            else None
        )
        require_model_namespace(args.mode, advantage_root=advantage_root)
        # This import is deliberately lazy in the reader, but checked here before
        # recorder launch so a missing private dependency cannot fail silently.
        from ood.image_reader import recorder_ring_factory

        recorder_ring_factory()
        importlib.import_module("aiohttp")
        import torch

        device = torch.device(str(cfg["model"]["device"]))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Monitor requires {device}; CUDA is unavailable")
        if args.mode == "advantage":
            from hil_umi_advantage.models_pytorch.pi0_pytorch import AdvantageEstimator
            from hil_umi_advantage.training.config import get_config

            from ood.advantage_model_client import resolve_advantage_checkpoint

            if not get_config(str(cfg["model"]["config_name"])).advantage_estimator:
                raise ValueError("Select an Advantage estimator configuration")
            if not callable(getattr(AdvantageEstimator, "sample_values", None)):
                raise RuntimeError(
                    "Installed hil_umi_advantage lacks Advantage inference"
                )
            resolve_advantage_checkpoint(cfg["model"]["ckpt_dir"])
        else:
            from training.models.embodiment.openpi_pytorch.eval_action_model import (
                OpenPiPytorchEvalActionModel,
            )

            if not callable(
                getattr(
                    OpenPiPytorchEvalActionModel, "predict_action_batch_samples", None
                )
            ):
                raise RuntimeError(
                    "Installed training package lacks the shared-prefix sampler"
                )
            weights = (
                resolve_integration_path(cfg["model"]["model_path"])
                / "model.safetensors"
            )
            if not weights.is_file():
                raise FileNotFoundError(f"Model weights not found: {weights}")
            stats = resolve_integration_path(
                cfg["model"]["openpi_data"]["norm_stats_path"]
            )
            if not stats.exists():
                raise FileNotFoundError(f"Normalization statistics not found: {stats}")
    except Exception as exc:
        raise SystemExit(f"OOD preflight failed: {exc}") from exc


if __name__ == "__main__":
    main()
