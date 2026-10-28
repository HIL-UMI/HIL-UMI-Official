#!/usr/bin/env python3
"""Validate an Advantage source checkout."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[PENDING] {message}")


def fail(message: str, failures: list[str]) -> None:
    print(f"[FAIL] {message}")
    failures.append(message)


def check_imports(failures: list[str]) -> None:
    modules = (
        "av",
        "cv2",
        "imageio_ffmpeg",
        "jax",
        "lerobot",
        "matplotlib",
        "numpy",
        "pandas",
        "polars",
        "pyarrow",
        "safetensors",
        "torch",
        "transformers",
        "wandb",
        "hil_umi_advantage",
        "hil_umi_advantage.models_pytorch.gemma_pytorch",
        "hil_umi_advantage.models_pytorch.pi0_pytorch",
        "hil_umi_advantage.training.data_loader",
    )
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            fail(f"import {name}: {exc}", failures)
            continue
        version = getattr(module, "__version__", "unknown")
        ok(f"import {name} ({version})")


def check_checkpoint(path: Path, failures: list[str]) -> None:
    model_path = path / "model.safetensors"
    if not model_path.is_file():
        fail(f"PyTorch pi05_base is missing: {model_path}", failures)
        return
    try:
        from safetensors import safe_open

        with safe_open(model_path, framework="pt", device="cpu") as handle:
            key_count = len(handle.keys())
    except Exception as exc:
        fail(f"cannot read safetensors header at {model_path}: {exc}", failures)
        return
    if key_count == 0:
        fail(f"safetensors contains no tensors: {model_path}", failures)
        return
    ok(f"pi05_base model.safetensors ({key_count} tensors, {model_path.stat().st_size / 2**30:.2f} GiB)")


def check_dataset(path: Path, failures: list[str]) -> None:
    info_path = path / "meta/info.json"
    if not info_path.is_file():
        fail(f"dataset metadata is missing: {info_path}", failures)
        return
    try:
        info = json.loads(info_path.read_text())
        from hil_umi_advantage.training.advantage_schema import detect_advantage_dataset_schema

        schema = detect_advantage_dataset_schema(info.get("features", {}))
    except Exception as exc:
        fail(f"invalid Advantage dataset {path}: {exc}", failures)
        return

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    features = info.get("features", {})
    if total_episodes <= 0 or total_frames <= 0:
        fail(f"dataset has invalid totals: {path}", failures)
        return
    if "stage_progress_gt" not in features:
        fail(f"dataset has no stage_progress_gt feature: {path}", failures)
        return
    ok(f"dataset {path} (schema={schema.name}, episodes={total_episodes}, frames={total_frames})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        choices=("advantage",),
        default="advantage",
        help="Workflow to validate (kept explicit for forward-compatible launch commands)",
    )
    parser.add_argument("--runtime-only", action="store_true", help="Skip GPU, checkpoint, and dataset checks")
    parser.add_argument("--dataset", action="append", type=Path, default=[], help="Dataset to validate; repeatable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    root = Path(os.getenv("ADVANTAGE_ROOT", PROJECT_ROOT)).expanduser().resolve()
    data_root = Path(os.getenv("ADVANTAGE_DATA_ROOT", root / "data")).expanduser().resolve()
    checkpoint_root = Path(os.getenv("ADVANTAGE_PI05_BASE", root / "checkpoints/pi05_base")).expanduser().resolve()

    ok(f"root={root}")
    ok(f"python={sys.executable} ({platform.python_version()})")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        fail(f"unsupported platform: {platform.system()} {platform.machine()}", failures)
    else:
        ok(f"platform={platform.platform()}")
    check_imports(failures)

    if not args.runtime_only:
        try:
            import torch
        except ImportError:
            fail("PyTorch is not installed", failures)
            return 1

        if not torch.cuda.is_available():
            fail("PyTorch cannot access an NVIDIA GPU", failures)
        else:
            devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            ok(f"CUDA devices={devices}; torch CUDA={torch.version.cuda}")
        check_checkpoint(checkpoint_root, failures)

        datasets = args.dataset
        if not datasets:
            candidates = (
                data_root / "fold_towel/base/origin",
                data_root / "fold_towel/stage1/origin",
                data_root / "fold_towel/stage2/origin",
            )
            datasets = [path for path in candidates if (path / "meta/info.json").is_file()]
        if not datasets:
            warn(f"no dataset has been installed under {data_root}")
        for dataset in datasets:
            check_dataset(dataset.expanduser().resolve(), failures)

    if failures:
        print(f"\nDoctor failed with {len(failures)} problem(s).")
        return 1
    print("\nDoctor passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
