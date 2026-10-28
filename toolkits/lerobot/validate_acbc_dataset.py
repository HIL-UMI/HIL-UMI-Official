#!/usr/bin/env python3
# Modified for HIL-UMI.
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate a self-contained advantage-compatible ACBC LeRobot dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.data.datasets.openpi_pytorch.single_franka.acbc import (  # noqa: E402
    AcbcDatasetError,
    validate_acbc_dataset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="LeRobot ACBC dataset inside the HIL-UMI checkout")
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="HIL-UMI checkout that must contain dataset_root",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the requested dataset and print its exact prompt mapping."""
    args = _parse_args()
    try:
        summary = validate_acbc_dataset(
            args.dataset_root,
            repo_root=args.repo_root,
        )
    except AcbcDatasetError as error:
        raise SystemExit(f"ACBC dataset validation failed: {error}") from error

    print(f"ACBC dataset: {summary.dataset_root}")
    print(f"Episodes: {summary.total_episodes}")
    print(f"Frames: {summary.total_frames}")
    print(f"Video files: {summary.video_files}")
    for task_index, prompt in summary.tasks.items():
        count = summary.task_counts.get(task_index, 0)
        print(f"task_index={task_index}: count={count}, prompt={prompt!r}")
    print("ACBC dataset validation: PASS")


if __name__ == "__main__":
    main()
