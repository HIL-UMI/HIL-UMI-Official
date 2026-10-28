# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Run the scene-reference frame selector without robot or camera hardware."""

from __future__ import annotations

import argparse
import threading

from training.envs.realworld.common.scene_alignment import (
    ReferenceSceneAlignmentWebUI,
)
from training.envs.realworld.common.scene_reference_set import (
    SceneReferenceSet,
    default_scene_reference_root,
    resolve_scene_reference_preset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the offline calibration server."""
    parser = argparse.ArgumentParser(
        description=(
            "Select and persist one recorded front-camera frame per position. "
            "This command does not start Ray, ROS, Franka, or RealSense."
        )
    )
    parser.add_argument(
        "--preset", required=True, help="Named preset, e.g. stack_cubes"
    )
    parser.add_argument(
        "--root",
        default=str(default_scene_reference_root()),
        help="Directory containing named scene-reference presets",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Position to show when the page first opens",
    )
    return parser.parse_args()


def main() -> None:
    """Serve the offline frame-selection page until interrupted."""
    args = parse_args()
    preset_dir = resolve_scene_reference_preset(args.preset, root_dir=args.root)
    preset = SceneReferenceSet.load(preset_dir)
    epochs = list(range(preset.num_references))
    if args.start_epoch not in epochs:
        raise ValueError(f"--start-epoch must be in [0, {preset.num_references - 1}].")

    ui = ReferenceSceneAlignmentWebUI(
        str(preset.sources[0].video_dir),
        video_suffix=preset.view_name,
        cache_dir=str(preset.references_dir),
        video_paths=preset.video_paths(),
        source_epoch_indices=preset.source_epoch_indices(),
        calibration_epochs=epochs,
        host=args.host,
        port=args.port,
        default_reference_opacity=1.0,
    )
    try:
        ui.start_epoch(args.start_epoch)
        saved_count = sum(
            preset.reference_image_path(epoch).is_file()
            and preset.reference_metadata_path(epoch).is_file()
            for epoch in epochs
        )
        print(f"Preset: {preset.name} ({saved_count}/{preset.num_references} saved)")
        print(f"Offline calibration UI: {ui.url}")
        print("No robot, ROS, Ray, or RealSense process was started.")
        print("Press Ctrl-C here after saving the desired frames.")
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nOffline calibration UI stopped.")
    finally:
        ui.close()


if __name__ == "__main__":
    main()
