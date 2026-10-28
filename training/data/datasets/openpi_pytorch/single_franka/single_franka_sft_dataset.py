# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Standard LeRobot dataset wrapper for single-Franka tcp_rot6d SFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from training.data.lerobot_paths import resolve_lerobot_dataset_root
from training.utils.logging import get_logger

logger = get_logger()


def _as_int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _full_chunk_start_indices(
    episode_data_index: dict[str, Any],
    action_horizon: int,
) -> list[int]:
    """Return frame indices whose future action window fits in one episode."""
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")

    starts = episode_data_index["from"]
    ends = episode_data_index["to"]
    if len(starts) != len(ends):
        raise ValueError(
            "LeRobot episode_data_index has mismatched 'from'/'to' lengths: "
            f"{len(starts)} vs {len(ends)}."
        )

    valid_indices: list[int] = []
    for start_value, end_value in zip(starts, ends):
        start = _as_int(start_value)
        end = _as_int(end_value)
        # LeRobot's ``to`` index is exclusive.  A horizon-H chunk starting at
        # ``i`` needs ``i + H - 1 < end``, so valid starts are [start, end-H].
        valid_stop = end - action_horizon + 1
        if valid_stop > start:
            valid_indices.extend(range(start, valid_stop))
    return valid_indices


class SingleFrankaSftDataset(LeRobotDataset):
    """LeRobot single-Franka frames with an action chunk of horizon ``H``.

    Feature keys match the UMI-derived Franka export::

        image, extra_view_image, state, actions[, task]

    ``state`` and ``actions`` are both 10-dim
    ``[xyz, rot6d, gripper]`` tensors.
    """

    def __init__(
        self,
        *,
        data_path: str,
        action_horizon: int,
        fps: int | None = None,
        tolerance_s: float = 1e-4,
    ) -> None:
        root = resolve_lerobot_dataset_root(data_path)
        repo_id = root.name
        meta_fps = None
        info_path = root / "meta" / "info.json"
        if info_path.is_file():
            import json

            with info_path.open("r", encoding="utf-8") as f:
                meta_fps = int(json.load(f).get("fps", 10))
        resolved_fps = int(fps) if fps is not None else (meta_fps or 10)

        delta_timestamps = {
            "actions": [t / float(resolved_fps) for t in range(action_horizon)]
        }
        logger.info(
            "SingleFrankaSftDataset root=%s repo_id=%s fps=%d horizon=%d",
            root,
            repo_id,
            resolved_fps,
            action_horizon,
        )
        super().__init__(
            repo_id=repo_id,
            root=str(root),
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
        )
        self._dataset_root = Path(root)
        self._resolved_fps = resolved_fps
        self._action_horizon = int(action_horizon)
        self._valid_start_indices = _full_chunk_start_indices(
            self.episode_data_index,
            self._action_horizon,
        )
        raw_len = super().__len__()
        skipped = raw_len - len(self._valid_start_indices)
        if not self._valid_start_indices:
            raise ValueError(
                f"SingleFrankaSftDataset {root} has no samples with a full "
                f"{self._action_horizon}-frame action chunk."
            )
        if skipped > 0:
            logger.info(
                "SingleFrankaSftDataset skipped %d/%d frame starts whose "
                "%d-frame action chunk would cross an episode boundary.",
                skipped,
                raw_len,
                self._action_horizon,
            )

    def __len__(self) -> int:
        """Return only starts with a complete action chunk."""
        valid_start_indices = getattr(self, "_valid_start_indices", None)
        if valid_start_indices is None:
            return super().__len__()
        return len(valid_start_indices)

    def __getitem__(self, idx: int) -> Any:
        """Map filtered sample indices back to raw LeRobot frame indices."""
        valid_start_indices = getattr(self, "_valid_start_indices", None)
        if valid_start_indices is None:
            return super().__getitem__(idx)
        idx = _as_int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"SingleFrankaSftDataset index {idx} out of range for "
                f"{len(self)} full-chunk samples."
            )
        return super().__getitem__(valid_start_indices[idx])

    @property
    def dataset_root(self) -> Path:
        """Return the resolved local LeRobot dataset root."""
        return self._dataset_root

    @property
    def resolved_fps(self) -> int:
        """Return the frame rate used to build action timestamps."""
        return self._resolved_fps
