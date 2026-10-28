# Modified for HIL-UMI.
from collections.abc import Callable, Sequence
from pathlib import Path
import random

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from hil_umi_advantage.training.advantage_schema import BASE_UMI_SCHEMA
from hil_umi_advantage.training.advantage_schema import STAGE_UMI_SCHEMA
from hil_umi_advantage.training.advantage_schema import SUPPORTED_ADVANTAGE_SCHEMAS
from hil_umi_advantage.training.advantage_schema import AdvantageDatasetSchema
from hil_umi_advantage.training.advantage_schema import canonicalize_masked_state
from hil_umi_advantage.training.advantage_schema import detect_advantage_dataset_schema

__all__ = [
    "BASE_UMI_SCHEMA",
    "STAGE_UMI_SCHEMA",
    "SUPPORTED_ADVANTAGE_SCHEMAS",
    "AdvantageDatasetSchema",
    "AdvantageLerobotDataset",
    "CanonicalAdvantageDataset",
    "canonicalize_masked_state",
    "detect_advantage_dataset_schema",
]


class AdvantageLerobotDataset(LeRobotDataset):
    """LeRobot dataset for advantage training with a random same-episode comparison frame."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        *,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        selected_video_keys: Sequence[str] | None = None,
    ):
        """Initialize the LeRobot dataset and optionally restrict decoded video features."""
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend,
        )
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(episodes)} if episodes else {}
        self.selected_video_keys = tuple(selected_video_keys or self.meta.video_keys)
        unknown_video_keys = sorted(set(self.selected_video_keys) - set(self.meta.video_keys))
        if unknown_video_keys:
            raise ValueError(f"Unknown selected video keys for {repo_id}: {unknown_video_keys}")

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = super()._get_query_timestamps(current_ts, query_indices)
        return {key: query_timestamps[key] for key in self.selected_video_keys}

    def get_sample_with_imgs_from_idx(self, idx: int) -> dict:
        """Return one sample with only the selected video frames decoded."""
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
            query_indices, _ = self._get_query_indices(idx, arr_idx)

        if self.selected_video_keys:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            for camera_key in self.selected_video_keys:
                item[camera_key] = self.image_transforms(item[camera_key])

        return item

    def __getitem__(self, idx: int) -> dict:
        """Return a current sample, a random same-episode sample, and their progress difference."""
        item = self.get_sample_with_imgs_from_idx(idx)
        ep_idx = item["episode_index"].item()
        current_timestamp = item["timestamp"].item()

        episode_values = {"episode_length": self.meta.episodes[ep_idx]["length"]}
        if self.delta_indices is not None:
            episode_values = self.handle_delta_indices(idx, ep_idx, episode_values)

        task_idx = item["task_index"].item()
        episode_values["task"] = self.meta.tasks[task_idx]

        final_item = self.handle_timestep_difference_mode(idx, ep_idx, ep_idx, current_timestamp)
        final_item = {**final_item, **item, **episode_values}
        comparison_progress = final_item["his_-100_stage_progress_gt"].item()
        current_progress = final_item["stage_progress_gt"].item()
        final_item["progress"] = current_progress - comparison_progress
        return final_item

    def handle_delta_indices(self, idx: int, ep_idx: int, episode_values: dict) -> dict:
        """Fill episode values with the configured action sequence."""
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        query_indices, padding = self._get_query_indices(idx, arr_idx)
        query_result = self._query_hf_dataset(query_indices)
        return {**episode_values, **padding, **query_result}

    def handle_timestep_difference_mode(
        self,
        idx: int,
        ep_idx: int,
        expected_ep_idx: int,
        current_timestamp: float,
    ) -> dict:
        """Add a random, different timestep from the same episode with a his_-100_ prefix."""
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        ep_start_idx = self.episode_data_index["from"][arr_idx].item()
        ep_end_idx = self.episode_data_index["to"][arr_idx].item()
        if ep_end_idx - ep_start_idx < 2:
            raise ValueError(
                f"Advantage training requires at least two frames per episode; episode {ep_idx} is shorter"
            )

        while True:
            random_idx = random.randint(ep_start_idx, ep_end_idx - 1)
            if random_idx == idx:
                continue

            random_item = self.get_sample_with_imgs_from_idx(random_idx)
            random_ep_idx = random_item["episode_index"].item()
            random_timestamp = random_item["timestamp"].item()
            if random_ep_idx == expected_ep_idx and random_timestamp != current_timestamp:
                break

        return {f"his_-100_{key}": value for key, value in random_item.items()}


class CanonicalAdvantageDataset:
    """Adapt a supported raw dataset to the existing two-camera UMI advantage transform."""

    def __init__(self, dataset, schema: AdvantageDatasetSchema, state_dim: int):
        self.dataset = dataset
        self.schema = schema
        self.state_dim = state_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        result = {
            "extra_view_image": item[self.schema.front_key],
            "image": item[self.schema.wrist_key],
            "his_-100_extra_view_image": item[f"his_-100_{self.schema.front_key}"],
            "his_-100_image": item[f"his_-100_{self.schema.wrist_key}"],
            # The towel-fold advantage config masks state. Fixing its size here makes
            # the single-arm base data and dual-arm recorder data batch-compatible.
            "state": canonicalize_masked_state(item[self.schema.state_key], self.state_dim),
            "actions": item[self.schema.action_key],
            "episode_length": item["episode_length"],
            "frame_index": item["frame_index"],
            "episode_index": item["episode_index"],
            "progress": item["progress"],
            "task_index": item["task_index"],
        }
        if "prompt" in item:
            result["prompt"] = item["prompt"]
        return result
