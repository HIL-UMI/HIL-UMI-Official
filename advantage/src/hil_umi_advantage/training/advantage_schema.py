"""Lightweight metadata and state helpers for the Advantage data interfaces."""

from collections.abc import Mapping
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class AdvantageDatasetSchema:
    """Raw LeRobot keys needed to produce the canonical two-camera advantage sample."""

    name: str
    front_key: str
    wrist_key: str
    state_key: str
    action_key: str

    @property
    def video_keys(self) -> tuple[str, str]:
        return self.front_key, self.wrist_key


BASE_UMI_SCHEMA = AdvantageDatasetSchema(
    name="base_umi",
    front_key="extra_view_image",
    wrist_key="image",
    state_key="state",
    action_key="actions",
)
STAGE_UMI_SCHEMA = AdvantageDatasetSchema(
    name="stage_umi",
    front_key="observation.images.cam_front",
    wrist_key="observation.images.cam_left",
    state_key="observation.state",
    action_key="action",
)
SUPPORTED_ADVANTAGE_SCHEMAS = (BASE_UMI_SCHEMA, STAGE_UMI_SCHEMA)


def canonicalize_masked_state(raw_state: Any, model_state_dim: int) -> np.ndarray:
    """Return the fixed-width zero state used by the masked advantage model."""
    if model_state_dim <= 0:
        raise ValueError(f"model_state_dim must be positive, got {model_state_dim}")
    return np.zeros(model_state_dim, dtype=np.float32)


def detect_advantage_dataset_schema(features: Mapping[str, Any]) -> AdvantageDatasetSchema:
    """Detect one of the two fold-towel schemas from LeRobot feature metadata."""
    feature_names = set(features)
    matches = [
        schema
        for schema in SUPPORTED_ADVANTAGE_SCHEMAS
        if {schema.front_key, schema.wrist_key, schema.state_key, schema.action_key}.issubset(feature_names)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous advantage dataset schema; matched {[schema.name for schema in matches]}")

    expected = [
        sorted({schema.front_key, schema.wrist_key, schema.state_key, schema.action_key})
        for schema in SUPPORTED_ADVANTAGE_SCHEMAS
    ]
    raise ValueError(f"Unsupported advantage dataset schema. Expected one of these feature sets: {expected}")


@dataclasses.dataclass(frozen=True)
class EpisodePaths:
    episode_index: int
    parquet: Path
    front_video: Path
    wrist_video: Path
    front_key: str
    wrist_key: str


@dataclasses.dataclass(frozen=True)
class EpisodeData:
    timestamps: np.ndarray
    frame_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    stage_progress_gt: np.ndarray | None

    @property
    def num_frames(self) -> int:
        return len(self.timestamps)
