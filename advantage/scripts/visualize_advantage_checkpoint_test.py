import json
from pathlib import Path

import numpy as np
import pytest

from scripts import visualize_advantage_checkpoint as visualize


def _create_episode_files(dataset_dir: Path, episode_index: int) -> None:
    for relative_path in (
        f"data/chunk-000/episode_{episode_index:06d}.parquet",
        f"videos/chunk-000/extra_view_image/episode_{episode_index:06d}.mp4",
        f"videos/chunk-000/image/episode_{episode_index:06d}.mp4",
    ):
        path = dataset_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_resolve_checkpoint_accepts_step_and_selects_latest(tmp_path: Path):
    step_5 = tmp_path / "5"
    step_10 = tmp_path / "10"
    step_5.mkdir()
    step_10.mkdir()
    (step_5 / "model.safetensors").touch()
    (step_10 / "model.safetensors").touch()

    assert visualize.resolve_checkpoint(tmp_path) == step_10
    assert visualize.resolve_checkpoint(step_5) == step_5


def test_select_episode_is_deterministic_and_uses_complete_episodes(tmp_path: Path):
    info = {
        "total_episodes": 51,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    for episode_index in range(51):
        _create_episode_files(tmp_path, episode_index)

    selected, _ = visualize.select_episode(tmp_path, seed=42, episode_index=None)

    assert selected.episode_index == 40


def test_select_episode_rejects_incomplete_explicit_episode(tmp_path: Path):
    info = {
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps(info))

    with pytest.raises(FileNotFoundError):
        visualize.select_episode(tmp_path, seed=42, episode_index=0)


def test_relative_ground_truth_and_action_chunk():
    progress = np.asarray([0.25, 0.5, 0.1], dtype=np.float32)
    actions = np.arange(12, dtype=np.float32).reshape(4, 3)

    np.testing.assert_allclose(visualize.compute_relative_ground_truth(progress), [0.0, 0.25, -0.15])
    np.testing.assert_array_equal(
        visualize.make_action_chunk(actions, frame_index=2, action_horizon=4),
        actions[[2, 3, 3, 3]],
    )


def test_overlay_values_returns_side_by_side_bgr_frame():
    front = np.zeros((48, 64, 3), dtype=np.uint8)
    wrist = np.full((48, 64, 3), 127, dtype=np.uint8)

    result = visualize.overlay_values(
        front,
        wrist,
        frame_index=12,
        timestamp=0.4,
        ground_truth=0.25,
        prediction=0.2,
    )

    assert result.shape == (48, 128, 3)
    assert result.dtype == np.uint8
    assert np.any(result != 0)
