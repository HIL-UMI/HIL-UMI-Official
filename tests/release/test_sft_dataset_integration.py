# Copyright 2026 HIL-UMI contributors.
# SPDX-License-Identifier: Apache-2.0

"""Real v2.1 video datasets through the retained CPU SFT loading interface.

The dedicated CI job installs the pinned public LeRobot revision and PyAV
15.1.0. These tests need neither an OpenPI model environment nor robot hardware.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.lerobot_integration

HORIZON = 20
FPS = 10
EPISODE_LENGTHS = (23, 21, 19)
IMAGE_KEYS = ("image", "extra_view_image")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture(params=(False, True), ids=("sft", "acbc"))
def real_dataset(tmp_path, request, monkeypatch):
    # Keep Arrow's real on-disk dataset cache inside the temporary test area,
    # including when another test has already imported datasets.config.
    datasets_config = pytest.importorskip("datasets.config")
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    lerobot = pytest.importorskip(
        "lerobot.common.datasets.lerobot_dataset",
        reason="Run the dedicated sft-dataset job with the pinned LeRobot/video dependencies.",
    )
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    root = tmp_path / "single_franka"
    acbc = request.param
    prompts = (
        ("fold the towel, Advantage: positive", "fold the towel, Advantage: negative")
        if acbc
        else ("fold the towel", "unfold the towel")
    )
    features = {
        "state": {"dtype": "float32", "shape": (10,), "names": None},
        "actions": {"dtype": "float32", "shape": (10,), "names": None},
        **{
            key: {
                "dtype": "video",
                "shape": (64, 64, 3),
                "names": ["height", "width", "channels"],
            }
            for key in IMAGE_KEYS
        },
    }
    writer = lerobot.LeRobotDataset.create(
        repo_id="test/single_franka",
        root=root,
        fps=FPS,
        robot_type="franka_umi_left_tcp_rot6d",
        features=features,
        video_backend="pyav",
    )
    for episode_index, length in enumerate(EPISODE_LENGTHS):
        for frame_index in range(length):
            # The large gap between episodes makes a crossed action chunk
            # distinguishable from any valid continuation within an episode.
            values = np.arange(10, dtype=np.float32) + episode_index * 100 + frame_index
            writer.add_frame(
                {
                    "state": values,
                    "actions": values + np.float32(0.5),
                    "image": np.full((64, 64, 3), 30 + frame_index, dtype=np.uint8),
                    "extra_view_image": np.full(
                        (64, 64, 3), 180 + frame_index, dtype=np.uint8
                    ),
                    "task": prompts[episode_index % 2],
                }
            )
        writer.save_episode()
    return root, prompts, acbc


def test_real_sft_chunks_prompts_and_decoded_views(real_dataset):
    import torch

    from training.data.datasets.openpi_pytorch.single_franka.acbc import (
        PromptFromLeRobotTask,
        validate_acbc_dataset,
    )
    from training.data.datasets.openpi_pytorch.single_franka.single_franka_sft_dataset import (
        SingleFrankaSftDataset,
    )

    root, prompts, acbc = real_dataset
    dataset = SingleFrankaSftDataset(data_path=str(root), action_horizon=HORIZON)
    assert dataset.dataset_root == root
    assert dataset.resolved_fps == FPS
    # Four complete starts in episode 0, two in episode 1, and none in the
    # 19-frame episode. The boundary tails cannot be padded into SFT samples.
    assert len(dataset) == 6
    assert len(dataset.hf_dataset) == sum(EPISODE_LENGTHS)
    expected_starts = [(0, i) for i in range(4)] + [(1, i) for i in range(2)]
    task_prompts = PromptFromLeRobotTask(dataset.meta.tasks)
    seen_prompts = set()
    for index, (episode_index, frame_index) in enumerate(expected_starts):
        sample = dataset[index]
        assert int(sample["episode_index"]) == episode_index
        assert int(sample["frame_index"]) == frame_index
        assert sample["actions"].shape == (HORIZON, 10)
        assert sample["state"].shape == (10,)
        expected_actions = (
            torch.arange(HORIZON, dtype=torch.float32)[:, None]
            + torch.arange(10, dtype=torch.float32)[None, :]
            + episode_index * 100
            + frame_index
            + 0.5
        )
        torch.testing.assert_close(sample["actions"], expected_actions)
        assert not sample["actions_is_pad"].any()
        torch.testing.assert_close(sample["state"], expected_actions[0] - 0.5)
        expected_prompt = prompts[episode_index]
        assert sample["task"] == expected_prompt
        # The actual task_index -> prompt transform must preserve ACBC labels
        # and replace any accidental static prompt on the input sample.
        assert task_prompts({**sample, "prompt": "stale"})["prompt"] == expected_prompt
        seen_prompts.add(expected_prompt)
        for key in IMAGE_KEYS:
            assert sample[key].shape == (3, 64, 64)
            assert sample[key].dtype == torch.float32
            assert torch.isfinite(sample[key]).all()
        # Independent encoded camera streams must remain distinguishable after
        # the real LeRobot/PyAV decoder and its uint8 -> float conversion.
        assert float(sample["extra_view_image"].mean() - sample["image"].mean()) > 0.4
    assert seen_prompts == set(prompts)
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]
    torch.testing.assert_close(
        dataset[-1]["actions"], dataset[len(dataset) - 1]["actions"]
    )

    # No episode can accommodate a 24-frame chunk; this rejects the dataset
    # rather than silently accepting windows that continue into another one.
    with pytest.raises(
        ValueError, match="no samples with a full 24-frame action chunk"
    ):
        SingleFrankaSftDataset(data_path=str(root), action_horizon=24)

    if acbc:
        summary = validate_acbc_dataset(root)
        assert summary.total_episodes == 3
        assert summary.total_frames == sum(EPISODE_LENGTHS)
        assert summary.video_files == 6
        assert set(summary.tasks.values()) == set(prompts)
        counts_by_prompt = {
            summary.tasks[key]: value for key, value in summary.task_counts.items()
        }
        assert counts_by_prompt == {prompts[0]: 42, prompts[1]: 21}


def test_real_sft_artifacts_match_metadata_and_video_frames(real_dataset):
    import av
    import pyarrow.parquet as pq

    root, prompts, _ = real_dataset
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["fps"] == FPS
    assert info["total_episodes"] == 3
    assert info["total_frames"] == sum(EPISODE_LENGTHS)
    assert info["total_tasks"] == 2
    for key in ("state", "actions"):
        assert info["features"][key]["shape"] == [10]
    for key in IMAGE_KEYS:
        assert info["features"][key]["dtype"] == "video"
    task_rows = _read_jsonl(root / "meta" / "tasks.jsonl")
    tasks = {row["task_index"]: row["task"] for row in task_rows}
    assert set(tasks.values()) == set(prompts)
    episode_rows = _read_jsonl(root / "meta" / "episodes.jsonl")
    assert [row["length"] for row in episode_rows] == list(EPISODE_LENGTHS)
    parquet_paths = sorted((root / "data").rglob("episode_*.parquet"))
    assert len(parquet_paths) == 3
    for episode_index, parquet_path in enumerate(parquet_paths):
        table = pq.read_table(parquet_path)
        assert table.num_rows == EPISODE_LENGTHS[episode_index]
        assert set(table["episode_index"].to_pylist()) == {episode_index}
        assert {tasks[value] for value in table["task_index"].to_pylist()} == {
            prompts[episode_index % 2]
        }

    video_paths = sorted((root / "videos").rglob("episode_*.mp4"))
    assert len(video_paths) == 6
    for path in video_paths:
        episode_index = int(path.stem.removeprefix("episode_"))
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            assert (stream.width, stream.height) == (64, 64)
            assert float(stream.average_rate) == FPS
            assert (
                sum(1 for _ in container.decode(video=0))
                == EPISODE_LENGTHS[episode_index]
            )
