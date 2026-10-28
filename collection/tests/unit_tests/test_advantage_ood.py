# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import ood.replay as replay_module
import pytest
import yaml
from ood.advantage_model_client import (
    AdvantageModelClient,
    resolve_advantage_checkpoint,
)
from ood.advantage_monitor import (
    AdvantageOODMonitor,
    FrameAnchor,
    is_advantage_ood,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(*, ring_slots: int = 60, frame_offset: int = 50) -> dict:
    return {
        "model": {
            "config_name": "ADVANTAGE_TORCH_TOWEL_FOLD",
            "ckpt_dir": "/tmp/not-loaded-in-this-test",
            "device": "cuda:0",
            "prompt": "fold the towel",
            "warmup_runs": 0,
        },
        "detection": {
            "threshold": 0.0,
            "frame_offset": frame_offset,
            "interval_s": 0.5,
        },
        "source": {
            "socket_path": "/tmp/test-advantage.sock",
            "camera_frequency_hz": 30,
            "cameras": {
                "front": {
                    "name": "front",
                    "ring_name": "front-ring",
                    "width": 4,
                    "height": 3,
                    "ring_slots": ring_slots,
                },
                "wrist_left": {
                    "name": "wrist",
                    "ring_name": "wrist-ring",
                    "width": 4,
                    "height": 3,
                    "ring_slots": ring_slots,
                },
            },
            "model_inputs": {"front": "front", "wrist": "wrist_left"},
        },
        "replay": {"output_root": "/tmp"},
        "server": {"host": "127.0.0.1", "port": 18765},
    }


def test_advantage_decision_is_strictly_below_threshold():
    assert is_advantage_ood(-0.0001, 0.0)
    assert not is_advantage_ood(0.0, 0.0)
    assert not is_advantage_ood(0.1, 0.0)


def test_ring_must_retain_more_frames_than_the_requested_offset():
    with pytest.raises(ValueError, match="ring_slots must be greater"):
        AdvantageOODMonitor(_config(ring_slots=50, frame_offset=50))


def test_unique_pair_tick_requires_both_cameras_to_advance():
    previous = FrameAnchor(1.0, 10, {"front": 100, "wrist_left": 200}, None, None)
    half_updated = FrameAnchor(1.1, 11, {"front": 101, "wrist_left": 200}, None, None)
    complete = FrameAnchor(1.2, 12, {"front": 101, "wrist_left": 201}, None, None)

    assert not AdvantageOODMonitor._paired_anchor_advanced(
        previous, half_updated, ("front", "wrist_left")
    )
    assert AdvantageOODMonitor._paired_anchor_advanced(
        previous, complete, ("front", "wrist_left")
    )


def test_episode_lifecycle_preserves_continuous_advantage_history(tmp_path):
    monitor = AdvantageOODMonitor(_config())
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    anchor = FrameAnchor(
        1.0,
        -1,
        {"front": 100, "wrist_left": 200},
        None,
        None,
        {"front": image, "wrist_left": image},
    )
    monitor._history.append(anchor)
    monitor._latest_paired_snapshot = Mock()
    monitor._replay = Mock()
    run_root = (tmp_path / "run_20260822_030405Z").resolve()
    monitor._camera_run_root = str(run_root)

    monitor._handle_lifecycle(
        {
            "state": "recording",
            "run_root": str(run_root),
            "episode": 1,
            "timestamp": 2.0,
        }
    )
    monitor._handle_lifecycle({"state": "discarded"})

    assert list(monitor._history) == [anchor]
    assert monitor._latest_paired_snapshot is not None
    monitor._replay.start_episode.assert_called_once()
    monitor._replay.discard.assert_called_once_with("discarded")


def test_evaluate_uses_owned_reference_images_instead_of_ring():
    cfg = _config(frame_offset=1)
    cfg["detection"]["send_anchor_images"] = False
    monitor = AdvantageOODMonitor(cfg)
    current = np.full((3, 4, 3), 2, dtype=np.uint8)
    reference_image = np.full((3, 4, 3), 1, dtype=np.uint8)
    snapshot = replay_module.FrameSnapshot(
        captured_at=2.0,
        playback_frame_index=-1,
        images={"front": current, "wrist_left": current},
        source_frame_ids={"front": 2, "wrist_left": 102},
    )
    reference = FrameAnchor(
        1.0,
        -1,
        {"front": 1, "wrist_left": 101},
        None,
        None,
        {"front": reference_image, "wrist_left": reference_image},
    )
    monitor._camera_reader = Mock()
    monitor._camera_reader.read_frame_ids.side_effect = AssertionError(
        "ring history should not be read"
    )
    monitor._model = Mock()
    monitor._model.predict.return_value = (0.1, 5.0)
    monitor._replay = Mock()
    monitor._replay.resolve_frame_index.return_value = None

    event = monitor._evaluate(snapshot, reference, "idle")

    assert event is not None
    assert event.advantage_score == 0.1
    monitor._camera_reader.read_frame_ids.assert_not_called()
    call = monitor._model.predict.call_args.kwargs
    assert np.array_equal(call["reference_front"], reference_image)
    assert np.array_equal(call["reference_wrist"], reference_image)


def test_checkpoint_resolver_selects_highest_numeric_step(tmp_path):
    for step in ("500", "1000"):
        path = tmp_path / step
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"weights")
    ignored = tmp_path / "latest"
    ignored.mkdir()
    (ignored / "model.safetensors").write_bytes(b"weights")

    assert resolve_advantage_checkpoint(tmp_path) == (tmp_path / "1000").resolve()


def test_checkpoint_resolver_has_actionable_empty_directory_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No Advantage model.safetensors"):
        resolve_advantage_checkpoint(tmp_path)


def test_advantage_client_rejects_non_rgb_or_non_uint8_inputs():
    with pytest.raises(ValueError, match=r"\[H,W,3\]"):
        AdvantageModelClient._validate_image("front", np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB uint8"):
        AdvantageModelClient._validate_image(
            "front", np.zeros((4, 4, 3), dtype=np.float32)
        )


def test_advantage_client_sets_prompt_for_generic_config_without_default():
    @dataclass(frozen=True)
    class DataConfig:
        default_prompt: str | None = None

    @dataclass(frozen=True)
    class TrainConfig:
        name: str
        data: DataConfig

    original = TrainConfig("ADVANTAGE_TORCH_UMI", DataConfig())
    configured = AdvantageModelClient._with_task_prompt(original, "stamp the paper")

    assert configured.data.default_prompt == "stamp the paper"
    assert original.data.default_prompt is None
    assert configured.name == original.name


@pytest.mark.parametrize("prompt", [None, "", "   ", 123])
def test_advantage_client_rejects_empty_or_non_string_task_prompts(prompt):
    with pytest.raises(ValueError, match="non-empty string"):
        AdvantageModelClient._with_task_prompt(None, prompt)


def test_advantage_client_requires_a_configurable_prompt_field():
    @dataclass(frozen=True)
    class DataConfig:
        repo_id: str = "dataset"

    @dataclass(frozen=True)
    class TrainConfig:
        name: str
        data: DataConfig

    original = TrainConfig("not_a_prompt_config", DataConfig())
    with pytest.raises(ValueError, match="no configurable default prompt"):
        AdvantageModelClient._with_task_prompt(original, "stamp the paper")


def test_advantage_yaml_is_standalone_and_contains_deployment_controls():
    with (REPO_ROOT / "config" / "advantage_ood.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    assert "ood" not in cfg
    assert cfg["runtime"]["advantage_root"] == "advantage"
    assert cfg["model"]["ckpt_dir"] == "checkpoints/advantage/ft-base"
    assert cfg["model"]["prompt"]
    assert np.isfinite(float(cfg["detection"]["threshold"]))
    assert cfg["detection"]["frame_offset"] == 50
    assert cfg["detection"]["interval_s"] == 0.5
    assert cfg["detection"]["send_anchor_images"] is True
    assert cfg["replay"]["pretrigger_s"] == 3.0
