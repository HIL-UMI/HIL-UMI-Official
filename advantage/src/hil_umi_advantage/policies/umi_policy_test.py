import numpy as np
import pytest

from hil_umi_advantage.policies import umi_policy


def _sample() -> dict:
    return {
        "images": {
            "front": np.full((3, 8, 12), 0.5, dtype=np.float32),
            "left_wrist": np.full((3, 8, 12), 0.25, dtype=np.float32),
            "his_-100_front": np.full((3, 8, 12), 0.75, dtype=np.float32),
            "his_-100_left_wrist": np.full((3, 8, 12), 1.0, dtype=np.float32),
        },
        "state": np.arange(10, dtype=np.float32),
        "actions": np.ones((50, 10), dtype=np.float32),
        "progress": np.float32(-0.25),
        "frame_index": np.int64(10),
        "episode_index": np.int64(2),
        "episode_length": np.int64(100),
    }


def test_umi_inputs_maps_two_timestamps_and_masks_state():
    result = umi_policy.UMIInputs(mask_state=True)(_sample())

    assert tuple(result["image"]) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "base_-100_rgb",
        "left_wrist_-100_rgb",
    )
    assert all(image.shape == (8, 12, 3) for image in result["image"].values())
    assert all(image.dtype == np.uint8 for image in result["image"].values())
    assert all(result["image_mask"].values())
    np.testing.assert_array_equal(result["state"], np.zeros(10, dtype=np.float32))
    assert result["actions"].shape == (50, 10)
    assert result["progress"] == np.float32(-0.25)


def test_umi_inputs_requires_both_current_cameras():
    sample = _sample()
    del sample["images"]["front"]
    with pytest.raises(ValueError, match="front"):
        umi_policy.UMIInputs()(sample)


def test_umi_outputs_slices_to_ten_dimensions():
    actions = np.zeros((4, 32), dtype=np.float32)
    assert umi_policy.UMIOutputs()({"actions": actions})["actions"].shape == (4, 10)
