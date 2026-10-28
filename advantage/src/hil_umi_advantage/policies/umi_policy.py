# Modified for HIL-UMI.
"""Policy transforms for single-arm Franka UMI datasets."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

from hil_umi_advantage import transforms


def _parse_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class UMIInputs(transforms.DataTransformFn):
    """Convert two-camera UMI samples to the image names used by pi0.

    The optional ``his_-100_*`` images are the randomly sampled comparison
    observation used by the ADVANTAGE advantage estimator. The name is retained for
    compatibility; the comparison frame is not necessarily 100 frames earlier.
    """

    mask_state: bool = True

    rename_map: ClassVar[dict[str, str]] = {
        "front": "base_0_rgb",
        "left_wrist": "left_wrist_0_rgb",
        "his_-100_front": "base_-100_rgb",
        "his_-100_left_wrist": "left_wrist_-100_rgb",
    }
    required_cameras: ClassVar[tuple[str, ...]] = ("front", "left_wrist")

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        missing = set(self.required_cameras) - set(images)
        if missing:
            raise ValueError(f"Missing required UMI cameras: {sorted(missing)}")
        unexpected = set(images) - set(self.rename_map)
        if unexpected:
            raise ValueError(f"Unexpected UMI cameras: {sorted(unexpected)}")

        state = np.asarray(data["state"])
        inputs = {
            "image": {self.rename_map[key]: _parse_image(value) for key, value in images.items()},
            "image_mask": {self.rename_map[key]: np.True_ for key in images},
            "state": np.zeros_like(state) if self.mask_state else state,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        for key in ("frame_index", "episode_length", "progress", "episode_index"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class UMIOutputs(transforms.DataTransformFn):
    """Return the original 10-dimensional Franka UMI action representation."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}
