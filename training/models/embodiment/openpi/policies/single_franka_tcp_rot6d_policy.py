# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.
"""Policy transforms for single-Franka TCP rot6d SFT.

state/actions = [xyz(3), rot6d(6), gripper(1)].
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

_TCP_ROT6D_ACTION_DIM = 10


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SingleFrankaTcpRot6dInputs(transforms.DataTransformFn):
    """Feeds single-Franka TCP rot6d observations into pi0 / pi05."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])[..., :_TCP_ROT6D_ACTION_DIM]
        state = transforms.pad_to_dim(state, self.action_dim)

        wrist_image = _parse_image(data["observation/image"])
        extra_view = data.get("observation/extra_view_image")
        if extra_view is None:
            base_image = np.zeros_like(wrist_image)
            base_mask = np.False_
        else:
            base_image = _parse_image(extra_view)
            base_mask = np.True_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # "left_wrist_0_rgb": np.zeros_like(wrist_image),
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": base_mask,
                "left_wrist_0_rgb": np.True_,
                # "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])[..., :_TCP_ROT6D_ACTION_DIM]
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SingleFrankaTcpRot6dOutputs(transforms.DataTransformFn):
    """Recover 10-d rot6d action from padded model output."""

    output_action_dim: int = _TCP_ROT6D_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
