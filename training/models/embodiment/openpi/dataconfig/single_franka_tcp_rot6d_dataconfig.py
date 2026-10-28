# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0
"""Data config for single-Franka TCP rot6d SFT."""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from training.models.embodiment.openpi.policies import single_franka_tcp_rot6d_policy
from training.models.embodiment.openpi.transforms import (
    SINGLE_ARM_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)


@dataclasses.dataclass(frozen=True)
class SingleFrankaTcpRot6dDataConfig(DataConfigFactory):
    default_prompt: str | None = None

    # SE(3) delta at train, absolute recovery at inference. The gripper slot
    # stays absolute through the rigid-body transform.
    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/extra_view_image": "extra_view_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                single_franka_tcp_rot6d_policy.SingleFrankaTcpRot6dInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[single_franka_tcp_rot6d_policy.SingleFrankaTcpRot6dOutputs()],
        )

        if self.extra_delta_transform:
            data_transforms = data_transforms.push(
                inputs=[RigidBodyDeltaActions(SINGLE_ARM_ROT6D_LAYOUT)],
                outputs=[RigidBodyAbsoluteActions(SINGLE_ARM_ROT6D_LAYOUT)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )
