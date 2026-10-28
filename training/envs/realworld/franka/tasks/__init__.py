# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from training.envs.realworld.common.wrappers import apply_single_franka_tcp_wrappers

from .single_franka_tcp_env import SingleFrankaTCPEnv


def create_single_franka_tcp_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    return apply_single_franka_tcp_wrappers(
        SingleFrankaTCPEnv(
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
        ),
        env_cfg,
    )


register(
    id="SingleFrankaTCPEnv-v1",
    entry_point="training.envs.realworld.franka.tasks:create_single_franka_tcp_env",
)
