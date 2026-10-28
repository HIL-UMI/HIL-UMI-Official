# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class SupportedEnvType(Enum):
    REALWORLD = "realworld"


def get_env_cls(env_type: str, env_cfg=None):
    """Resolve the real-world backend included in this release."""
    if SupportedEnvType(env_type) is not SupportedEnvType.REALWORLD:
        raise ValueError(f"HIL-UMI ships only the realworld backend, got {env_type!r}")
    from training.envs.realworld import RealWorldEnv

    return RealWorldEnv
