# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Real-world interfaces; hardware dependencies load only when requested."""

from importlib import import_module

_EXPORTS = {
    "FrankaEnv": (".franka", "FrankaEnv"),
    "FrankaRobotConfig": (".franka", "FrankaRobotConfig"),
    "FrankaRobotState": (".franka", "FrankaRobotState"),
    "SingleFrankaTCPEnv": (".franka.tasks.single_franka_tcp_env", "SingleFrankaTCPEnv"),
    "SingleFrankaTCPRobotConfig": (
        ".franka.tasks.single_franka_tcp_env",
        "SingleFrankaTCPRobotConfig",
    ),
    "RealWorldEnv": (".realworld_env", "RealWorldEnv"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value
