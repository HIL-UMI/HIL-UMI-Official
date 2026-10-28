# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0


"""Wrapper stack for the released absolute 10D single-Franka workflow."""

from typing import Any, Mapping

import gymnasium as gym

from .keyboard_eval_control_wrapper import KeyboardEvalControlWrapper


def _apply_keyboard_wrapper(env: gym.Env, mode: str | None) -> gym.Env:
    if env.get_wrapper_attr("config").is_dummy or not mode:
        return env
    if mode == "eval_control":
        return KeyboardEvalControlWrapper(env)
    raise ValueError(f"Unsupported keyboard wrapper {mode!r}; use eval_control")


def apply_single_franka_tcp_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for absolute single-Franka TCP rot6d policies.

    Relative-frame and quaternion-to-Euler wrappers are intentionally skipped:
    the environment already exposes the exact 10D rot6d policy contract.
    Autonomous evaluation may still use the keyboard start/success/failure
    gate.
    """
    unsupported_modes = {
        "use_spacemouse": cfg.get("use_spacemouse", False),
        "use_gello": cfg.get("use_gello", False),
        "use_pico": cfg.get("use_pico", False),
        "use_vive": cfg.get("use_vive", False),
    }
    active_modes = [
        name for name, enabled in unsupported_modes.items() if bool(enabled)
    ]
    if active_modes:
        raise ValueError(
            "SingleFrankaTCPEnv autonomous deployment does not apply delta "
            f"teleop wrappers; disable: {', '.join(active_modes)}."
        )
    if cfg.get("no_gripper", False):
        raise ValueError(
            "SingleFrankaTCPEnv requires no_gripper=False because action[9] "
            "is the policy gripper command."
        )

    if cfg.get("use_tcp_vive", False):
        raise ValueError(
            "VIVE policy intervention is not included; use the external recorder adapter for collection."
        )
    return _apply_keyboard_wrapper(
        env,
        cfg.get("keyboard_reward_wrapper", None),
    )
