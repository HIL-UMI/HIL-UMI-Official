# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0
"""Foot-pedal-gated wrapper for autonomous policy eval.

Pedal: ``a`` starts a rollout from idle; ``c`` ends with reward=1
("success"); ``b`` ends with reward=0 ("failure"). On end, returns
``terminated=True`` and lets the outer ``auto_reset`` drive home.
"""

import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from training.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardEvalControlWrapper(gym.Wrapper):
    """Foot-pedal-gated start/stop for autonomous policy eval rollouts."""

    IDLE_POLL_S = 0.05
    PEDAL_DEBOUNCE_S = 0.2
    WAIT_HEARTBEAT_S = 10.0

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._running = False
        self._last_obs: Any = None
        self._last_press_ts: dict[str, float] = {}
        self._epoch_index = int(
            getattr(self._base_env(), "scene_alignment_epoch_offset", 0)
        )

    def reset(self, *, seed=None, options=None):
        self._last_press_ts.clear()
        self.listener.pop_pressed_keys()
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        base_env = self._base_env()
        begin_alignment = getattr(base_env, "begin_scene_alignment", None)
        alignment_enabled = bool(
            callable(begin_alignment) and begin_alignment(self._epoch_index)
        )
        # Block until the operator has arranged the scene and presses 'a'.
        # This is intentional (the arms are homed and idle); log on entry and
        # emit a periodic heartbeat so the wait is not mistaken for a hang.
        self._log_info(
            "Arms homed and idle. Arrange the scene, then press pedal 'a' "
            "to start the next rollout (Ctrl-C to abort)."
        )
        last_heartbeat = time.monotonic()
        while True:
            if alignment_enabled:
                base_env.refresh_scene_alignment()
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self._log_info("Still waiting for pedal 'a' to start the rollout...")
            for key in self.listener.pop_pressed_keys():
                if key == "a":
                    if alignment_enabled:
                        obs = base_env.capture_scene_alignment_observation()
                        self._last_obs = obs
                        base_env.finish_scene_alignment()
                    self._running = True
                    self._epoch_index += 1
                    self._log_info("Pedal 'a' pressed; starting rollout.")
                    return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            # Idle: poll pedal, hold robot via the controller's last target.
            time.sleep(self.IDLE_POLL_S)
            for key in self.listener.pop_pressed_keys():
                now = time.monotonic()
                if (
                    now - self._last_press_ts.get(key, -math.inf)
                    < self.PEDAL_DEBOUNCE_S
                ):
                    continue
                self._last_press_ts[key] = now
                if key == "a":
                    self._running = True
                    return self._idle_response(event="start")
            return self._idle_response(event=None)

        # Running: forward to the wrapped env.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        terminated = False
        truncated = False

        result: str | None = None
        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                continue
            self._last_press_ts[key] = now
            if key == "c":
                terminated = True
                reward = 1.0
                result = "success"
                self._running = False
                break
            if key == "b":
                terminated = True
                reward = 0.0
                result = "failure"
                self._running = False
                break

        if result is not None:
            base_env = self._base_env()
            save_curve = getattr(base_env, "save_state_action_curve", None)
            if callable(save_curve):
                curve_path = save_curve(result=result)
                if curve_path is not None:
                    info["state_action_curve_path"] = curve_path
            save_pose_trace = getattr(base_env, "save_pose_trace", None)
            if callable(save_pose_trace):
                pose_trace_path = save_pose_trace(result=result)
                if pose_trace_path is not None:
                    info["pose_trace_path"] = pose_trace_path
        info["eval_phase"] = "rec" if self._running else "pre"
        info["eval_result"] = result
        return obs, reward, terminated, truncated, info

    def _idle_response(self, event: str | None):
        info = {"eval_phase": "pre", "eval_event": event, "eval_result": None}
        return self._last_obs, 0.0, False, False, info

    def _log_info(self, message: str) -> None:
        logger = getattr(self._base_env(), "_logger", None)
        if logger is not None:
            logger.info(message)

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)
