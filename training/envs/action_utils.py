# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from training.envs import SupportedEnvType


def resolve_action_execution_horizon(env_cfg, prediction_horizon: int) -> int:
    """Resolve and validate the number of predicted actions to execute.

    ``num_action_chunks`` remains the model prediction horizon. Evaluation
    environments may set ``action_execution_horizon`` to execute only a prefix
    before requesting a fresh prediction.
    """
    configured = env_cfg.get("action_execution_horizon", prediction_horizon)
    try:
        execution_horizon = int(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"action_execution_horizon must be an integer, got {configured!r}."
        ) from exc
    if not 1 <= execution_horizon <= prediction_horizon:
        raise ValueError(
            "action_execution_horizon must satisfy "
            f"1 <= horizon <= {prediction_horizon}, got {execution_horizon}."
        )
    return execution_horizon


def resolve_sync_action_chunk_final_wait_s(env_cfg) -> float:
    """Resolve the extra wait after the final synchronous chunk action.

    The wait is applied before collecting the observation used for the next
    synchronous inference call. Asynchronous evaluation does not consume this
    setting.
    """
    configured = env_cfg.get("sync_action_chunk_final_wait_s", 0.0)
    if isinstance(configured, bool):
        raise ValueError(
            "sync_action_chunk_final_wait_s must be a finite non-negative "
            f"number, got {configured!r}."
        )
    try:
        wait_s = float(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sync_action_chunk_final_wait_s must be a finite non-negative "
            f"number, got {configured!r}."
        ) from exc
    if not np.isfinite(wait_s) or wait_s < 0.0:
        raise ValueError(
            "sync_action_chunk_final_wait_s must be a finite non-negative "
            f"number, got {configured!r}."
        )
    return wait_s


def compute_chunk_steps(
    max_steps_per_rollout_epoch: int,
    action_execution_horizon: int,
) -> int:
    """Return the number of predict/execute cycles in one rollout epoch."""
    if action_execution_horizon <= 0:
        raise ValueError("action_execution_horizon must be positive.")
    return int(max_steps_per_rollout_epoch) // action_execution_horizon


def slice_action_chunk(
    actions: torch.Tensor | np.ndarray,
    action_execution_horizon: int,
) -> torch.Tensor | np.ndarray:
    """Keep the leading action prefix that will be executed by the environment."""
    if actions.ndim != 3:
        raise ValueError(
            "Predicted actions must have shape [batch, horizon, action_dim], "
            f"got {tuple(actions.shape)}."
        )
    predicted_horizon = int(actions.shape[1])
    if action_execution_horizon > predicted_horizon:
        raise ValueError(
            f"Cannot execute {action_execution_horizon} actions from a predicted "
            f"chunk of length {predicted_horizon}."
        )
    prefix = actions[:, :action_execution_horizon]
    if isinstance(prefix, torch.Tensor):
        return prefix.contiguous()
    return np.ascontiguousarray(prefix)


def prepare_actions(
    raw_chunk_actions,
    env_type: str,
    model_type: str,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
    wm_env_type=None,
    env_cfg=None,
) -> torch.Tensor | np.ndarray:
    if isinstance(raw_chunk_actions, torch.Tensor):
        raw_chunk_actions = raw_chunk_actions.detach().cpu().contiguous()
        if raw_chunk_actions.dtype == torch.bfloat16:
            raw_chunk_actions = raw_chunk_actions.float()
        raw_chunk_actions = raw_chunk_actions.numpy()

    if SupportedEnvType(env_type) is not SupportedEnvType.REALWORLD:
        raise ValueError(f"Unsupported HIL-UMI environment: {env_type!r}")
    return raw_chunk_actions
