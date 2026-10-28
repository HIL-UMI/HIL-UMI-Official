# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from training.config import torch_dtype_from_precision
from training.utils.logging import get_logger

logger = get_logger()


def get_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Build an OpenPI PyTorch Pi0.5 model from ``actor.model`` config.

    ``cfg.model_path`` points at a new-format checkpoint containing
    ``model.safetensors``. Model shape comes from YAML; no checkpoint
    ``config.json`` is read. ``cfg.openpi.task`` selects the SFT or eval
    wrapper around the shared Pi0 core.
    """
    import pathlib

    import safetensors.torch
    from omegaconf import OmegaConf

    from training.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
    from training.models.embodiment.openpi_pytorch.utils.model_builders import (
        _build_eval_model,
        _build_sft_model,
    )

    model_cfg = cfg.openpi
    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )

    task = OmegaConf.select(model_cfg, "task", default=None)
    if task is None:
        raise ValueError(
            "actor.model.openpi.task is required: set it to 'sft' or "
            "'eval' to pick the concrete OpenPI PyTorch model variant."
        )
    task = str(task).lower()

    if task not in {"sft", "eval"}:
        raise ValueError(
            f"actor.model.openpi.task={task!r} is not supported; use 'sft' or 'eval'."
        )

    model_path = pathlib.Path(cfg.model_path)
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"openpi_pytorch checkpoint not found: {weights_path}")

    pi0_kwargs = {
        "pi05": True,
        "action_horizon": int(cfg.num_action_chunks),
        "action_dim": int(model_cfg.model_action_dim),
        "paligemma_variant": str(model_cfg.paligemma_variant),
        "action_expert_variant": str(model_cfg.action_expert_variant),
        "dtype": "bfloat16",
        "pcd": False,
    }
    discrete_state_input = OmegaConf.select(
        model_cfg, "discrete_state_input", default=None
    )
    if discrete_state_input is not None:
        pi0_kwargs["discrete_state_input"] = bool(discrete_state_input)
    max_token_len = OmegaConf.select(model_cfg, "max_token_len", default=None)
    if max_token_len is not None:
        pi0_kwargs["max_token_len"] = int(max_token_len)

    pi0_config = Pi0Config(**pi0_kwargs)
    model = pi0_config.create()
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(param.numel() for param in model.parameters())
    if target_dtype is not None:
        model = model.to(target_dtype)

    num_steps = int(cfg.num_steps)
    action_chunk = int(cfg.num_action_chunks)
    action_env_dim = int(cfg.action_dim)

    logger.info(
        "openpi_pytorch[%s]: loaded %s (%.2fB params) strict from %s "
        "precision=%s num_steps=%s",
        task,
        pi0_config,
        n_params / 1e9,
        weights_path,
        cfg.precision,
        num_steps,
    )

    if task == "eval":
        return _build_eval_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
        )

    if task == "sft":
        return _build_sft_model(
            model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )

    raise ValueError(
        f"actor.model.openpi.task={task!r} is not supported; use 'eval' or 'sft'."
    )
