# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Per-task model builders for the OpenPI 0.5 factory.

:func:`training.models.embodiment.openpi_pytorch.get_model` loads the vendored
``Pi0`` core and then dispatches on ``actor.model.openpi.task`` to one of the
builders here, each of which wraps the core in the concrete variant:

* :func:`_build_eval_model` → :class:`OpenPiPytorchEvalActionModel`
* :func:`_build_sft_model`  → :class:`OpenPiPytorchSFTActionModel`

The eval builder assembles the shared ``openpi.transforms`` pipeline via
:func:`transforms_pipeline.build_openpi_transforms`; the SFT builder holds no
transforms (the SFT data loader applies them upstream).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _resolve_data_kwargs(cfg):
    """Read the optional ``openpi_data`` override block as a plain dict."""
    from omegaconf import OmegaConf

    data_kwargs = OmegaConf.select(cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    return data_kwargs


def _build_eval_model(
    cfg,
    model_cfg,
    model,
    *,
    num_steps,
    action_chunk,
    action_env_dim,
):
    """Build the eval variant: openpi.transforms pipeline, no value head, no
    chain collection, no training-mode forward.

    Produces a bare :class:`OpenPiPytorchEvalActionModel` — same eval path the
    RL subclass uses, but stripped of every PPO-only bit. Selected by
    ``actor.model.openpi.task: eval``.
    """
    from omegaconf import OmegaConf

    from training.models.embodiment.openpi_pytorch.eval_action_model import (
        OpenPiPytorchEvalActionModel,
    )
    from training.models.embodiment.openpi_pytorch.transforms_pipeline import (
        build_openpi_transforms,
    )

    config_name = str(OmegaConf.select(model_cfg, "config_name", default=""))
    if not config_name:
        raise ValueError(
            "actor.model.openpi.config_name is required for task='eval' "
            "(it selects the upstream openpi TrainConfig, e.g. 'pi05_behavior')."
        )

    input_transforms, output_transforms = build_openpi_transforms(
        cfg.model_path, config_name, data_kwargs=_resolve_data_kwargs(cfg)
    )

    eval_model = OpenPiPytorchEvalActionModel(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
        action_chunk=action_chunk,
        config_name=config_name,
        state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
    )
    eval_model.setup_wrappers(input_transforms, output_transforms)
    return eval_model


def _build_sft_model(
    model,
    *,
    num_steps,
    action_env_dim,
):
    """Build the SFT variant.

    The observation/action transform is applied upstream in the environment SFT
    data loader, which routes each frame through the same openpi transform
    pipeline the eval/RL paths use, so the SFT model holds no processor and no
    transforms — it just computes the flow-matching loss.
    """
    from training.models.embodiment.openpi_pytorch.sft_action_model import (
        OpenPiPytorchSFTActionModel,
    )

    return OpenPiPytorchSFTActionModel(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
    )
