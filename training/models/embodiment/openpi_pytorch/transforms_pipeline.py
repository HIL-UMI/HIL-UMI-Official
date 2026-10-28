# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
from typing import Any, Sequence


def _explicit_norm_stats_location(
    data_kwargs: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """Resolve an explicit norm-stats file/directory into loader arguments."""
    if not data_kwargs:
        return None
    norm_stats_path = data_kwargs.get("norm_stats_path")
    if not norm_stats_path:
        return None
    norm_dir = pathlib.Path(str(norm_stats_path)).expanduser()
    if norm_dir.name == "norm_stats.json":
        norm_dir = norm_dir.parent
    return str(norm_dir.parent), norm_dir.name


def build_openpi_transforms(
    model_path: str,
    config_name: str,
    data_kwargs: dict[str, Any] | None = None,
    *,
    norm_stats_dir: str | None = None,
    norm_stats_asset_id: str | None = None,
) -> tuple[Sequence, Sequence]:
    """Build ``(input_transforms, output_transforms)`` for ``config_name``.

    Returns two lists ready for :func:`openpi.transforms.compose`, matching
    ``training/models/embodiment/openpi/__init__.py`` exactly:

    * input:  ``[InjectDefaultPrompt(None), *data.inputs, Normalize, *model.inputs]``
    * output: ``[*model.outputs, Unnormalize, *data.outputs]``

    Norm stats resolve in this order:

    1. explicit ``norm_stats_dir`` + ``norm_stats_asset_id`` arguments;
    2. ``data_kwargs["norm_stats_path"]`` (file or containing directory);
    3. checkpoint-relative ``{model_path}/{data_config.asset_id}``.

    This lets real-robot eval reuse task stats outside a checkpoint while SFT
    loaders keep their explicit assets-dir/asset-id contract.
    """
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    from training.models.embodiment.openpi.dataconfig import get_openpi_config

    train_config = get_openpi_config(
        config_name, model_path=str(model_path), data_kwargs=data_kwargs
    )
    upstream_model_config = train_config.model

    data_config = train_config.data.create(
        train_config.assets_dirs, upstream_model_config
    )

    explicit_location = _explicit_norm_stats_location(data_kwargs)
    if norm_stats_dir is None and explicit_location is not None:
        norm_stats_dir, explicit_asset_id = explicit_location
        if norm_stats_asset_id is None:
            norm_stats_asset_id = explicit_asset_id

    asset_id = norm_stats_asset_id or data_config.asset_id
    if asset_id is None:
        raise ValueError("asset_id is required to load norm_stats.")
    stats_dir = (
        norm_stats_dir
        if norm_stats_dir is not None
        else download.maybe_download(str(model_path))
    )
    norm_stats = _checkpoints.load_norm_stats(stats_dir, asset_id)
    if norm_stats is None:
        raise FileNotFoundError(
            f"openpi_pytorch: norm_stats not found at {stats_dir}/{asset_id}/"
            "norm_stats.json. For eval/RL the checkpoint dir must bundle them; "
            "for SFT set actor.model.openpi.assets_dir/asset_id to the stats dir."
        )

    input_transforms = [
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    return input_transforms, output_transforms
