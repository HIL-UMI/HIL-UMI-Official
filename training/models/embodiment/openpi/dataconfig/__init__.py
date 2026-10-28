# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0
# openpi model configs

import dataclasses
import difflib
import pathlib
from typing import Optional

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from omegaconf import DictConfig, OmegaConf
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)

from training.models.embodiment.openpi.dataconfig.franka_co_training_dataconfig import (
    LeRobotFrankaEEDataConfig,
)
from training.models.embodiment.openpi.dataconfig.realworld_dataconfig import (
    LeRobotRealworldDataConfig,
)
from training.models.embodiment.openpi.dataconfig.single_franka_tcp_rot6d_dataconfig import (
    SingleFrankaTcpRot6dDataConfig,
)

_CONFIGS = [
TrainConfig(
        name="pi05_franka",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=8, discrete_state_input=False
        ),  # discrete_state_input=False: stateless policy, True: with state policy
        data=LeRobotFrankaEEDataConfig(
            repo_id="physical-intelligence/real_rl",  # Not important
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="checkpoints/torch/pi05_franka_pretrained/assets"
            ),
            output_action_dim=6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        seed=0,
        batch_size=16,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=8,
        num_train_steps=5_000,
        log_interval=5,
        save_interval=250,
    ),
TrainConfig(
        name="pi0_realworld",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=LeRobotRealworldDataConfig(
            repo_id="realworld_franka_bin_relocation",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            extra_delta_transform=False,
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
TrainConfig(
        name="pi05_realworld",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotRealworldDataConfig(
            repo_id="realworld_franka_bin_relocation",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_base/assets"),
            extra_delta_transform=False,
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
TrainConfig(
        name="pi05_singlefranka_tcp_rot6d",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=20, discrete_state_input=False
        ),
        data=SingleFrankaTcpRot6dDataConfig(
            repo_id="umi_left_franka_tcp_rot6d",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_base/assets"),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    )
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    """Return a copy of the config with assets/weight paths set from model_path."""
    data_config = config.data
    if (
        dataclasses.is_dataclass(data_config)
        and hasattr(data_config, "assets")
        and dataclasses.is_dataclass(data_config.assets)
    ):
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, assets_dir=model_path),
        )

    replace_kwargs = {
        "data": data_config,
        "pytorch_weight_path": model_path,
    }
    if dataclasses.is_dataclass(config) and any(
        field.name == "assets_dirs" for field in dataclasses.fields(config)
    ):
        replace_kwargs["assets_dirs"] = model_path

    return dataclasses.replace(config, **replace_kwargs)


def _override_with_data_kwargs(config: TrainConfig, data_kwargs) -> TrainConfig:
    """Return a copy of the config with data_config set from openpi_data."""
    if isinstance(data_kwargs, DictConfig):
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    data_kwargs = dict(data_kwargs)
    norm_stats_path = data_kwargs.pop("norm_stats_path", None)
    data_config = dataclasses.replace(config.data, **data_kwargs)
    config = dataclasses.replace(config, data=data_config)
    if norm_stats_path is not None:
        norm_dir = pathlib.Path(norm_stats_path).expanduser()
        if norm_dir.is_file():
            norm_dir = norm_dir.parent
        new_data = dataclasses.replace(
            config.data,
            assets=dataclasses.replace(
                config.data.assets,
                assets_dir=str(norm_dir.parent),
                asset_id=norm_dir.name,
            ),
        )
        replace_kwargs: dict = {"data": new_data}
        if dataclasses.is_dataclass(config) and any(
            field.name == "assets_dirs" for field in dataclasses.fields(config)
        ):
            replace_kwargs["assets_dirs"] = norm_dir.parent
        config = dataclasses.replace(config, **replace_kwargs)
    return config


def get_openpi_config(
    config_name: str,
    model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    repo_id: Optional[str] = None,
    data_kwargs: Optional[dict] = None,
) -> TrainConfig:
    """Get a config by name.

    Args:
        config_name: Name of the config to load.
        model_path: Optional path to override model weights and assets.
        batch_size: Optional batch size override.
        repo_id: Optional LeRobot repo_id or local data path to override.
            When using a local path, the original asset_id is preserved so
            that norm_stats can still be loaded from the model checkpoint.
    """
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    config = _CONFIGS_DICT[config_name]
    norm_stats_path = None
    if data_kwargs is not None:
        if isinstance(data_kwargs, DictConfig):
            norm_stats_path = data_kwargs.get("norm_stats_path")
        else:
            norm_stats_path = data_kwargs.get("norm_stats_path")
    if model_path is not None:
        config = _override_with_model_path(config, model_path)
    if data_kwargs is not None:
        config = _override_with_data_kwargs(config, data_kwargs)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)

    if repo_id is not None:
        original_repo_id = config.data.repo_id
        assets = config.data.assets
        if norm_stats_path is None:
            assets = dataclasses.replace(assets, asset_id=original_repo_id)
        new_data = dataclasses.replace(config.data, repo_id=repo_id, assets=assets)
        config = dataclasses.replace(config, data=new_data)

    return config
