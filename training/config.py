# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0


import dataclasses
import logging
import math
import os
from dataclasses import asdict
from typing import ClassVar, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from training.envs import SupportedEnvType
from training.scheduler.cluster import Cluster
from training.utils.placement import HybridComponentPlacement


@dataclasses.dataclass(frozen=True)
class SupportedModel:
    value: str

    models: ClassVar[dict[str, "SupportedModel"]] = {}

    @classmethod
    def register(cls, value: str, force: bool = False) -> "SupportedModel":
        if not value:
            raise ValueError("model_type must be a non-empty string.")
        if value in cls.models:
            if not force:
                raise ValueError(
                    f"Model type `{value}` is already registered. "
                    "Set force=True to override it."
                )
        else:
            cls.models[value] = cls.__private_create__(value)
        return cls.models[value]

    @classmethod
    def get(cls, value: str) -> "SupportedModel":
        if value not in cls.models:
            supported_models = sorted(cls.models)
            raise NotImplementedError(
                f"Model Type: {value} not supported. Supported models: {supported_models}"
            )
        return cls.models[value]

    def __new__(cls, value: str):
        return cls.get(value)

    @classmethod
    def __private_create__(cls, value: str) -> "SupportedModel":
        obj = object.__new__(cls)
        object.__setattr__(obj, "value", value)
        return obj


SupportedModel.OPENPI_PYTORCH = SupportedModel.register("openpi_pytorch", force=True)
EMBODIED_MODEL = {SupportedModel.OPENPI_PYTORCH}
SUPPORTED_TASK_TYPE = ["embodied_eval", "sft"]
SUPPORTED_TRAINING_BACKENDS = ["fsdp"]
__all__ = ["build_config"]


def torch_dtype_from_precision(
    precision: Union[int, str, None],
) -> Optional[torch.dtype]:
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    elif precision in [None, "null"]:
        return None
    else:
        raise ValueError(
            f"Could not parse the precision of `{precision}` to a valid torch.dtype"
        )


def validate_fsdp_cfg(cfg: DictConfig) -> DictConfig:
    def validate_amp_cfg(config: DictConfig) -> DictConfig:
        """Validate AMP configuration and ensure mutual exclusivity with FSDP mixed_precision."""

        param_dtype = config.mixed_precision.param_dtype
        reduce_dtype = config.mixed_precision.reduce_dtype
        buffer_dtype = config.mixed_precision.buffer_dtype

        all_none = param_dtype is None and reduce_dtype is None and buffer_dtype is None

        all_fp32 = (
            param_dtype == "fp32" and reduce_dtype == "fp32" and buffer_dtype == "fp32"
        )

        use_fsdp_mixed_precision = not (all_none or all_fp32)

        amp_autocast = config.get("amp_autocast", {})
        config.amp_autocast = {
            "enabled": amp_autocast.get("enabled", False),
            "precision": amp_autocast.get("precision", "bf16"),
        }

        grad_scaler = config.get("grad_scaler", {})
        config.grad_scaler = {
            "enabled": grad_scaler.get("enabled", False),
            "init_scale": grad_scaler.get("init_scale", None),
            "growth_interval": grad_scaler.get("growth_interval", None),
        }

        if "amp" in config:
            logging.warning(
                "fsdp_config.amp is no longer supported, use fsdp_config.amp_autocast and fsdp_config.grad_scaler instead"
            )

        if config.amp_autocast.enabled and use_fsdp_mixed_precision:
            assert False, (
                "amp_autocast should not be enabled when fsdp mixed_precision is enabled"
            )
        assert config.amp_autocast.precision in ["fp16", "bf16", "fp32"], (
            "fsdp.amp_autocast.precision must be one of ['fp16', 'bf16', 'fp32']"
        )
        return config

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.fsdp_config.strategy = cfg.fsdp_config.get("strategy", "fsdp")

        cfg.fsdp_config.sharding_strategy = cfg.fsdp_config.get(
            "sharding_strategy", "full_shard"
        )

        cfg.fsdp_config.forward_prefetch = cfg.fsdp_config.get(
            "forward_prefetch", False
        )
        cfg.fsdp_config.limit_all_gathers = cfg.fsdp_config.get(
            "limit_all_gathers", False
        )
        cfg.fsdp_config.backward_prefetch = cfg.fsdp_config.get(
            "backward_prefetch", None
        )
        cfg.fsdp_config.use_orig_params = cfg.fsdp_config.get("use_orig_params", False)
        cfg.fsdp_config.use_liger_kernel = cfg.fsdp_config.get(
            "use_liger_kernel", False
        )

        cfg.fsdp_config.cpu_offload = cfg.fsdp_config.get("cpu_offload", False)
        cfg.fsdp_config.offload_pin_memory = cfg.fsdp_config.get(
            "offload_pin_memory", False
        )
        cfg.fsdp_config.reshard_after_forward = cfg.fsdp_config.get(
            "reshard_after_forward", True
        )
        cfg.fsdp_config.enable_gradient_accumulation = cfg.fsdp_config.get(
            "enable_gradient_accumulation", False
        )

        assert cfg.fsdp_config.backward_prefetch in [
            None,
            "pre",
            "post",
        ], "fsdp_config.backward_prefetch must be one of [None, 'pre', 'post']"

        # validate mixed precision config
        assert hasattr(cfg.fsdp_config, "mixed_precision"), (
            "fsdp_config.mixed_precision is required in FSDP actor configuration."
        )
        mixed_precision_config = cfg.fsdp_config.mixed_precision
        mixed_precision_config.param_dtype = mixed_precision_config.get(
            "param_dtype", None
        )
        mixed_precision_config.reduce_dtype = mixed_precision_config.get(
            "reduce_dtype", None
        )
        mixed_precision_config.buffer_dtype = mixed_precision_config.get(
            "buffer_dtype", None
        )
        cfg.fsdp_config = validate_amp_cfg(cfg.fsdp_config)

    return cfg


def validate_embodied_cfg(cfg):
    """Validate the retained single-Franka evaluation configuration."""
    model_cfg = cfg.rollout.model
    assert SupportedModel(model_cfg.model_type) in EMBODIED_MODEL
    assert cfg.env.get("eval", None) is not None, "env.eval config is required"
    if cfg.env.get("train", None) is not None:
        raise ValueError(
            "Online environment training is not included; use the SFT entry point."
        )
    if cfg.get("reward", {}).get("use_reward_model", False):
        raise ValueError(
            "External reward workers are not included in policy evaluation."
        )
    if SupportedEnvType(cfg.env.eval.env_type) is not SupportedEnvType.REALWORLD:
        raise ValueError("HIL-UMI supports only the realworld backend")
    with open_dict(cfg):
        cfg.runner.val_check_interval = cfg.runner.get("val_check_interval", -1)
        cfg.env.eval.group_size = cfg.env.eval.get("group_size", 1)
        cfg.env.eval.rollout_epoch = cfg.env.eval.get("rollout_epoch", 1)
    component_placement = HybridComponentPlacement(cfg, Cluster())
    stage_num = cfg.rollout.pipeline_stage_num
    env_world_size = component_placement.get_world_size("env")
    if cfg.runner.get("enable_decoupled_mode", False):
        assert stage_num == 1, "enable_decoupled_mode requires one pipeline stage"
    total_envs = cfg.env.eval.total_num_envs
    assert total_envs > 0, "Evaluation requires at least one environment"
    assert total_envs % (env_world_size * stage_num) == 0, (
        "Evaluation environments must divide evenly across workers and pipeline stages"
    )
    assert total_envs // (env_world_size * stage_num) % cfg.env.eval.group_size == 0
    assert (
        cfg.env.eval.max_steps_per_rollout_epoch % model_cfg.num_action_chunks == 0
    ), "Evaluation rollout length must be divisible by the prediction horizon"
    return cfg


def validate_sft_cfg(cfg: DictConfig) -> DictConfig:
    assert cfg.actor.get("global_batch_size", None) is not None, (
        "the actor.global_batch_size is not set"
    )
    assert cfg.actor.get("micro_batch_size", None) is not None, (
        "the actor.micro_batch_size is not set"
    )

    with open_dict(cfg):
        post_train = _cfg_bool(cfg.data.get("post_train", False))
        mix_acbc = _cfg_bool(cfg.data.get("mix_acbc", False))
        if "post_train_demo_ratio" not in cfg.data:
            cfg.data.post_train_demo_ratio = 0.5
        if "post_train_correction_ratio" not in cfg.data:
            cfg.data.post_train_correction_ratio = 0.5
        if "mix_acbc_primary_ratio" not in cfg.data:
            cfg.data.mix_acbc_primary_ratio = 0.5
        if "mix_acbc_secondary_ratio" not in cfg.data:
            cfg.data.mix_acbc_secondary_ratio = 0.5
        if post_train:
            assert cfg.data.get("post_train_data_path", None) not in (None, ""), (
                "data.post_train=true requires data.post_train_data_path"
            )
            demo_ratio = float(cfg.data.post_train_demo_ratio)
            correction_ratio = float(cfg.data.post_train_correction_ratio)
            assert math.isfinite(demo_ratio) and demo_ratio > 0, (
                "data.post_train_demo_ratio must be a positive finite number"
            )
            assert math.isfinite(correction_ratio) and correction_ratio > 0, (
                "data.post_train_correction_ratio must be a positive finite number"
            )
        if mix_acbc:
            assert not post_train, (
                "data.mix_acbc=true cannot be combined with data.post_train=true"
            )
            assert _cfg_bool(cfg.data.get("acbc", False)), (
                "data.mix_acbc=true requires data.acbc=true"
            )
            assert _cfg_bool(cfg.data.get("prompt_from_task", False)), (
                "data.mix_acbc=true requires data.prompt_from_task=true"
            )
            assert cfg.data.get("mix_acbc_data_path", None) not in (None, ""), (
                "data.mix_acbc=true requires data.mix_acbc_data_path"
            )
            primary_ratio = float(cfg.data.mix_acbc_primary_ratio)
            secondary_ratio = float(cfg.data.mix_acbc_secondary_ratio)
            assert math.isfinite(primary_ratio) and primary_ratio > 0, (
                "data.mix_acbc_primary_ratio must be a positive finite number"
            )
            assert math.isfinite(secondary_ratio) and secondary_ratio > 0, (
                "data.mix_acbc_secondary_ratio must be a positive finite number"
            )

        if cfg.data.get("train_data_paths", None) is None:
            # if train_data_paths is None, the code will just eval the model
            assert cfg.data.get("val_data_paths", None) is not None, (
                "the data.train_data_paths is None, so data.val_data_paths is required"
            )
        elif cfg.data.get("val_data_paths", None) is not None:
            # set the val_check_interval to max_epochs
            if cfg.runner.get("val_check_interval", None) is None:
                cfg.runner.val_check_interval = cfg.runner.max_epochs
        else:
            # set the val_check_interval to -1 if there is no eval data or is not set
            cfg.runner.val_check_interval = cfg.runner.get("val_check_interval", -1)

        model_type = cfg.actor.model.get("model_type", None)
        if model_type is not None:
            SupportedModel(model_type)
    return cfg


def _cfg_bool(value: object) -> bool:
    """Parse bool-like config values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise AssertionError(f"Expected a boolean config value, got {value!r}")


def validate_cfg(cfg: DictConfig) -> DictConfig:
    """Validate the shipped single-Franka SFT and evaluation paths."""
    task_type = str(cfg.runner.task_type)
    if task_type not in SUPPORTED_TASK_TYPE:
        raise ValueError(f"task_type must be one of {SUPPORTED_TASK_TYPE}")
    only_eval = task_type == "embodied_eval"
    model_cfg = cfg.rollout.model if only_eval else cfg.actor.model
    SupportedModel(str(model_cfg.model_type))
    if not only_eval and cfg.actor.training_backend not in SUPPORTED_TRAINING_BACKENDS:
        raise ValueError("HIL-UMI training requires the fsdp backend")
    if cfg.runner.get("use_training_pipeline", False):
        raise ValueError("The optional training pipeline backend is not shipped")

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.runner.per_worker_log = cfg.runner.get("per_worker_log", False)
        cfg.runner.per_worker_log_path = None
        if cfg.runner.per_worker_log:
            cfg.runner.per_worker_log_path = os.path.join(
                cfg.runner.logger.log_path, "worker_logs"
            )
        profiling_cfg = cfg.cluster.get("profiling", None)
        if profiling_cfg is not None and bool(profiling_cfg.get("enabled", True)):
            if not profiling_cfg.get("output_dir", None):
                cfg.cluster.profiling.output_dir = os.path.abspath(
                    os.path.join(
                        cfg.runner.logger.log_path,
                        cfg.runner.logger.experiment_name,
                        "profiling",
                    )
                )
        if only_eval:
            cfg.runner.only_eval = True

    Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    cfg = validate_sft_cfg(cfg) if task_type == "sft" else validate_embodied_cfg(cfg)
    if only_eval:
        return cfg
    component_placement = HybridComponentPlacement(cfg, Cluster())
    actor_world_size = component_placement.get_world_size("actor")
    assert (
        cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * actor_world_size)
        == 0
    ), (
        "actor.global_batch_size must be divisible by actor.micro_batch_size * actor_world_size"
    )
    cfg.actor = validate_fsdp_cfg(cfg.actor)
    return cfg


def build_config(cls, cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        cfg = asdict(cfg)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in cfg:
            kwargs[f.name] = cfg.get(f.name)

    return cls(**kwargs)
