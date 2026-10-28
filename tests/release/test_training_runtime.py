# Copyright 2026 The HIL-UMI Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the retained training package and renamed runtime entrypoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from training.config import SupportedModel, validate_cfg
from training.envs import get_env_cls
from training.scheduler.cluster.cluster import Cluster, ClusterEnvVar

ROOT = Path(__file__).resolve().parents[2]


def test_realworld_package_import_does_not_load_hardware_or_ros(monkeypatch):
    for dependency in ("cv2", "rosgraph", "pyrealsense2"):
        monkeypatch.setitem(sys.modules, dependency, None)
    source = ROOT / "training" / "envs" / "realworld" / "__init__.py"
    name = "release_realworld_import_contract"
    spec = importlib.util.spec_from_file_location(
        name, source, submodule_search_locations=[str(source.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert "RealWorldEnv" in module.__all__
    assert "FrankaEnv" not in vars(module)


@pytest.mark.parametrize("model_type", ["qwen2.5", "openvla", "gr00t"])
def test_removed_models_are_rejected_by_public_registry(model_type):
    with pytest.raises(NotImplementedError, match="not supported"):
        SupportedModel(model_type)


def test_removed_task_is_rejected_before_ray_initialization(monkeypatch):
    def fail_if_cluster_starts(*_args, **_kwargs):
        raise AssertionError("Unsupported task must not create a Ray cluster")

    monkeypatch.setattr(Cluster, "__init__", fail_if_cluster_starts)
    cfg = OmegaConf.create({"runner": {"task_type": "reasoning"}})
    with pytest.raises(ValueError, match="task_type"):
        validate_cfg(cfg)


@pytest.mark.parametrize("value", ["auto", "repo", "package"])
def test_ray_sync_ships_only_the_training_package(value, monkeypatch):
    configured = {"auto": "auto", "repo": str(ROOT), "package": str(ROOT / "training")}[
        value
    ]
    assert (
        Cluster.get_full_env_var_name(ClusterEnvVar.CODE_WORKING_DIR)
        == "TRAINING_CODE_WORKING_DIR"
    )
    monkeypatch.setenv("TRAINING_CODE_WORKING_DIR", configured)
    fragment, roots = Cluster._prepare_ray_code_sync_runtime_env_fragment()
    assert fragment == {"py_modules": [str((ROOT / "training").resolve())]}
    assert str(ROOT.resolve()) in roots
    child_fragment = (
        Cluster._job_code_sync_fragment_for_child_runtime_env(fragment) or {}
    )
    assert "py_modules" not in child_fragment
    assert "working_dir" not in child_fragment


def test_realworld_gym_factory_and_dummy_wrapper_need_no_ros(monkeypatch):
    import gymnasium as gym

    from training.envs.realworld.realworld_env import RealWorldEnv

    def fail_if_ros_starts():
        raise AssertionError("A dummy environment must not touch ROS")

    monkeypatch.setattr(
        RealWorldEnv, "realworld_setup", staticmethod(fail_if_ros_starts)
    )
    cfg = OmegaConf.create(
        {
            "override_cfg": {
                "is_dummy": True,
                "camera_serials": ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"],
                "camera_type": "realsense",
                "enable_camera_player": False,
                "image_width": 64,
                "image_height": 48,
                "step_frequency": 1000.0,
            },
            "video_cfg": {},
            "seed": 0,
            "use_fixed_reset_state_ids": False,
            "auto_reset": False,
            "ignore_terminations": False,
            "group_size": 1,
            "main_image_key": "wrist_1",
            "extra_image_key": "wrist_2",
            "state_keys": ["tcp_pose_rot6d", "gripper_position"],
            "init_params": {"id": "SingleFrankaTCPEnv-v1"},
            "max_episode_steps": 100,
        }
    )
    cls = get_env_cls("realworld")
    env = cls(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    try:
        assert gym.spec("SingleFrankaTCPEnv-v1").entry_point.startswith("training.")
        observation, _ = env.reset()
        assert tuple(observation["states"].shape) == (1, 10)
        assert tuple(observation["main_images"].shape) == (1, 48, 64, 3)
        actions = observation["states"].cpu().numpy().copy()
        result = env.step(actions, auto_reset=False)
        assert len(result) == 5
        assert np.isfinite(result[1].cpu().numpy()).all()
    finally:
        env.env.close()


def test_unshipped_environment_fails_clearly():
    with pytest.raises(ValueError):
        get_env_cls("libero")


@pytest.mark.parametrize("task_type", ["embodied", "reasoning"])
def test_online_training_tasks_are_rejected_without_starting_ray(
    task_type, monkeypatch
):
    def fail_if_cluster_starts(*_args, **_kwargs):
        raise AssertionError("Removed training tasks must not create a Ray cluster")

    monkeypatch.setattr(Cluster, "__init__", fail_if_cluster_starts)
    cfg = OmegaConf.create({"runner": {"task_type": task_type}})
    with pytest.raises(ValueError, match="task_type"):
        validate_cfg(cfg)


def test_removed_policy_task_is_rejected_before_loading_weights(tmp_path):
    from training.models.embodiment.openpi_pytorch import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(tmp_path / "missing-weights"),
            "precision": "fp32",
            "openpi": {"task": "dagger"},
        }
    )
    with pytest.raises(ValueError, match="use 'sft' or 'eval'"):
        get_model(cfg)


def test_sft_factory_preserves_checkpoint_keys_and_produces_trainable_loss(
    tmp_path, monkeypatch
):
    import torch
    from safetensors.torch import save_file

    from training.models.embodiment.openpi_pytorch import get_model
    from training.models.embodiment.openpi_pytorch.pi0_model.model import Observation
    from training.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
    from training.models.embodiment.openpi_pytorch.sft_action_model import (
        OpenPiPytorchSFTActionModel,
    )

    class SmallCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.action_dim = 32

        def compute_loss(self, observation, actions, *, train):
            assert train
            assert observation.state.shape == (2, 32)
            return ((self.weight - actions) ** 2).mean(dim=-1)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file({"weight": torch.tensor(0.75)}, checkpoint / "model.safetensors")
    monkeypatch.setattr(Pi0Config, "create", lambda _self: SmallCore())
    cfg = OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "precision": "fp32",
            "num_action_chunks": 20,
            "action_dim": 10,
            "num_steps": 5,
            "openpi": {
                "task": "sft",
                "model_action_dim": 32,
                "paligemma_variant": "dummy",
                "action_expert_variant": "dummy",
            },
        }
    )
    model = get_model(cfg)
    assert isinstance(model, OpenPiPytorchSFTActionModel)
    assert set(model.state_dict()) == {"model.weight"}
    assert model.model.weight.item() == 0.75
    observation = Observation(images={}, image_masks={}, state=torch.zeros(2, 32))
    loss = model(data=(observation, torch.zeros(2, 20, 32)))
    assert loss.ndim == 0
    loss.backward()
    assert torch.isfinite(model.model.weight.grad)
    assert model.model.weight.grad.item() != 0


def test_robot_wrapper_rejects_removed_policy_intervention():
    from types import SimpleNamespace

    from training.envs.realworld.common.wrappers.apply import (
        apply_single_franka_tcp_wrappers,
    )

    env = SimpleNamespace(config=SimpleNamespace(is_dummy=True))
    with pytest.raises(ValueError, match="external recorder"):
        apply_single_franka_tcp_wrappers(env, {"use_tcp_vive": True})
