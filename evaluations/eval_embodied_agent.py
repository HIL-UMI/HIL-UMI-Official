# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from training.config import validate_cfg
from training.runners.embodied_eval_runner import EmbodiedEvalRunner
from training.scheduler import Cluster
from training.utils.placement import HybridComponentPlacement
from training.workers.env.env_worker import EnvWorker
from training.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)
logger = logging.getLogger(__name__)


def _close_worker_group(worker_group, *, cleanup_workers: bool = False) -> None:
    """Best-effort cleanup for a launched worker group."""
    if worker_group is None:
        return
    if cleanup_workers:
        try:
            worker_group.cleanup().wait()
        except Exception:
            logger.exception("Environment worker cleanup failed")
    try:
        worker_group._close()
    except Exception:
        logger.exception("Worker group shutdown failed")


@hydra.main(
    version_base="1.1",
    config_path="realworld",
    config_name="umi_left_franka_tcp_rot6d_eval",
)
def main(cfg) -> None:
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    rollout_group = None
    env_group = None
    try:
        # Create rollout worker group
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_placement,
        )
        # Create env worker group
        env_placement = component_placement.get_strategy("env")
        env_group = EnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

        runner = EmbodiedEvalRunner(
            cfg=cfg,
            rollout=rollout_group,
            env=env_group,
        )

        runner.init_workers()
        runner.run()
    finally:
        _close_worker_group(env_group, cleanup_workers=True)
        _close_worker_group(rollout_group)


if __name__ == "__main__":
    main()
