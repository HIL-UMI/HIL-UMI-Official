# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from training.config import validate_cfg
from training.runners.sft_runner import SFTRunner
from training.scheduler import Cluster
from training.utils.placement import HybridComponentPlacement
from training.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="umi_left_franka_sft_openpi_pytorch_tcp_rot6d"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    logging.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")

    if cfg.actor.training_backend == "fsdp" or cfg.actor.training_backend == "fsdp2":
        actor_group = FSDPVlaSftWorker.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )

    else:
        raise ValueError(f"{cfg.actor.training_backend} backend is not supported yet")

    runner = SFTRunner(
        cfg=cfg,
        actor=actor_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
