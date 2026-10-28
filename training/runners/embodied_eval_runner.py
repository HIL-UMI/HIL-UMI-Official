# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import time
import typing

from training.scheduler import Channel
from training.scheduler import WorkerGroupFuncResult as Handle
from training.utils.distributed import ScopedTimer
from training.utils.logging import get_logger
from training.utils.metric_logger import MetricLogger
from training.utils.metric_utils import compute_evaluate_metrics, print_metrics_table

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from training.workers.env.env_worker import EnvWorker
    from training.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_time = time.time()
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)
        print_metrics_table(
            step=0,
            total_steps=1,
            start_time=start_time,
            metrics=eval_metrics,
            log_path=self.metric_logger.log_path,
        )

        self.metric_logger.finish()
