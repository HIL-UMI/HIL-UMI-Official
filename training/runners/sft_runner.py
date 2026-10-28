# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
from typing import TYPE_CHECKING, Optional

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from training.scheduler import WorkerGroupFuncResult as Handle
from training.utils.distributed import ScopedTimer
from training.utils.metric_logger import MetricLogger
from training.utils.runner_utils import EarlyStopController, check_progress

if TYPE_CHECKING:
    from training.workers.sft.fsdp_sft_worker import FSDPSftWorker

logger = logging.getLogger(__name__)


class SFTRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: "FSDPSftWorker",
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0
        early_stop_cfg = cfg.runner.get("early_stop", None)
        self.early_stop = (
            EarlyStopController(early_stop_cfg) if early_stop_cfg is not None else None
        )

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self) -> None:
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def run(self) -> None:
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        for _step in range(start_step, self.max_steps):
            if hasattr(self.actor, "set_global_step"):
                # set global step
                self.actor.set_global_step(self.global_step)

            with self.timer("step"):
                actor_handle: Handle = self.actor.run_training()
                actor_metrics = actor_handle.wait()

                self.global_step += 1

                eval_model, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

                should_stop = False
                if eval_model:
                    eval_handle: Handle = self.actor.run_eval()
                    eval_metrics = eval_handle.wait()

                    if self.early_stop is not None:
                        should_stop, best_val_acc_improved = self.early_stop.update(
                            eval_metrics[0]
                        )
                        if best_val_acc_improved:
                            self._save_checkpoint(is_best=True)

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            if eval_model:
                time_metrics["evaluate"] = eval_handle.consume_duration()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            merged_actor_metrics = {}
            # get the merged actor metrics from all ranks
            for metrics in actor_metrics:
                for k, v in metrics.items():
                    if k not in merged_actor_metrics:
                        merged_actor_metrics[k] = v
            training_metrics = {
                f"train/{k}": v for k, v in merged_actor_metrics.items()
            }
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(training_metrics)

            if eval_model:
                evaluate_metrics = {f"eval/{k}": v for k, v in eval_metrics[0].items()}
                logging_metrics.update(evaluate_metrics)
                self.metric_logger.log(evaluate_metrics, _step)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)
            if should_stop:
                break

        if self.early_stop is not None and self.early_stop.best_val_acc > 0:
            logger.info(
                f"Early stopping triggered! Best val_acc: {self.early_stop.best_val_acc:.4f}"
            )
        self.metric_logger.finish()

    def run_eval(self) -> None:
        with self.timer("evaluate"):
            eval_handle: Handle = self.actor.run_eval()
            eval_metrics = eval_handle.wait()

        time_metrics = self.timer.consume_durations()
        time_metrics["evaluate"] = eval_handle.consume_duration()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

        raw_eval = (
            eval_metrics[0]
            if isinstance(eval_metrics, (list, tuple)) and len(eval_metrics) > 0
            else {}
        )
        evaluate_metrics = {f"eval/{k}": v for k, v in raw_eval.items()}

        logging_metrics = {}
        logging_metrics.update(time_metrics)
        logging_metrics.update(evaluate_metrics)

        logger.info(f"Eval metrics: {evaluate_metrics}")
        self.metric_logger.finish()

    def _save_checkpoint(self, is_best: bool = False) -> None:
        checkpoint_root = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
        )
        if is_best:
            base_output_dir = os.path.join(checkpoint_root, "checkpoints/best_model")
        else:
            base_output_dir = os.path.join(
                checkpoint_root,
                f"checkpoints/global_step_{self.global_step}",
            )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        if is_best and self.early_stop is not None:
            logger.info(
                f"Saved best model (val_acc={self.early_stop.best_val_acc:.4f}) to {base_output_dir}"
            )
        self._maybe_offload_checkpoint(base_output_dir)

    def _maybe_offload_checkpoint(self, local_dir: str) -> None:
        """Move a saved checkpoint dir to ``runner.checkpoint_offload_dir``.

        The checkpoint is moved right after the save completes, so large
        checkpoints do not accumulate on the local (training) filesystem.
        The destination mirrors the directory layout relative to the parent
        of ``runner.logger.log_path``, e.g.::

            <offload_dir>/<run_dir>/<experiment_name>/checkpoints/global_step_<N>

        The local copy is removed only after the copy has fully completed.
        """
        offload_root = self.cfg.runner.get("checkpoint_offload_dir", None)
        if not offload_root:
            return
        log_path = os.path.abspath(os.path.expanduser(self.cfg.runner.logger.log_path))
        rel_path = os.path.relpath(
            os.path.abspath(local_dir), os.path.dirname(log_path)
        )
        dest_dir = os.path.join(offload_root, rel_path)
        if os.path.exists(dest_dir):
            logger.warning(
                f"Offload destination {dest_dir} already exists; keeping local copy at {local_dir}"
            )
            return
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        logger.info(f"Offloading checkpoint: {local_dir} -> {dest_dir}")
        try:
            # shutil.move copies first and removes the source only after the
            # copy has completed, so an interrupted offload leaves the local
            # checkpoint intact.
            shutil.move(local_dir, dest_dir)
        except OSError as e:
            logger.warning(
                f"Failed to offload checkpoint to {dest_dir}: {e}. "
                f"Local copy kept at {local_dir}"
            )

    def set_max_steps(self) -> None:
        self.num_steps_per_epoch = self.actor.get_max_steps_per_epoch().wait()[0]
        max_epochs = self.cfg.runner.get("max_epochs", -1)
        max_steps = self.cfg.runner.get("max_steps", -1)

        step_limits = []
        if max_epochs > 0:
            step_limits.append(self.num_steps_per_epoch * max_epochs)
        if max_steps >= 0:
            step_limits.append(max_steps)

        # If both limits are configured, stop at whichever one is reached first.
        # If neither is configured, keep the historical default of one epoch.
        self.max_steps = min(step_limits) if step_limits else self.num_steps_per_epoch

    @property
    def epoch(self) -> int:
        return self.global_step // self.num_steps_per_epoch
