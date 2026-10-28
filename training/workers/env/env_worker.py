# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from training.data.embodied_io_struct import (
    EnvOutput,
)
from training.envs import get_env_cls
from training.envs.action_utils import (
    compute_chunk_steps,
    prepare_actions,
    resolve_action_execution_horizon,
    resolve_sync_action_chunk_final_wait_s,
    slice_action_chunk,
)
from training.envs.realworld.time_aligned_action_buffer import (
    TimeAlignedActionBuffer,
    resolve_time_aligned_action_buffer_config,
)
from training.envs.utils import get_env_attr
from training.envs.wrappers import RecordVideo
from training.scheduler import Channel, Cluster, Worker
from training.utils.nested_dict_process import (
    update_nested_cfg,
)
from training.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        if str(cfg.runner.get("task_type", "embodied_eval")) != "embodied_eval":
            raise ValueError(
                "EnvWorker supports only embodied_eval; policy updates use SFT."
            )
        self.cfg = cfg
        self.eval_video_cnt = 0
        self.env_list = []
        self.eval_env_list = []
        self._envs_closed = False
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.only_eval = True
        self.enable_eval = True
        self.model_cfg = cfg.rollout.model
        self.eval_rollout_epoch = cfg.env.eval.get("rollout_epoch", 1)
        self.eval_enable_offload = cfg.env.eval.get("enable_offload", False)
        self.eval_num_envs_per_stage = (
            cfg.env.eval.total_num_envs // self._world_size // self.stage_num
        )
        self.eval_batch_size = cfg.env.eval.total_num_envs // self.stage_num
        self.eval_action_execution_horizon = resolve_action_execution_horizon(
            cfg.env.eval, int(self.model_cfg.num_action_chunks)
        )
        self.truncate_eval_action_chunk = self.eval_action_execution_horizon < int(
            self.model_cfg.num_action_chunks
        )
        self.n_eval_chunk_steps = compute_chunk_steps(
            cfg.env.eval.max_steps_per_rollout_epoch, self.eval_action_execution_horizon
        )
        self.eval_sync_action_chunk_final_wait_s = (
            resolve_sync_action_chunk_final_wait_s(cfg.env.eval)
        )
        self.eval_async_action_buffer_cfg = resolve_time_aligned_action_buffer_config(
            cfg.env.eval, int(self.model_cfg.num_action_chunks)
        )
        self.eval_save_pose_trace = bool(
            OmegaConf.select(
                cfg, "env.eval.override_cfg.save_pose_trace", default=False
            )
        )
        if self.eval_save_pose_trace and not self.eval_async_action_buffer_cfg.enabled:
            raise ValueError(
                "save_pose_trace currently requires async_action_buffer.enabled=true."
            )
        if self.eval_async_action_buffer_cfg.enabled:
            if str(cfg.env.eval.env_type) != "realworld":
                raise ValueError(
                    "async_action_buffer is supported only for realworld evaluation."
                )
            if self.stage_num != 1 or self.eval_batch_size != 1:
                raise ValueError(
                    "async_action_buffer requires one pipeline stage and one real-world evaluation environment."
                )
        self.eval_prev_done = [
            torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
            for _ in range(self.stage_num)
        ]
        self.env_decoupled_mode = cfg.runner.get("enable_decoupled_mode", False)
        if self.env_decoupled_mode:
            self.batch_router = {}
            assert self._component_placement.get_world_size(
                "env"
            ) >= self._component_placement.get_world_size("rollout")

    def init_worker(self):
        self.broadcast(True, groups=[(self._group_name, list(range(self._world_size)))])
        self.update_env_cfg()
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
        self.eval_env_list = self._setup_env_and_wrappers(
            eval_env_cls, self.cfg.env.eval, self.eval_num_envs_per_stage
        )
        if self.eval_enable_offload:
            assert all(
                callable(get_env_attr(env, "offload")) for env in self.eval_env_list
            ), "eval envs must have an offload method to enable offload!"
            for env in self.eval_env_list:
                get_env_attr(env, "offload")()

    def cleanup(self) -> None:
        """Close every environment owned by this worker exactly once."""
        if self._envs_closed:
            return
        self._envs_closed = True

        cleanup_errors: list[Exception] = []
        seen_env_ids: set[int] = set()
        for env in (*self.env_list, *self.eval_env_list):
            if id(env) in seen_env_ids:
                continue
            seen_env_ids.add(id(env))
            try:
                env.close()
            except Exception as exc:  # Continue closing independent environments.
                self.log_warning(f"Failed to close environment: {exc}")
                cleanup_errors.append(exc)
        self.env_list.clear()
        self.eval_env_list.clear()

        if cleanup_errors:
            self.log_error(
                f"Environment cleanup completed with {len(cleanup_errors)} failure(s)"
            )

    def update_env_cfg(self):
        overrides = self.cfg.env.eval.get("override_cfgs", None)
        if overrides is not None:
            assert len(overrides) > self._rank
            base = OmegaConf.to_container(
                self.cfg.env.eval.get("override_cfg", {}), resolve=True
            )
            local = OmegaConf.to_container(overrides[self._rank], resolve=True)
            self.cfg.env.eval.override_cfg = OmegaConf.create(
                update_nested_cfg(base, local)
            )

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []
        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            env_list.append(env)
        return env_list

    def env_evaluate_step(
        self,
        raw_actions: torch.Tensor,
        stage_id: int,
        *,
        truncate_action_chunk: bool = True,
        sync_action_chunk_final_wait_s: float = 0.0,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        if truncate_action_chunk and self.truncate_eval_action_chunk:
            raw_actions = slice_action_chunk(
                raw_actions,
                self.eval_action_execution_horizon,
            )
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            env_cfg=self.cfg.env.eval,
        )
        env_info = {}

        chunk_step_kwargs = {}
        if sync_action_chunk_final_wait_s > 0.0:
            if str(self.cfg.env.eval.env_type) != "realworld":
                raise ValueError(
                    "sync_action_chunk_final_wait_s is supported only for "
                    "realworld evaluation."
                )
            chunk_step_kwargs["sync_action_chunk_final_wait_s"] = (
                sync_action_chunk_final_wait_s
            )
        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(
                chunk_actions,
                **chunk_step_kwargs,
            )
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = infos.get("final_observation") if isinstance(infos, dict) else None

        current_dones = chunk_dones.any(dim=1)  # [num_envs] bool
        if self.cfg.env.eval.auto_reset:
            newly_done = current_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if newly_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            env_infos=infos if isinstance(infos, dict) else None,
        )
        return env_output, env_info

    def finish_rollout(self, mode="eval"):
        if mode != "eval":
            raise ValueError("Only evaluation rollouts are supported")
        for env in self.eval_env_list:
            if self.cfg.env.eval.video_cfg.save_video:
                flush_video = get_env_attr(env, "flush_video")
                if callable(flush_video):
                    flush_video()
            if not self.cfg.env.eval.auto_reset:
                env.update_reset_state_ids()

    def _build_rollout_input_data(self, env_batch: dict[str, Any]) -> dict[str, Any]:
        return {"obs": env_batch["obs"], "final_obs": env_batch["final_obs"]}

    @staticmethod
    def _infer_rollout_batch_size(data: Any) -> int:
        from training.scheduler.worker.routing import infer_batch_size

        return infer_batch_size(data)

    def _send_async_action_buffer_request(
        self,
        *,
        rollout_channel: Channel,
        input_channel: Channel,
        env_output: EnvOutput,
        request_id: int,
        observation_step: int,
    ):
        """Send one stamped observation and start a non-blocking action receive."""
        env_batch = env_output.to_dict()
        request_payload = self._build_rollout_input_data(env_batch)
        request_payload["async_action_buffer_request"] = {
            "kind": "predict",
            "request_id": int(request_id),
            "observation_step": int(observation_step),
        }
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=rollout_channel,
            data=request_payload,
            mode="eval",
            tag="rollout_results",
            route_key=0,
        )
        return self.recv_from(
            group_name=self.cfg.rollout.group_name,
            channel=input_channel,
            tag="eval_rollout_results",
            route_key=0,
            async_op=True,
            batch_size=self.eval_batch_size,
        )

    def _send_async_action_buffer_stop(self, rollout_channel: Channel) -> None:
        """Tell the rollout worker that the dynamic async request stream ended."""
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=rollout_channel,
            data={
                # Routed payload validation requires one batch-shaped field. The
                # rollout worker ignores this marker after the stop message arrives.
                "async_action_buffer_route_batch": np.zeros(
                    (self.eval_batch_size, 1),
                    dtype=np.uint8,
                ),
                "async_action_buffer_request": {"kind": "stop"},
            },
            mode="eval",
            tag="rollout_results",
            route_key=0,
            batch_size=self.eval_batch_size,
        )

    @staticmethod
    def _async_action_response_actions(response: Any) -> np.ndarray:
        if not isinstance(response, dict) or "actions" not in response:
            raise ValueError(
                "Async action-buffer rollout response must contain an actions field."
            )
        actions = response["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        return np.asarray(actions, dtype=np.float32)

    def _evaluate_with_time_aligned_action_buffer(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
    ):
        """Evaluate at the environment rate while inference runs concurrently."""
        if self.env_decoupled_mode:
            raise ValueError(
                "async_action_buffer cannot be combined with enable_decoupled_mode."
            )

        cfg = self.eval_async_action_buffer_cfg
        prediction_horizon = int(self.model_cfg.num_action_chunks)
        action_buffer = TimeAlignedActionBuffer(
            prediction_horizon=prediction_horizon,
            handover_blend_steps=cfg.handover_blend_steps,
            temporal_ensemble_decay=cfg.temporal_ensemble_decay,
            motion_continuity_enabled=cfg.motion_continuity_enabled,
            motion_tracking_gain=cfg.motion_tracking_gain,
            max_translation_velocity_m_per_step=(
                cfg.max_translation_velocity_m_per_step
            ),
            max_translation_acceleration_m_per_step2=(
                cfg.max_translation_acceleration_m_per_step2
            ),
            max_translation_jerk_m_per_step3=(cfg.max_translation_jerk_m_per_step3),
            max_rotation_velocity_rad_per_step=(cfg.max_rotation_velocity_rad_per_step),
            max_rotation_acceleration_rad_per_step2=(
                cfg.max_rotation_acceleration_rad_per_step2
            ),
            max_rotation_jerk_rad_per_step3=cfg.max_rotation_jerk_rad_per_step3,
            translation_correction_activate_m=(cfg.translation_correction_activate_m),
            translation_correction_release_m=(cfg.translation_correction_release_m),
            rotation_correction_activate_rad=cfg.rotation_correction_activate_rad,
            rotation_correction_release_rad=cfg.rotation_correction_release_rad,
        )
        trace_tracker = None
        set_pose_trace_context = None
        if self.eval_save_pose_trace:
            from training.envs.realworld.franka.pose_trace import (
                AsyncPoseTraceTracker,
            )

            trace_tracker = AsyncPoseTraceTracker(prediction_horizon)
            set_pose_trace_context = get_env_attr(
                self.eval_env_list[0],
                "set_async_pose_trace_context",
            )
            if not callable(set_pose_trace_context):
                raise ValueError(
                    "save_pose_trace is not supported by the configured "
                    "real-world environment."
                )
        eval_metrics = defaultdict(list)
        control_step = 0
        request_id = 0
        last_request_step = 0
        pending_work = None
        pending_request_id = None
        pending_observation_step = None
        env_output = None
        exhaustion_warning_active = False

        def install_response(response: Any) -> bool:
            nonlocal pending_observation_step, pending_request_id, pending_work
            if not isinstance(response, dict):
                raise ValueError(
                    "Async action-buffer rollout response must be a dictionary."
                )
            metadata = response.get("async_action_buffer_response", {})
            response_request_id = int(metadata.get("request_id", -1))
            observation_step = int(metadata.get("observation_step", -1))
            if response_request_id < 0 or observation_step < 0:
                raise ValueError(
                    "Async action-buffer response is missing request timestamps."
                )
            if (
                response_request_id != pending_request_id
                or observation_step != pending_observation_step
            ):
                raise ValueError(
                    "Async action-buffer response does not match the outstanding "
                    f"request: expected request={pending_request_id}, "
                    f"observation_step={pending_observation_step}; got "
                    f"request={response_request_id}, observation_step="
                    f"{observation_step}."
                )
            response_actions = self._async_action_response_actions(response)
            result = action_buffer.install(
                response_actions,
                observation_step=observation_step,
                ready_step=control_step,
            )
            inference_seconds = float(metadata.get("inference_seconds", float("nan")))
            if result.accepted:
                if trace_tracker is not None:
                    trace_tracker.install(
                        response_actions,
                        request_id=response_request_id,
                        observation_step=observation_step,
                        ready_step=control_step,
                        inference_seconds=inference_seconds,
                    )
                self.log_info(
                    "AsyncActionBuffer installed "
                    f"request={response_request_id} observation_step="
                    f"{observation_step} ready_step={control_step} "
                    f"aligned_index={result.aligned_index} "
                    f"inference_seconds={inference_seconds:.4f}"
                )
            else:
                self.log_warning(
                    "AsyncActionBuffer dropped stale prediction "
                    f"request={response_request_id} age_steps="
                    f"{result.response_age_steps}: {result.reason}"
                )
            pending_work = None
            pending_request_id = None
            pending_observation_step = None
            return result.accepted

        def request_prediction(*, wait: bool) -> bool:
            nonlocal last_request_step, pending_observation_step, pending_request_id
            nonlocal pending_work, request_id
            if env_output is None:
                raise RuntimeError("Cannot request actions without an observation.")
            if pending_work is not None:
                raise RuntimeError("An asynchronous prediction is already pending.")
            pending_request_id = request_id
            pending_observation_step = control_step
            pending_work = self._send_async_action_buffer_request(
                rollout_channel=rollout_channel,
                input_channel=input_channel,
                env_output=env_output,
                request_id=request_id,
                observation_step=control_step,
            )
            last_request_step = control_step
            request_id += 1
            if not wait:
                return True
            return install_response(pending_work.wait())

        try:
            for eval_rollout_epoch in range(self.eval_rollout_epoch):
                if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                    if pending_work is not None:
                        pending_work.wait()
                        pending_work = None
                        pending_request_id = None
                        pending_observation_step = None
                    action_buffer.clear()
                    if trace_tracker is not None:
                        trace_tracker.clear()
                    self.eval_env_list[0].is_start = True
                    self.eval_prev_done[0] = torch.zeros(
                        self.eval_num_envs_per_stage,
                        dtype=torch.bool,
                    )
                    extracted_obs, infos = self.eval_env_list[0].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                        env_infos=infos if isinstance(infos, dict) else None,
                    )
                    request_prediction(wait=True)

                for _ in range(self.cfg.env.eval.max_steps_per_rollout_epoch):
                    if pending_work is not None and pending_work.done():
                        accepted = install_response(pending_work.wait())
                        if not accepted:
                            request_prediction(wait=False)

                    selection = action_buffer.select(control_step)
                    if trace_tracker is not None:
                        trace_context = trace_tracker.select(
                            control_step,
                            blended=selection.blended,
                        )
                        set_pose_trace_context(trace_context)
                    if selection.exhausted and not exhaustion_warning_active:
                        self.log_warning(
                            "AsyncActionBuffer exhausted; holding its last action "
                            f"at control_step={control_step}."
                        )
                        exhaustion_warning_active = True
                    elif not selection.exhausted:
                        exhaustion_warning_active = False
                    raw_action = selection.action[:, None, :]
                    env_output, env_info = self.env_evaluate_step(
                        raw_action,
                        stage_id=0,
                        truncate_action_chunk=False,
                    )
                    for key, value in env_info.items():
                        eval_metrics[key].append(value)
                    control_step += 1

                    done = bool(env_output.dones.any().item())
                    if done:
                        if pending_work is not None:
                            pending_work.wait()
                            pending_work = None
                            pending_request_id = None
                            pending_observation_step = None
                        action_buffer.clear()
                        if trace_tracker is not None:
                            trace_tracker.clear()
                        if self.cfg.env.eval.auto_reset:
                            request_prediction(wait=True)
                        else:
                            break
                        continue

                    should_replan = (
                        control_step - last_request_step >= cfg.replan_interval_steps
                    )
                    if pending_work is None and should_replan:
                        request_prediction(wait=False)

                self.finish_rollout(mode="eval")
        finally:
            if pending_work is not None:
                pending_work.wait()
            self._send_async_action_buffer_stop(rollout_channel)

        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                get_env_attr(self.eval_env_list[stage_id], "offload")()
        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()
        return eval_metrics

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        if self.eval_async_action_buffer_cfg.enabled:
            return self._evaluate_with_time_aligned_action_buffer(
                input_channel,
                rollout_channel,
            )
        eval_metrics = defaultdict(list)
        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                        env_infos=infos if isinstance(infos, dict) else None,
                    )
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=self._build_rollout_input_data(env_batch),
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    rollout_results = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.eval_batch_size,
                        infer_batch_size_fn=self._infer_rollout_batch_size
                        if self.env_decoupled_mode
                        else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    raw_chunk_actions = (
                        rollout_results.actions
                        if hasattr(rollout_results, "actions")
                        else rollout_results
                    )
                    if isinstance(raw_chunk_actions, torch.Tensor):
                        raw_chunk_actions = raw_chunk_actions.detach().cpu().numpy()
                    else:
                        raw_chunk_actions = np.asarray(raw_chunk_actions)
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions,
                        stage_id,
                        sync_action_chunk_final_wait_s=(
                            self.eval_sync_action_chunk_final_wait_s
                        ),
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch == self.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=self._build_rollout_input_data(env_batch),
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                get_env_attr(self.eval_env_list[stage_id], "offload")()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
