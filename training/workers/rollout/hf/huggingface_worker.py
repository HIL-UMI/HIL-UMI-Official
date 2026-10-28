# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from training.envs.action_utils import (
    compute_chunk_steps,
    resolve_action_execution_horizon,
    slice_action_chunk,
)
from training.envs.realworld.time_aligned_action_buffer import (
    resolve_time_aligned_action_buffer_config,
)
from training.models import get_model
from training.models.embodiment.base_policy import BasePolicy
from training.scheduler import Channel, Worker, split_channel_message


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        if str(cfg.runner.get("task_type", "embodied_eval")) != "embodied_eval":
            raise ValueError(
                "MultiStepRolloutWorker supports only embodied_eval; policy updates use SFT."
            )
        self.cfg = cfg
        self.only_eval = True
        self.model_cfg = cfg.rollout.model
        self.device = self.torch_platform.current_device()
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = cfg.rollout.get("enable_offload", False)
        self.eval_rollout_epoch = cfg.env.eval.get("rollout_epoch", 1)
        self.eval_batch_size = cfg.env.eval.total_num_envs // self.num_pipeline_stages
        self.per_node_eval_batch_size = self.eval_batch_size // self._world_size
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        self.eval_action_execution_horizon = resolve_action_execution_horizon(
            cfg.env.eval, int(self.model_cfg.num_action_chunks)
        )
        self.truncate_eval_action_chunk = self.eval_action_execution_horizon < int(
            self.model_cfg.num_action_chunks
        )
        self.n_eval_chunk_steps = compute_chunk_steps(
            cfg.env.eval.max_steps_per_rollout_epoch, self.eval_action_execution_horizon
        )
        self.eval_async_action_buffer_cfg = resolve_time_aligned_action_buffer_config(
            cfg.env.eval, int(self.model_cfg.num_action_chunks)
        )
        self.env_decoupled_mode = cfg.runner.get("enable_decoupled_mode", False)
        if self.env_decoupled_mode:
            self.batch_router = {"rollout_results": []}
        self.rollout_queue_size = cfg.rollout.get("rollout_queue_size", 0)

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.model_cfg)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=0,
                eval_batch_size=self.per_node_eval_batch_size,
            )

        if self.enable_offload:
            self.offload_model()

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode != "eval":
            raise ValueError("Only evaluation sampling is supported")
        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs, mode="eval"
            )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        return actions, result

    def _predict_rollout_actions(
        self, env_obs: dict[str, Any], mode: Literal["eval"] = "eval", **_kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.predict(env_obs, mode=mode)

    async def recv_from_and_record_batch_routes_with_timeout(
        self,
        group_name: str,
        channel: Any | None,
        *,
        route_key: Any = None,
        tag: str | None = None,
        batch_size: int | None = None,
        merge_fn: Optional[Callable[[list[Any]], Any]] = None,
        infer_batch_size_fn: Optional[Callable[[Any], int]] = None,
        timeout_time: float = 0.02,
        recv_queue_size: int = 0,
    ):
        """Receive routed batch shards and record their return routes.

        This method is used in env-decoupled mode. It builds a receive plan for the
        source worker group, receives shard messages from ``channel`` one by one, and
        stops when all planned items are received or ``timeout_time`` is reached.

        Each received channel item must be a dict with ``batch_index`` (route
        metadata) and ``batch`` (payload shard).

        The ``batch_index`` values are stored in ``self.batch_router[tag]`` so a
        later send call can split the response and send each shard back to the original
        source rank. The received payload shards are validated, then merged with
        ``merge_fn`` or the default ``merge_batches``.

        Args:
            group_name: Source worker group name.
            channel: Channel used to receive routed batch shards.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag used to build receive keys and index recorded routes.
            batch_size: Expected batch size for each planned receive entry.
            merge_fn: Optional custom function for merging received shards.
            infer_batch_size_fn: Optional function used to infer shard batch size during
                validation.
            timeout_time: Maximum time in seconds to wait before finalizing partial
                results.
            recv_queue_size: Number of receive queue entries used when building the
                receive plan.

        Returns:

            A merged payload and its split sizes. If only one shard is received, the
            current implementation returns that shard directly.
        """
        from training.scheduler import (
            decoupled_build_recv_plan,
            get_batch_size,
            get_group_world_size,
            merge_batches,
        )

        world_size = get_group_world_size(self._manager_proxy, group_name)
        plan = decoupled_build_recv_plan(
            src_group_name=group_name,
            dst_group_name=self.worker_address.root_group_name,
            recv_rank=None,
            src_world_size=self._world_size,
            dst_world_size=world_size,
            tag=tag,
            route_key=route_key,
            batch_size=batch_size,
            recv_queue_size=recv_queue_size,
        )

        def _finalize(received_items: list[Any]):
            if not received_items:
                assert False, "received_items is empty"

            # get the tag from the received_items
            _, _, _, tag = split_channel_message(received_items[0]["batch_index"])

            assert tag in self.batch_router, (
                f"{tag=} need to be already in the batch_router"
            )
            # Save the batch_index to the batch_router.
            list_received_items = []
            for item in received_items:
                batch_index = item["batch_index"]
                received_item = item["batch"]
                list_received_items.append(received_item)
                # Save the batch_index to the batch_router.
                self.batch_router[tag].append(batch_index)
            received_items = list_received_items
            split_sizes = [
                get_batch_size(item, infer_batch_size_fn) for item in received_items
            ]

            if merge_fn is not None:
                return merge_fn(received_items), split_sizes
            if len(received_items) == 1:
                return received_items[0]
            return merge_batches(received_items), split_sizes

        timeout_time = timeout_time + time.time()
        get_items = None
        max_item_num = len(plan.entries)
        get_item_num = 0
        received_items = []
        while get_item_num < max_item_num:
            # get the items
            if get_items is None:
                get_items = channel.get(
                    key=plan.entries[get_item_num].key, async_op=True
                )
            else:
                # Now, the worker is getting a item, sleep to wait
                await asyncio.sleep(0.0001)

            # handle the get_items finish
            if get_items.done():
                # save the data and init the get_items to get next data
                received_items.append(await get_items.async_wait())
                get_items = None
                get_item_num = get_item_num + 1

            # handle the timeout case
            if time.time() >= timeout_time:
                max_item_num = get_item_num
                if get_items is not None:
                    received_items.append(await get_items.async_wait())
                    get_items = None
                    get_item_num = get_item_num + 1

        return _finalize(received_items)

    def send_to_recorded_batch_routes(
        self,
        group_name: str,
        channel: Any | None,
        data: Any,
        *,
        route_key: Any = None,
        tag: str | None = None,
        split_fn: Optional[Callable[[Any, list[int]], list[Any]]] = None,
        split_sizes: list[int],
    ):
        """Send split batch results back using recorded batch routes.

        This method is used after a previous receive call has populated
        ``self.batch_router[tag]`` with batch indices from incoming messages.
        The outgoing ``data`` is split according to ``split_sizes`` and each shard is
        sent to the rank encoded in the corresponding recorded batch index.

        Each outgoing channel item is a dict with ``batch_index`` (recorded batch
        index) and ``batch`` (payload shard).

        After all shards are queued, the recorded batch indices for ``tag`` are cleared
        to avoid reusing stale routes.

        Args:
            group_name: Destination worker group name.
            channel: Channel used to send the split payloads.
            data: Payload to split and send.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag whose recorded batch indices should be consumed.
            split_fn: Optional custom splitter. If omitted, ``split_batch`` is used.
            split_sizes: Batch sizes used to split ``data``. Must have the same length
                as ``self.batch_router[tag]``.

        Returns:

            AsyncRouteWork wrapping the async channel put operations.
        """
        from training.scheduler import build_send_key, split_batch
        from training.scheduler.collective import AsyncRouteWork

        assert tag in self.batch_router, (
            f"{tag=} need to be already in the batch_router"
        )

        assert len(self.batch_router[tag]) > 0, f"{self.batch_router[tag]=} is empty"
        assert len(self.batch_router[tag]) == len(split_sizes), (
            f"{self.batch_router[tag]=} length should equal {split_sizes=} length"
        )

        payloads = (
            split_fn(data, split_sizes)
            if split_fn is not None
            else split_batch(data, split_sizes)
        )

        works = []
        for i, payload in enumerate(payloads):
            batch_index = self.batch_router[tag][i]
            send_rank, _, mode, _ = split_channel_message(batch_index)
            # After enabling env_decoupled_mode, the data sending format is as follows:
            # {
            #     "batch_index": batch_index,
            #     "batch": batch,
            # }
            # The batch_index is the index of the batch in the data.
            # The batch is the data to send.
            # batch_index: {send_rank}_{batch_idx}_{mode}_{tag}
            # The send_rank is the rank of the worker that originally sent the data.
            # The batch_idx is the index of the batch in the data.
            # The tag is the tag of the data.
            senditem = {
                "batch_index": batch_index,
                "batch": payload,
            }
            key = build_send_key(
                src_group_name=self.worker_address.root_group_name,
                dst_group_name=group_name,
                src_rank=None,
                dst_rank=send_rank,
                tag=tag if mode is None else f"{mode}_{tag}",
                route_key=route_key,
            )
            work = channel.put(
                item=senditem,
                key=key,
                async_op=True,
            )
            works.append(work)

        # clear the batch_router for the next send
        self.batch_router[tag] = []
        return AsyncRouteWork(works, lambda _: None)

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        if self.eval_async_action_buffer_cfg.enabled:
            await self._evaluate_with_time_aligned_action_buffer(
                input_channel,
                output_channel,
            )
            if self.enable_offload:
                self.offload_model()
            return
        if self.env_decoupled_mode:
            while True:
                (
                    env_output,
                    split_sizes,
                ) = await self.recv_from_and_record_batch_routes_with_timeout(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="rollout_results",
                    batch_size=self.eval_batch_size,
                    merge_fn=self._merge_obs_batches,
                    infer_batch_size_fn=self._infer_env_batch_size,
                    timeout_time=0.02,
                    recv_queue_size=self.rollout_queue_size,
                )
                actions, _ = self._predict_rollout_actions(
                    env_output["obs"],
                    mode="eval",
                    final_obs=env_output.get("final_obs", None),
                )
                if self.truncate_eval_action_chunk:
                    actions = slice_action_chunk(
                        actions,
                        self.eval_action_execution_horizon,
                    )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().contiguous()
                self.send_to_recorded_batch_routes(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=actions,
                    tag="rollout_results",
                    split_sizes=split_sizes,
                )
        else:
            for _ in tqdm(
                range(self.eval_rollout_epoch),
                desc="Evaluating Rollout Epochs",
                disable=(self._rank != 0),
            ):
                for _ in range(self.n_eval_chunk_steps):
                    for stage_id in range(self.num_pipeline_stages):
                        env_output = await self.recv_from(
                            group_name=self.cfg.env.group_name,
                            channel=input_channel,
                            tag="eval_rollout_results",
                            route_key=stage_id,
                            async_op=True,
                            batch_size=self.eval_batch_size,
                            merge_fn=self._merge_obs_batches,
                            infer_batch_size_fn=self._infer_env_batch_size,
                        ).async_wait()
                        actions, _ = self._predict_rollout_actions(
                            env_output["obs"],
                            mode="eval",
                            final_obs=env_output.get("final_obs", None),
                        )
                        if self.truncate_eval_action_chunk:
                            actions = slice_action_chunk(
                                actions,
                                self.eval_action_execution_horizon,
                            )
                        if isinstance(actions, torch.Tensor):
                            actions = actions.detach().cpu().contiguous()
                        self.send_to(
                            group_name=self.cfg.env.group_name,
                            channel=output_channel,
                            data=actions,
                            tag="eval_rollout_results",
                            route_key=stage_id,
                            async_op=True,
                            batch_size=self.eval_batch_size,
                        )

            if self.enable_offload:
                self.offload_model()

    async def _evaluate_with_time_aligned_action_buffer(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ) -> None:
        """Serve dynamically timed prediction requests from a real-world env."""
        if self.env_decoupled_mode:
            raise ValueError(
                "async_action_buffer cannot be combined with enable_decoupled_mode."
            )
        while True:
            env_output = await self.recv_from(
                group_name=self.cfg.env.group_name,
                channel=input_channel,
                tag="eval_rollout_results",
                route_key=0,
                async_op=True,
                batch_size=self.eval_batch_size,
            ).async_wait()
            request = env_output.get("async_action_buffer_request", {})
            request_kind = request.get("kind")
            if request_kind == "stop":
                return
            if request_kind != "predict":
                raise ValueError(
                    "Async action-buffer request must have kind='predict' or "
                    f"kind='stop', got {request_kind!r}."
                )

            inference_start = time.perf_counter()
            actions, _ = self._predict_rollout_actions(
                env_output["obs"],
                mode="eval",
                final_obs=env_output.get("final_obs", None),
            )
            inference_seconds = time.perf_counter() - inference_start
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().contiguous()
            response = {
                "actions": actions,
                "async_action_buffer_response": {
                    "request_id": int(request["request_id"]),
                    "observation_step": int(request["observation_step"]),
                    "inference_seconds": float(inference_seconds),
                },
            }
            self.send_to(
                group_name=self.cfg.env.group_name,
                channel=output_channel,
                data=response,
                tag="eval_rollout_results",
                route_key=0,
                async_op=True,
            )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    def _merge_obs_batches(self, obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {
            "obs": merged_obs,
            "final_obs": merged_final_obs,
        }

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=0, eval_batch_size=self.per_node_eval_batch_size
            )
