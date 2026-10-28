# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils._pytree import tree_map

from training.models.embodiment.openpi_pytorch.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from training.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from training.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


def _to_numpy(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x


class OpenPiPytorchEvalActionModel(OpenPiPytorchActionModel):
    """Eval-capable wrapper around the vendored ``Pi0`` model.

    Drives observation construction through the upstream ``openpi.transforms``
    pipeline: the factory calls :meth:`setup_wrappers` with the composed
    input/output transform lists from :func:`get_openpi_config`, then
    :meth:`predict_action_batch` routes ``env_obs`` through
    :meth:`_repack_env_obs` → :meth:`input_transform` →
    :meth:`_observation_dict_to_device` → :meth:`Pi0.sample_actions` →
    :meth:`output_transform`.

    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )
        # ``action_chunk`` slices the env-action subspace from the model output
        # in :meth:`output_transform`.
        self.action_chunk = action_chunk
        # ``config_name`` is the openpi TrainConfig key (e.g. ``pi05_behavior``)
        # used by :meth:`_repack_env_obs` to switch the env→``observation/*``
        # repack rules (currently only the ``calvin`` split-state layout).
        self.config_name = config_name
        # Optional subset of the raw env state dim (openpi ``state_indices``).
        # ``None`` (the BEHAVIOR default) is an identity passthrough.
        self.state_indices = list(state_indices) if state_indices else None

        # openpi.transforms pipeline state (installed by :meth:`setup_wrappers`).
        self._input_transform_fn = None
        self._output_transform_fn = None

    # -------------------------------------------------------- transforms glue

    def setup_wrappers(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the openpi.transforms input/output pipelines.

        ``transforms`` is the list passed to the openpi-side ``compose`` —
        typically ``BehaviorInputs → Normalize(norm_stats) → ModelTransformFactory(model_config)``
        (the last stage carries the auto-downloading PaliGemma tokenizer).
        ``output_transforms`` is the matching reverse pipeline used to turn
        sampled model actions back into env-frame actions.
        """
        from openpi.transforms import compose

        self._input_transform_fn = compose(transforms)
        self._output_transform_fn = compose(output_transforms)

    def _ensure_wrappers(self) -> None:
        if self._input_transform_fn is None or self._output_transform_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}.setup_wrappers(...) must be called "
                "after construction (the factory does this); the openpi "
                "transforms pipeline is not yet installed."
            )

    def _select_configured_state(self, states):
        """Select a configured subset of the raw env state (openpi parity).

        Mirrors ``openpi.openpi_action_model.OpenPi0ForRLActionPrediction.
        _select_configured_state``: ``state_indices=None`` (BEHAVIOR) is an
        identity passthrough; a configured index list fancy-indexes the last
        dim so future non-BEHAVIOR envs can reuse the generic repack.
        """
        indices = self.state_indices
        if not indices:
            return states

        if hasattr(states, "shape"):
            state_dim = states.shape[-1]
        else:
            state_dim = np.asarray(states).shape[-1]
        if state_dim == len(indices):
            return states
        if state_dim <= max(indices):
            raise ValueError(
                f"Cannot select state_indices={indices} from state dim {state_dim}."
            )

        if torch.is_tensor(states):
            index_tensor = torch.as_tensor(indices, device=states.device)
            return states.index_select(-1, index_tensor)
        return np.asarray(states)[..., indices]

    def _repack_env_obs(self, env_obs: dict) -> dict:
        """Map the env's observation dict to the ``observation/*`` keys the
        openpi pipeline expects.

        Generic adapter mirroring
        ``openpi.openpi_action_model.OpenPi0ForRLActionPrediction.obs_processor``:
        it reads the standardized RLinf env keys (``states`` / ``main_images`` /
        ``task_descriptions`` and the optional ``wrist_images`` /
        ``extra_view_images``) and repacks them into openpi's ``observation/*``
        convention. All real per-env logic lives in the config-driven
        ``data_transforms.inputs`` (selected by ``config_name``), not here; the
        only per-env branch is the ``calvin`` split-state layout. Optional
        camera views are added only when the env actually provides them
        (BEHAVIOR, for instance, emits no ``extra_view_images`` key).
        """
        env_states = self._select_configured_state(env_obs["states"])
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self.config_name:
            processed_obs["observation/state_ee_pos"] = env_states[:, :3]
            processed_obs["observation/state_ee_rot"] = env_states[:, 3:6]
            processed_obs["observation/state_gripper"] = env_states[:, 6:7]
        else:
            processed_obs["observation/state"] = env_states
        wrist_images = env_obs.get("wrist_images")
        if wrist_images is not None:
            processed_obs["observation/wrist_image"] = wrist_images
        extra_view_images = env_obs.get("extra_view_images")
        if extra_view_images is not None:
            processed_obs["observation/extra_view_image"] = extra_view_images
        return processed_obs

    def input_transform(self, obs: dict, transpose: bool = False) -> dict:
        """Apply the openpi input pipeline per-sample then recombine into a batched dict.

        Two modes:

        * Rollout (``"prompt"`` key present) — runs the full pipeline including
          prompt tokenization; result has ``image``/``image_mask``/``state``/
          ``tokenized_prompt``/``tokenized_prompt_mask`` keys.
        * Train recompute (no ``"prompt"``; only ``observation/*`` and the cached
          ``tokenized_prompt`` keys present) — re-runs the pipeline using the
          cached tokens rather than re-tokenising every micro-batch.
        """
        self._ensure_wrappers()
        inputs = tree_map(lambda x: x, obs)
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {k: inputs[k] for k in inputs.keys() if "/" in k}

        inputs = tree_map(_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))

        batch_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: x[i], inputs)
            if transpose:
                sample = tree_map(
                    lambda x: (
                        x.transpose(1, 2, 0)
                        if isinstance(x, np.ndarray) and x.ndim == 3
                        else x
                    ),
                    sample,
                )
            if first_process:
                prompts = obs["prompt"]
                if isinstance(prompts, np.ndarray):
                    prompts = prompts.tolist()
                sample["prompt"] = prompts[i]
            else:
                # Pipeline still runs Tokenize, but the cached tokens below
                # overwrite its output — placeholder text is fine.
                sample["prompt"] = "xxxx"
            batch_samples.append(sample)

        with ThreadPoolExecutor(max_workers=min(len(batch_samples), 8)) as ex:
            transformed = list(ex.map(self._input_transform_fn, batch_samples))

        recombined = tree_map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
            *transformed,
        )
        if not first_process:
            recombined["tokenized_prompt"] = obs["tokenized_prompt"]
            recombined["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return recombined

    def output_transform(self, outputs: dict) -> dict:
        """Apply the openpi output pipeline per-sample then recombine."""
        self._ensure_wrappers()
        batch_size = outputs["actions"].shape[0]
        transformed = []
        for i in range(batch_size):
            sample = tree_map(
                lambda x: _to_numpy(x[i]) if torch.is_tensor(x) else x[i],
                outputs,
            )
            sample = self._output_transform_fn(sample)
            transformed.append(sample)
        recombined = tree_map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
            *transformed,
        )
        if self.action_chunk is not None:
            recombined["actions"] = recombined["actions"][:, : self.action_chunk]
        return recombined

    def _observation_dict_to_device(self, processed: dict) -> Observation:
        """Build a device observation, normalizing uint8 images on-device.

        Moving the compact uint8 tensors before :meth:`Observation.from_dict`
        avoids CPU float conversion and a 4x larger host-to-device transfer.
        """
        device = self.device

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        def _move_state(x):
            # The openpi Normalize stage runs in float64 (its norm_stats arrays
            # are float64); cast state back to float32 to match the legacy eval
            # processor (which did a final ``.float()``) and the model's compute
            # dtype. For pi05 the continuous state is unused (only the discrete
            # state tokens in the prompt are), but pi0 feeds it through
            # ``state_proj`` so float32 keeps the linear layer's dtype aligned.
            return (
                x.to(device=device, dtype=torch.float32)
                if isinstance(x, torch.Tensor)
                else x
            )

        device_processed = dict(processed)
        device_processed["image"] = {
            key: _move(value) for key, value in processed["image"].items()
        }
        device_processed["image_mask"] = {
            key: _move(value) for key, value in processed["image_mask"].items()
        }
        device_processed["state"] = _move_state(processed["state"])
        for key in (
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
            "pcd_xyz",
        ):
            if key in processed:
                device_processed[key] = _move(processed[key])
        return Observation.from_dict(device_processed)

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions via the deterministic Euler ODE sampler.

        Only ``mode="eval"`` is supported at the eval level — that path is
        shared by every transforms-pipeline variant (eval task + RL eval
        fallback). ``mode="train"`` is implemented in the RL subclass which
        overrides this method to add the chain-collecting SDE sampler.
        Calling with ``mode="train"`` on the eval class raises
        :class:`NotImplementedError` so an eval-only model loudly refuses to
        be used for on-policy rollouts.
        """
        del compute_values, kwargs  # accepted for call-site parity; eval ignores them
        if mode != "eval":
            raise NotImplementedError(
                f"{type(self).__name__} only supports predict_action_batch(mode='eval'); "
            )
        # Shared openpi.transforms pipeline.
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)
        return self._predict_eval(observation, noise=noise, rng=rng)

    @torch.no_grad()
    def predict_action_batch_samples(
        self,
        env_obs: dict[str, Any],
        *,
        num_samples: int,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        profile: bool = False,
        use_cuda_graph: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample K action chunks while sharing one observation prefix.

        Returns environment-frame actions shaped ``(B, K, T, D)``.  The
        optional profile records one synchronized GPU timeline; it is intended
        for online diagnostics and does not alter sampling.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        total_start = time.perf_counter()
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        input_cpu_ms = (time.perf_counter() - total_start) * 1000.0
        active_image_keys = tuple(
            key
            for key, mask in processed["image_mask"].items()
            if bool(np.asarray(_to_numpy(mask)).any())
        )

        input_start_event = None
        input_end_event = None
        input_gpu_start = time.perf_counter()
        if profile and self.device.type == "cuda":
            input_start_event = torch.cuda.Event(enable_timing=True)
            input_end_event = torch.cuda.Event(enable_timing=True)
            input_start_event.record()
        observation = self._observation_dict_to_device(processed)
        if input_end_event is not None:
            input_end_event.record()
        input_gpu_wall_ms = (time.perf_counter() - input_gpu_start) * 1000.0

        model_actions, core_timings = self.model.sample_action_samples(
            observation,
            num_samples=num_samples,
            num_steps=self.num_steps,
            noise=noise,
            rng=rng,
            profile=profile,
            image_keys=active_image_keys,
            use_cuda_graph=use_cuda_graph,
        )

        batch_size = observation.state.shape[0]
        output_start = time.perf_counter()
        flat_model_actions = model_actions.reshape(
            batch_size * num_samples,
            model_actions.shape[-2],
            model_actions.shape[-1],
        )
        flat_state = observation.state.repeat_interleave(num_samples, dim=0)
        env_outputs = self.output_transform(
            {"actions": flat_model_actions, "state": flat_state}
        )
        flat_actions = env_outputs["actions"].to(
            device=self.device, dtype=torch.float32
        )
        actions = flat_actions.reshape(
            batch_size,
            num_samples,
            flat_actions.shape[-2],
            flat_actions.shape[-1],
        )
        if profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        output_ms = (time.perf_counter() - output_start) * 1000.0

        timings_ms: dict[str, float] = {}
        if profile:
            if input_start_event is not None and input_end_event is not None:
                input_gpu_ms = input_start_event.elapsed_time(input_end_event)
            else:
                input_gpu_ms = input_gpu_wall_ms
            timings_ms = {
                "input_cpu_ms": input_cpu_ms,
                "input_gpu_ms": input_gpu_ms,
                **core_timings,
                "output_ms": output_ms,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
            }

        return actions, {
            "timings_ms": timings_ms,
            "active_image_keys": active_image_keys,
            "cuda_graph": use_cuda_graph,
        }

    def _predict_eval(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deterministic Euler ODE sampler shared by eval and the RL eval path.

        Returns the env-frame ``actions`` plus the rollout-side return dict
        with ``prev_logprobs=None`` / ``prev_values=None`` and a minimal
        ``forward_inputs`` (action + raw model_action) — exactly what
        :class:`huggingface_worker.HuggingFaceWorker.predict` expects from an
        eval call.
        """
        model_actions = self.model.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        env_outputs = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )
        # openpi Unnormalize runs in float64; cast env actions back to float32 to
        # match the legacy eval processor's ``.astype(np.float32)`` contract (and
        # the action dtype the env/rollout worker expects).
        actions = env_outputs["actions"].to(device=self.device, dtype=torch.float32)
        B = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
        }
        return actions, result
