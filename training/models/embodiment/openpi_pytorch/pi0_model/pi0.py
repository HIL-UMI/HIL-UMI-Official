# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Pi0 model for PyTorch, aligned with JAX models/pi0.py.

Flow matching model for continuous action generation.
"""

from __future__ import annotations

import logging
import threading
import time

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gemma, model, pointnet, siglip
from .pi0_config import Pi0Config
from .utils import _str_to_dtype

logger = logging.getLogger("openpi")


def _repeat_suffix_observation_batch(
    observation: model.Observation, repeats: int
) -> model.Observation:
    """Repeat only tensors consumed by :meth:`embed_suffix`.

    Images, prompt tokens, and point clouds are prefix-only. Keeping those at
    their original ``B`` batch avoids materializing another ``K`` copies of
    the already preprocessed images after their prefix cache has been built.
    """

    def repeat(value: torch.Tensor | None) -> torch.Tensor | None:
        return value.repeat_interleave(repeats, dim=0) if value is not None else None

    return model.Observation(
        images=observation.images,
        image_masks=observation.image_masks,
        state=repeat(observation.state),
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        pcd_xyz=observation.pcd_xyz,
    )


def _repeat_kv_cache_batch(kv_cache: tuple, repeats: int) -> tuple:
    """Repeat every per-layer ``(key, value)`` cache along its batch axis."""
    return tuple(
        (
            key.repeat_interleave(repeats, dim=0),
            value.repeat_interleave(repeats, dim=0),
        )
        for key, value in kv_cache
    )


class _DenoiseCudaGraphRunner:
    """Fixed-shape CUDA Graph for the repeated Euler suffix passes.

    Prefix construction remains eager because its values change with every
    camera frame. The graph owns stable buffers for the expanded prefix cache,
    state, mask, and noise; callers copy current values in before replay.
    """

    def __init__(
        self,
        pi0: Pi0,
        observation: model.Observation,
        noise: torch.Tensor,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        num_steps: int,
    ) -> None:
        self._lock = threading.Lock()
        self._observation = _repeat_suffix_observation_batch(observation, 1)
        self._noise = noise.clone()
        self._prefix_mask = prefix_mask.clone()
        self._kv_cache = tuple((key.clone(), value.clone()) for key, value in kv_cache)

        # Warm lazy torch.compile kernels before capture. This runs only during
        # the monitor's synthetic startup warmup, never on the live path.
        with torch.no_grad():
            for _ in range(2):
                pi0._sample_actions_with_prefix_cache(
                    self._observation,
                    self._noise,
                    self._prefix_mask,
                    self._kv_cache,
                    num_steps=num_steps,
                )
        torch.cuda.synchronize(noise.device)

        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self._graph):
            self._output = pi0._sample_actions_with_prefix_cache(
                self._observation,
                self._noise,
                self._prefix_mask,
                self._kv_cache,
                num_steps=num_steps,
            )
        torch.cuda.synchronize(noise.device)

    def replay(
        self,
        observation: model.Observation,
        noise: torch.Tensor,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
    ) -> torch.Tensor:
        """Copy current inputs, replay, and return an independently owned result."""
        with self._lock:
            self._observation.state.copy_(observation.state)
            self._noise.copy_(noise)
            self._prefix_mask.copy_(prefix_mask)
            for (static_key, static_value), (key, value) in zip(
                self._kv_cache, kv_cache, strict=True
            ):
                static_key.copy_(key)
                static_value.copy_(value)
            self._graph.replay()
            # The graph output is overwritten on its next replay. The action
            # tensor is tiny relative to the model, so clone it to preserve the
            # ordinary sampler ownership contract.
            return self._output.clone()


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Create attention mask from input mask and autoregressive mask.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.

    Args:
        input_mask: bool[B, N] - true if token is valid
        mask_ar: bool[N] - true where next token starts a new autoregressive block
    """
    mask_ar = mask_ar.expand(input_mask.shape[0], -1)
    cumsum = torch.cumsum(mask_ar.int(), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(attn_mask, valid_mask)


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions.

    Args:
        pos: (B,) float positions
        embedding_dim: output dimension (must be even)

    Returns:
        (B, embedding_dim) positional embedding
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = torch.linspace(
        0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=torch.float32
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum("i,j->ij", pos.float(), 1.0 / period * 2 * torch.pi)
    # Match JAX which keeps posemb in float32. However, PT Linear does not support
    # mixed float32/bf16 matmul, so cast back to the model's embed_dtype.
    # The caller should upcast to float32 if needed for high-precision ops.
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(
        pos.dtype
    )


class Pi0(model.BaseModel):
    """Pi0 flow matching model for continuous action generation."""

    def __init__(self, config: Pi0Config):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.pcd = config.pcd
        self.embed_dtype = _str_to_dtype(config.dtype)
        self._config = config
        self._denoise_cuda_graphs: dict[tuple, _DenoiseCudaGraphRunner] = {}

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)

        # Gemma LLM with dual experts
        # Expert 0 (PaliGemma) uses regular RMSNorm; Expert 1 (Action Expert) may use adaRMS
        adarms = [False, config.pi05]
        self.llm = gemma.Module(
            configs=[paligemma_config, action_expert_config],
            embed_dtype=config.dtype,
            adarms=adarms,
            use_gradient_checkpointing=False,
        )

        # SigLIP vision encoder
        self.img = siglip.SigLIPViT(
            variant="So400m/14",
            pool_type="none",
            num_classes=paligemma_config.width,
            use_gradient_checkpointing=False,
            dtype_mm=config.dtype,
        )

        action_expert_width = action_expert_config.width
        self.action_dim = config.action_dim

        # Action input projection
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)

        if config.pi05:
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_width)
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_width, action_expert_width
            )
            self.action_time_mlp_out = nn.Linear(
                action_expert_width, action_expert_width
            )

        # Action output projection
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)

        # Optional PointNet
        if config.pcd:
            pointnet_config = pointnet.get_config(config.pointnet_variant)
            self.pointnet = pointnet.UncoloredPointNet(
                n_coordinates=pointnet_config.n_coordinates,
                output_dim=pointnet_config.output_dim,
                hidden_dim=pointnet_config.hidden_dim,
                hidden_depth=pointnet_config.hidden_depth,
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.normal_(self.action_in_proj.weight, std=0.02)
        nn.init.zeros_(self.action_in_proj.bias)
        nn.init.normal_(self.action_out_proj.weight, std=0.02)
        nn.init.zeros_(self.action_out_proj.bias)

        if self.pi05:
            nn.init.normal_(self.time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_in.bias)
            nn.init.normal_(self.time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_out.bias)
        else:
            nn.init.normal_(self.state_proj.weight, std=0.02)
            nn.init.zeros_(self.state_proj.bias)
            nn.init.normal_(self.action_time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_in.bias)
            nn.init.normal_(self.action_time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_out.bias)

    def embed_prefix(
        self,
        obs: model.Observation,
        *,
        image_keys: tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the prefix (images + language + optional point cloud).

        Returns:
            tokens: (B, S, emb_dim) embedded tokens
            input_mask: (B, S) mask of valid tokens
            ar_mask: (S,) autoregressive mask (all False for prefix)
        """
        tokens = []
        input_mask = []

        # The legacy/training path preserves its per-view SigLIP calls.  The
        # eval sampler can provide the CPU-resolved valid image keys: all-
        # masked padding views are omitted and the real views are encoded in
        # one larger vision batch.  Removing an all-false token block is
        # attention-equivalent because it is neither a valid query nor key.
        selected_image_keys = tuple(obs.images) if image_keys is None else image_keys
        if image_keys is None:
            image_token_groups = [
                self.img(obs.images[name])[0] for name in selected_image_keys
            ]
        elif selected_image_keys:
            image_batch_size = obs.state.shape[0]
            batched_images = torch.cat(
                [obs.images[name] for name in selected_image_keys], dim=0
            )
            batched_image_tokens, _ = self.img(batched_images)
            image_token_groups = batched_image_tokens.split(image_batch_size, dim=0)
        else:
            image_token_groups = []

        for name, image_tokens in zip(
            selected_image_keys, image_token_groups, strict=True
        ):
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]
                )
            )

        # Add language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm.embed(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)

        # Add point cloud tokens
        if self.pcd and obs.pcd_xyz is not None:
            # pcd_xyz: (B, 16, 2025, 3)
            # PointNet expects (B, num_points, 3)
            B = obs.pcd_xyz.shape[0]
            pcd_flat = obs.pcd_xyz.reshape(B, -1, 3)  # (B, 16*2025, 3)
            pcd_tokens = self.pointnet(pcd_flat)  # (B, 16, 2048)
            # Reshape to match expected dimensions
            if pcd_tokens.dim() == 2:
                pcd_tokens = pcd_tokens.unsqueeze(1)  # (B, 1, 2048)

            tokens.append(pcd_tokens)
            input_mask.append(
                torch.ones(
                    pcd_tokens.shape[:2], dtype=torch.bool, device=pcd_tokens.device
                )
            )
        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)
        return tokens, input_mask, ar_mask

    def embed_suffix(
        self,
        obs: model.Observation,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed the suffix (state + noisy actions + time embedding).

        Args:
            obs: observation
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) float timestep values

        Returns:
            tokens: (B, S, emb_dim)
            input_mask: (B, S)
            ar_mask: (S,)
            adarms_cond: (B, emb_dim) or None
        """
        input_mask = []
        tokens = []

        B = noisy_actions.shape[0]

        # Cast to embed_dtype to ensure consistency between training and inference
        # (during training FSDP2 casts forward inputs, but inference may pass float32).
        noisy_actions = noisy_actions.to(self.embed_dtype)
        timestep = timestep.to(self.embed_dtype)

        if not self.pi05:
            # Add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(
                torch.ones(B, 1, dtype=torch.bool, device=state_token.device)
            )

        # Embed actions
        action_tokens = self.action_in_proj(noisy_actions)

        # Time embedding
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )

        if self.pi05:
            # Time MLP for adaRMS conditioning
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # Mix timestep + action through MLP
            time_tokens = einops.repeat(
                time_emb, "b emb -> b s emb", s=self.action_horizon
            )
            action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = F.silu(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(
            torch.ones(
                action_expert_tokens.shape[:2],
                dtype=torch.bool,
                device=action_expert_tokens.device,
            )
        )

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)

        # Build ar_mask with correct length matching input_mask.shape[1]
        autoregressive_prefix_len = 1 if self.pi05 else 2
        ar_mask = (
            torch.arange(input_mask.shape[1], device=tokens.device)
            < autoregressive_prefix_len
        )

        return tokens, input_mask, ar_mask, adarms_cond

    def compute_loss(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Returns:
            loss: (B, action_horizon) per-timestep MSE loss
        """
        B = actions.shape[0]
        device = actions.device

        # Preprocess first (requries float32 for image ops),
        # then cast to model dtype for FSDP2 mixed precision compatibility.
        observation = model.preprocess_observation(observation, train=train, rng=rng)

        embed_dtype = self.embed_dtype
        observation = model._observation_to_dtype(observation, embed_dtype)
        actions = actions.to(dtype=embed_dtype)
        dtype = actions.dtype

        # Sample noise and time (or use provided values for reproducibility)
        if noise is None:
            noise = torch.randn(
                actions.shape, device=device, dtype=dtype, generator=rng
            )
        else:
            noise = noise.to(dtype=dtype)
        if time is None:
            time = (
                torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                .sample((B,))
                .to(device=device, dtype=dtype)
            )
            time = time * 0.999 + 0.001
        else:
            time = time.to(dtype=dtype)
        time_expanded = time[:, None, None]

        # Flow matching interpolation
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # One forward pass for prefix + suffix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1

        prefix_out, suffix_out = self.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=attn_mask,
            adarms_cond=[None, adarms_cond],
        )[0]

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return torch.mean(torch.square(v_t - u_t), dim=-1)

    def build_prefix_cache(
        self,
        observation: model.Observation,
        *,
        image_keys: tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Embed prefix tokens and run one LLM pass to build the KV cache.

        The caller is responsible for preprocessing the observation (image
        resize/pad, mask defaults) — this method only consumes the prepared
        observation so it can be shared between the eval Euler sampler and the
        RL train-time forward where the observation has already been built.

        Returns:
            prefix_out:  (B, prefix_len, paligemma_width) paligemma-side hidden states.
            prefix_mask: (B, prefix_len) bool mask of valid prefix positions.
            kv_cache:    per-layer KV cache to feed into subsequent suffix passes.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, image_keys=image_keys
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        outputs, kv_cache = self.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        return outputs[0], prefix_mask, kv_cache

    def run_suffix(
        self,
        observation: model.Observation,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        kv_cache: tuple,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One suffix forward pass (action expert) given the prefix KV cache.

        Returns the action-expert hidden states sliced to the last
        ``action_horizon`` positions: (B, action_horizon, action_expert_width).
        """
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, t_tensor
        )
        suffix_len = suffix_tokens.shape[1]
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        full_attn_mask = torch.cat([prefix_to_suffix_mask, suffix_attn_mask], dim=-1)
        suffix_positions = (
            torch.sum(prefix_mask, dim=-1)[:, None]
            + torch.cumsum(suffix_mask.int(), dim=-1)
            - 1
        )
        outputs, _ = self.llm(
            [None, suffix_tokens],
            positions=suffix_positions,
            mask=full_attn_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return outputs[1][:, -self.action_horizon :]

    def velocity_from_suffix(self, suffix_out_act: torch.Tensor) -> torch.Tensor:
        """Project action-expert hidden states to a velocity prediction v_t."""
        return self.action_out_proj(suffix_out_act)

    def _sample_actions_with_prefix_cache(
        self,
        observation: model.Observation,
        noise: torch.Tensor,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        """Run Euler denoising for an already encoded observation prefix."""
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        dt = -1.0 / num_steps
        batch_size = noise.shape[0]
        device = noise.device
        x_t = noise
        t = 1.0

        while t >= -dt / 2:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            suffix_out_act = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_out_act)
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t

    def _sample_actions_with_cuda_graph(
        self,
        observation: model.Observation,
        noise: torch.Tensor,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        """Replay or capture a denoise graph for the current fixed shapes."""
        if noise.device.type != "cuda":
            raise ValueError("CUDA Graph denoising requires a CUDA device")
        key = (
            noise.device.index,
            num_steps,
            tuple(noise.shape),
            noise.dtype,
            tuple(prefix_mask.shape),
            tuple(
                (
                    tuple(cache_key.shape),
                    cache_key.dtype,
                    tuple(cache_value.shape),
                    cache_value.dtype,
                )
                for cache_key, cache_value in kv_cache
            ),
        )
        runner = self._denoise_cuda_graphs.get(key)
        if runner is None:
            runner = _DenoiseCudaGraphRunner(
                self,
                observation,
                noise,
                prefix_mask,
                kv_cache,
                num_steps=num_steps,
            )
            self._denoise_cuda_graphs[key] = runner
        return runner.replay(observation, noise, prefix_mask, kv_cache)

    def sample_actions(
        self,
        observation: model.Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions using Euler ODE solver.

        Args:
            observation: input observation
            num_steps: number of ODE solver steps
            noise: optional initial noise of shape (B, action_horizon, action_dim)
            rng: random generator

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        observation = model.preprocess_observation(observation, train=False)
        B = observation.state.shape[0]
        device = observation.state.device

        if noise is None:
            noise = torch.randn(
                B, self.action_horizon, self.action_dim, device=device, generator=rng
            )

        _, prefix_mask, kv_cache = self.build_prefix_cache(observation)
        return self._sample_actions_with_prefix_cache(
            observation,
            noise,
            prefix_mask,
            kv_cache,
            num_steps=num_steps,
        )

    def sample_action_samples(
        self,
        observation: model.Observation,
        *,
        num_samples: int,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        profile: bool = False,
        image_keys: tuple[str, ...] | None = None,
        use_cuda_graph: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Sample K actions while encoding the shared observation prefix once.

        Args:
            observation: input observation with base batch size ``B``.
            num_samples: stochastic samples per observation, ``K``.
            num_steps: number of Euler denoising steps.
            noise: optional explicit noise shaped ``(B, K, H, A)``.
            rng: random generator used when ``noise`` is omitted.
            profile: collect synchronized stage timings for diagnostics.
            image_keys: valid image views resolved before device transfer. If
                provided, all-masked padding views are omitted and remaining
                views share one SigLIP call.
            use_cuda_graph: capture/replay the fixed-shape Euler denoise loop.

        Returns:
            Model-space actions shaped ``(B, K, H, A)`` and stage timings.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        batch_size = observation.state.shape[0]
        device = observation.state.device
        expected_noise_shape = (
            batch_size,
            num_samples,
            self.action_horizon,
            self.action_dim,
        )
        if noise is None:
            noise = torch.randn(
                expected_noise_shape,
                device=device,
                generator=rng,
            )
        elif tuple(noise.shape) != expected_noise_shape:
            raise ValueError(
                f"noise must have shape {expected_noise_shape}, got {tuple(noise.shape)}"
            )
        else:
            noise = noise.to(device=device)

        cuda_profile = profile and device.type == "cuda"
        cuda_marks: dict[str, torch.cuda.Event] = {}
        wall_marks: dict[str, float] = {}

        def mark(name: str) -> None:
            if not profile:
                return
            if cuda_profile:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                cuda_marks[name] = event
            else:
                wall_marks[name] = time.perf_counter()

        mark("start")
        observation = model.preprocess_observation(observation, train=False)
        mark("preprocessed")
        _, prefix_mask, kv_cache = self.build_prefix_cache(
            observation, image_keys=image_keys
        )
        mark("prefix")
        sampled_observation = _repeat_suffix_observation_batch(observation, num_samples)
        sampled_prefix_mask = prefix_mask.repeat_interleave(num_samples, dim=0)
        sampled_kv_cache = _repeat_kv_cache_batch(kv_cache, num_samples)
        sampled_noise = noise.reshape(
            batch_size * num_samples, self.action_horizon, self.action_dim
        )
        mark("expanded")
        if use_cuda_graph:
            sampled_actions = self._sample_actions_with_cuda_graph(
                sampled_observation,
                sampled_noise,
                sampled_prefix_mask,
                sampled_kv_cache,
                num_steps=num_steps,
            )
        else:
            sampled_actions = self._sample_actions_with_prefix_cache(
                sampled_observation,
                sampled_noise,
                sampled_prefix_mask,
                sampled_kv_cache,
                num_steps=num_steps,
            )
        mark("denoised")

        timings: dict[str, float] = {}
        if profile:
            pairs = {
                "model_preprocess_ms": ("start", "preprocessed"),
                "prefix_ms": ("preprocessed", "prefix"),
                "cache_expand_ms": ("prefix", "expanded"),
                "denoise_ms": ("expanded", "denoised"),
            }
            if cuda_profile:
                cuda_marks["denoised"].synchronize()
                timings = {
                    key: cuda_marks[start].elapsed_time(cuda_marks[end])
                    for key, (start, end) in pairs.items()
                }
            else:
                timings = {
                    key: (wall_marks[end] - wall_marks[start]) * 1000.0
                    for key, (start, end) in pairs.items()
                }

        return sampled_actions.reshape(expected_noise_shape), timings

    def forward(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = True,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default forward computes loss."""
        return self.compute_loss(
            observation, actions, train=train, rng=rng, noise=noise, time=time
        )

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None
    ):
        """Enable gradient checkpointing for memory efficiency.

        Args:
            gradient_checkpointing_kwargs: Optional kwargs forwarded to the activation
                checkpoint. Currently honors ``use_reentrant`` (default ``False``), so
                the FSDP ``gradient_checkpointing_use_reentrant`` setting is respected.
        """
        kwargs = gradient_checkpointing_kwargs or {}
        use_reentrant = kwargs.get("use_reentrant", False)
        self.llm.gradient_checkpointing = True
        self.llm.gradient_checkpointing_use_reentrant = use_reentrant
        self.img.encoder.gradient_checkpointing = True
        self.img.encoder.gradient_checkpointing_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (used by the eval / no-recompute path)."""
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False
