# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Load and call the pi0.5 evaluation model for OOD sampling."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from ood.config_paths import resolve_integration_path
from ood.model_environment import require_model_namespace

logger = logging.getLogger(__name__)


def _to_container(cfg: Any) -> Any:
    """Convert OmegaConf/DictConfig to a plain dict when needed."""
    if hasattr(cfg, "_content") or str(type(cfg)).endswith("DictConfig'>"):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


class Pi05ModelClient:
    """Load a pi0.5 evaluation model and sample K actions."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.cfg = _to_container(model_cfg)
        self._model: Any | None = None
        self._device: torch.device | None = None
        self._last_inference_timings_ms: dict[str, float] = {}
        self._use_cuda_graph = bool(self.cfg.get("use_cuda_graph", True))

    def load(self) -> None:
        """Load the model via ``training.models.embodiment.openpi_pytorch.get_model``."""
        require_model_namespace("pi05")
        # Import lazily from the separately installed training component.
        from training.models import get_model

        cfg = OmegaConf.create(self.cfg)
        required = (
            "model_type",
            "model_path",
            "precision",
            "is_lora",
            "num_steps",
            "num_action_chunks",
            "action_dim",
            "openpi",
        )
        missing = [key for key in required if key not in cfg]
        if missing:
            raise ValueError(f"Missing model config fields: {', '.join(missing)}")

        checkpoint = resolve_integration_path(str(cfg.model_path))
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Pi0.5 checkpoint weights not found: {weights}")

        requested_device = torch.device(str(cfg.get("device", "cuda")))
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"OOD model requires {requested_device}, but CUDA is not visible to "
                "this Python process (torch.cuda.is_available() is False). Check "
                "the NVIDIA driver/container GPU passthrough before starting the monitor."
            )

        cfg.model_path = str(checkpoint)
        logger.info("Loading pi0.5 model from %s", checkpoint)
        self._model = get_model(cfg)
        if self._model is None:
            raise ValueError(f"Unsupported policy model_type: {cfg.model_type!s}")
        if not callable(getattr(self._model, "predict_action_batch_samples", None)):
            raise RuntimeError(
                "The installed training component does not provide "
                "predict_action_batch_samples(); update the training component to the shared-prefix "
                "Pi0.5 sampler before starting the OOD monitor."
            )
        # Standalone ``training.models.get_model`` has no Worker device context,
        # so it returns the model on CPU. Move it explicitly for OOD inference.
        self._model = self._model.to(requested_device)
        self._device = next(self._model.parameters()).device
        if self._device.type != requested_device.type:
            raise RuntimeError(
                f"Model ended up on {self._device}, requested {requested_device}"
            )
        logger.info("Model loaded on %s", self._device)
        logger.info("Pi0.5 CUDA Graph denoise: %s", self._use_cuda_graph)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Model not loaded")
        return self._device

    @property
    def action_chunk(self) -> int:
        return int(self.cfg.get("num_action_chunks", 10))

    @property
    def action_dim(self) -> int:
        return int(self.cfg.get("action_dim", 30))

    @property
    def last_inference_timings_ms(self) -> dict[str, float]:
        return dict(self._last_inference_timings_ms)

    def predict_action_batch_k(
        self,
        env_obs: dict[str, Any],
        k: int = 10,
        rng: torch.Generator | None = None,
        *,
        profile: bool = True,
    ) -> np.ndarray:
        """Sample K action chunks for the given observation.

        Args:
            env_obs: dict with keys ``states`` ([B,D] tensor), ``main_images``
                ([B,H,W,C] tensor), ``task_descriptions`` (list[str]), and
                optional extra camera keys.
            k: number of stochastic samples.
            rng: optional torch Generator.

        Returns:
            actions: numpy array of shape [K, action_chunk, action_dim] in the
                environment frame. For ``pi05_singlefranka_tcp_rot6d``, the
                eval transform has already recovered absolute TCP actions;
                OODMonitor converts them back to body-frame relative actions
                before comparison.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first")

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        def _validate_single_batch(value: Any) -> None:
            if isinstance(value, np.ndarray):
                if value.ndim == 0 or value.shape[0] != 1:
                    raise ValueError(
                        f"Expected numpy input with batch size 1, got {value.shape}"
                    )
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.shape[0] != 1:
                    raise ValueError(
                        f"Expected tensor input with batch size 1, got {tuple(value.shape)}"
                    )
            elif isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(
                        f"Expected list input with batch size 1, got {len(value)}"
                    )
            elif isinstance(value, tuple):
                if len(value) != 1:
                    raise ValueError(
                        f"Expected tuple input with batch size 1, got {len(value)}"
                    )

        for value in env_obs.values():
            _validate_single_batch(value)

        started = time.perf_counter()
        actions, result = self._model.predict_action_batch_samples(
            env_obs,
            num_samples=k,
            rng=rng,
            profile=profile,
            use_cuda_graph=self._use_cuda_graph,
        )
        actions_np = actions.detach().cpu().numpy()
        expected = (1, k, self.action_chunk, self.action_dim)
        if actions_np.shape != expected:
            raise RuntimeError(
                f"Model returned actions with shape {actions_np.shape}; expected {expected}"
            )
        timings = dict((result or {}).get("timings_ms") or {})
        timings["client_total_ms"] = (time.perf_counter() - started) * 1000.0
        self._last_inference_timings_ms = timings
        return actions_np[0]

    def warmup(
        self,
        env_obs: dict[str, Any],
        *,
        k: int,
        runs: int = 1,
        seed: int = 0,
    ) -> list[float]:
        """Warm fixed-shape kernels without advancing the production RNG."""
        if runs < 0:
            raise ValueError(f"runs must be >= 0, got {runs}")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        elapsed_ms: list[float] = []
        for _ in range(runs):
            started = time.perf_counter()
            self.predict_action_batch_k(
                env_obs,
                k=k,
                rng=generator,
                profile=False,
            )
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        self._last_inference_timings_ms = {}
        return elapsed_ms
