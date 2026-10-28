# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Training-identical ADVANTAGE stage-advantage inference adapter."""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any

# ADVANTAGE uses JAX only for parts of its transform tree. Keep it away from the
# PyTorch inference GPU. Set these before importing hil_umi_advantage modules.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch

from ood.config_paths import resolve_integration_path
from ood.model_environment import require_model_namespace

logger = logging.getLogger(__name__)

EXPECTED_IMAGE_KEYS = (
    "base_-100_rgb",
    "left_wrist_-100_rgb",
    "base_0_rgb",
    "left_wrist_0_rgb",
)


def resolve_advantage_checkpoint(checkpoint_root: str | Path) -> Path:
    """Resolve a direct checkpoint or the highest numeric checkpoint child."""
    root = resolve_integration_path(checkpoint_root)
    if (root / "model.safetensors").is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"Advantage checkpoint directory not found: {root}")
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and child.name.isdigit()
        and (child / "model.safetensors").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No Advantage model.safetensors found in the checkpoint directory "
            f"or a numeric child: {root}"
        )
    return max(candidates, key=lambda path: int(path.name))


class AdvantageModelClient:
    """Load the ADVANTAGE stage-advantage model and score one two-frame pair."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.cfg = dict(model_cfg)
        self._device: torch.device | None = None
        self._model: Any | None = None
        self._train_config: Any | None = None
        self._transform: Any | None = None
        self._model_lib: Any | None = None
        self._preprocessing: Any | None = None
        self._jax: Any | None = None
        self._contract_checked = False
        self._checkpoint_dir: Path | None = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Advantage model has not been loaded")
        return self._device

    @property
    def checkpoint_dir(self) -> Path:
        if self._checkpoint_dir is None:
            raise RuntimeError("Advantage model has not been loaded")
        return self._checkpoint_dir

    @staticmethod
    def _with_task_prompt(train_config: Any, prompt: Any) -> Any:
        """Set the deployment prompt, including configs without a default."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Advantage prompt must be a non-empty string")
        data = train_config.data
        if not dataclasses.is_dataclass(data) or not any(
            field.name == "default_prompt" for field in dataclasses.fields(data)
        ):
            raise ValueError(
                f"ADVANTAGE config {train_config.name!r} has no configurable default prompt"
            )
        return dataclasses.replace(
            train_config,
            data=dataclasses.replace(data, default_prompt=prompt),
        )

    def load(self) -> None:
        """Build the exact training transforms and strictly load the checkpoint."""
        required = ("config_name", "ckpt_dir", "device", "prompt")
        missing = [key for key in required if self.cfg.get(key) in (None, "")]
        if missing:
            raise ValueError(
                "Missing Advantage model config fields: " + ", ".join(missing)
            )

        require_model_namespace("advantage")
        checkpoint_dir = resolve_advantage_checkpoint(self.cfg["ckpt_dir"])
        requested_device = torch.device(str(self.cfg["device"]))
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Advantage model requires {requested_device}, but CUDA is not "
                "visible to this Python process"
            )
        # Lazy imports keep policy mode independent of the Advantage dependency tree.
        import hil_umi_advantage.transforms as transforms
        import jax
        import safetensors.torch
        from hil_umi_advantage.models import model as model_lib
        from hil_umi_advantage.models_pytorch import preprocessing_pytorch
        from hil_umi_advantage.models_pytorch.pi0_pytorch import AdvantageEstimator
        from hil_umi_advantage.training import config as config_lib

        train_config = config_lib.get_config(str(self.cfg["config_name"]))
        if not bool(getattr(train_config, "advantage_estimator", False)):
            raise ValueError(
                f"ADVANTAGE config {train_config.name!r} is not an advantage estimator"
            )
        train_config = self._with_task_prompt(train_config, self.cfg["prompt"])
        data_config = train_config.data.create(
            train_config.assets_dirs, train_config.model
        )
        norm_stats = {} if train_config.skip_norm_stats else data_config.norm_stats
        if norm_stats is None:
            raise ValueError("Advantage normalization statistics are unavailable")
        transform = transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.model_transforms.inputs,
            ]
        )

        logger.info(
            "Loading ADVANTAGE stage-advantage checkpoint %s with config %s",
            checkpoint_dir,
            train_config.name,
        )
        model = AdvantageEstimator(train_config.model).to(requested_device)
        model.eval()
        safetensors.torch.load_model(
            model, checkpoint_dir / "model.safetensors", strict=True
        )

        self._device = requested_device
        self._model = model
        self._train_config = train_config
        self._transform = transform
        self._model_lib = model_lib
        self._preprocessing = preprocessing_pytorch
        self._jax = jax
        self._checkpoint_dir = checkpoint_dir
        logger.info("ADVANTAGE stage-advantage model loaded on %s", requested_device)

    @staticmethod
    def _validate_image(name: str, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [H,W,3], got {value.shape}")
        if value.dtype != np.uint8:
            raise ValueError(f"{name} must be RGB uint8, got {value.dtype}")
        return value

    def _build_observation(
        self,
        *,
        current_front: np.ndarray,
        current_wrist: np.ndarray,
        reference_front: np.ndarray,
        reference_wrist: np.ndarray,
        frame_index: int = 0,
        episode_index: int = 0,
    ) -> Any:
        if self._transform is None or self._train_config is None:
            raise RuntimeError("Advantage model not loaded; call load() first")
        raw = {
            "extra_view_image": self._validate_image("current_front", current_front),
            "image": self._validate_image("current_wrist", current_wrist),
            "his_-100_extra_view_image": self._validate_image(
                "reference_front", reference_front
            ),
            "his_-100_image": self._validate_image("reference_wrist", reference_wrist),
            # UMIInputs masks this state to zero; the model transform pads it to 32.
            "state": np.zeros(10, dtype=np.float32),
            "actions": np.zeros(
                (self._train_config.model.action_horizon, 10), dtype=np.float32
            ),
            "episode_length": max(1, int(frame_index) + 1),
            "frame_index": int(frame_index),
            "episode_index": int(episode_index),
            "progress": np.float32(0.0),
        }
        transformed = self._transform(raw)
        batch = self._jax.tree.map(
            lambda value: torch.as_tensor(np.stack([np.asarray(value)], axis=0)),
            transformed,
        )
        observation = self._model_lib.Observation.from_dict(batch)
        if not self._contract_checked:
            self._assert_contract(observation)
            self._contract_checked = True
        return self._jax.tree.map(lambda value: value.to(self.device), observation)

    def _assert_contract(self, observation: Any) -> None:
        processed = self._preprocessing.preprocess_observation_pytorch_custom(
            observation,
            train=False,
            return_full_obs=False,
            apply_aug=False,
        )
        actual_keys = tuple(processed.images)
        if actual_keys != EXPECTED_IMAGE_KEYS:
            raise RuntimeError(
                f"Unexpected Advantage image order {actual_keys}; "
                f"expected {EXPECTED_IMAGE_KEYS}"
            )
        if any(
            tuple(image.shape) != (1, 3, 224, 224)
            for image in processed.images.values()
        ):
            raise RuntimeError("Advantage images must preprocess to [1,3,224,224]")
        if any(not bool(mask.all()) for mask in processed.image_masks.values()):
            raise RuntimeError("All four Advantage image masks must be true")
        if tuple(processed.state.shape) != (1, 32) or not bool(
            torch.all(processed.state == 0)
        ):
            raise RuntimeError("Advantage state must be zero with shape [1,32]")
        if tuple(processed.tokenized_prompt.shape) != (1, 200):
            raise RuntimeError("Advantage prompt tokens must have shape [1,200]")

    def predict(
        self,
        *,
        current_front: np.ndarray,
        current_wrist: np.ndarray,
        reference_front: np.ndarray,
        reference_wrist: np.ndarray,
        frame_index: int = 0,
        episode_index: int = 0,
    ) -> tuple[float, float]:
        """Return ``(advantage, model_latency_ms)`` for one temporal pair."""
        if self._model is None:
            raise RuntimeError("Advantage model not loaded; call load() first")
        observation = self._build_observation(
            current_front=current_front,
            current_wrist=current_wrist,
            reference_front=reference_front,
            reference_wrist=reference_wrist,
            frame_index=frame_index,
            episode_index=episode_index,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            values = self._model.sample_values(self.device, observation)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if tuple(values.shape) != (1, 1):
            raise RuntimeError(
                f"Advantage model returned {tuple(values.shape)}, expected (1, 1)"
            )
        score = float(values[0, 0].float().cpu())
        if not np.isfinite(score):
            raise RuntimeError(f"Advantage model returned non-finite score: {score}")
        return score, latency_ms

    def warmup(
        self,
        *,
        runs: int,
        front_shape: tuple[int, int, int],
        wrist_shape: tuple[int, int, int],
    ) -> list[float]:
        """Warm the production input shape before attaching recorder resources."""
        if runs < 0:
            raise ValueError("model.warmup_runs must be >= 0")
        front = np.zeros(front_shape, dtype=np.uint8)
        wrist = np.zeros(wrist_shape, dtype=np.uint8)
        elapsed: list[float] = []
        for _ in range(runs):
            _score, latency_ms = self.predict(
                current_front=front,
                current_wrist=wrist,
                reference_front=front,
                reference_wrist=wrist,
            )
            elapsed.append(latency_ms)
        return elapsed
