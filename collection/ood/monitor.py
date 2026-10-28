# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""OOD monitor loop for collection teleoperation."""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from ood.config_paths import resolve_integration_path
from ood.distance import (
    GeometricDistanceConfig,
    compute_ood_metrics,
    is_ood,
    load_gripper_scale,
)
from ood.image_reader import CameraConfig, CameraSetReader
from ood.model_client import Pi05ModelClient
from ood.replay import EpisodeReplayRecorder, FrameSnapshot, ReplayCatalog
from ood.server import OODEvent, OODWebServer
from ood.state_receiver import StateReceiver

logger = logging.getLogger(__name__)


def _mat_to_pose9d(mat4x4: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous transform to position(3) + rot6d(6)."""
    pos = mat4x4[:3, 3].astype(np.float32)
    rot6d = np.concatenate(
        [
            mat4x4[:3, 0],
            mat4x4[:3, 1],
        ]
    ).astype(np.float32)
    return np.concatenate([pos, rot6d])


def _pose9d_to_se3(pose9d: np.ndarray) -> np.ndarray:
    """Decode pose9d values into valid homogeneous transforms."""
    pose9d = np.asarray(pose9d, dtype=np.float32)
    if pose9d.shape[-1] != 9:
        raise ValueError(f"expected pose9d trailing dimension 9, got {pose9d.shape}")
    r6 = pose9d[..., 3:9].astype(np.float64)
    r1 = r6[..., :3]
    r2 = r6[..., 3:]
    n1 = np.linalg.norm(r1, axis=-1, keepdims=True)
    if np.any(n1 < 1e-8):
        raise ValueError("degenerate rot6d first column")
    b1 = r1 / n1
    r2_perp = r2 - np.sum(b1 * r2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(r2_perp, axis=-1, keepdims=True)
    if np.any(n2 < 1e-8):
        raise ValueError("degenerate rot6d second column")
    b2 = r2_perp / n2
    b3 = np.cross(b1, b2, axis=-1)
    transform = np.zeros(pose9d.shape[:-1] + (4, 4), dtype=np.float64)
    transform[..., :3, :3] = np.stack([b1, b2, b3], axis=-1)
    transform[..., :3, 3] = pose9d[..., :3]
    transform[..., 3, 3] = 1.0
    return transform


def _se3_to_pose9d(transform: np.ndarray) -> np.ndarray:
    """Encode homogeneous transforms as position + rotation-6d."""
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        [transform[..., :3, 3], transform[..., :3, 0], transform[..., :3, 1]],
        axis=-1,
    ).astype(np.float32)


def _to_relative_action_chunk(
    absolute_actions: np.ndarray, reference_pose9d: np.ndarray
) -> np.ndarray:
    """Convert absolute TCP actions to the trained body-frame representation."""
    absolute_actions = np.asarray(absolute_actions, dtype=np.float32)
    if absolute_actions.shape[-1] != 10:
        raise ValueError(f"expected 10-D actions, got {absolute_actions.shape}")
    reference = _pose9d_to_se3(np.asarray(reference_pose9d, dtype=np.float32))
    absolute = _pose9d_to_se3(absolute_actions[..., :9])
    relative = np.linalg.inv(reference) @ absolute
    relative_pose = _se3_to_pose9d(relative)
    # Gripper is scalar_abs in SINGLE_ARM_ROT6D_LAYOUT.
    return np.concatenate([relative_pose, absolute_actions[..., 9:10]], axis=-1)


def _to_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (np.ndarray,)):
        if value.size == 0:
            return float(default)
        value = value.flat[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _encode_image_to_b64(img: np.ndarray, quality: int = 70) -> str:
    """Encode an RGB uint8 image to base64 JPEG."""
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class OODMonitor:
    """Asynchronous OOD monitor for data_recorder teleoperation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._stop = False
        self._step = 0
        # This flag tracks replay capture only. Inference readiness is independent
        # of the recorder's A/B episode state.
        self._recording_active = False
        self._has_seen_state = False
        self._last_monitor_status: tuple[str, str, str, str] | None = None

        # OOD parameters.
        ood_cfg = cfg["ood"]
        self._k = int(ood_cfg["k"])
        self._threshold_immediate = float(ood_cfg["threshold_immediate"])
        self._threshold_chunk = float(ood_cfg["threshold_chunk"])
        self._interval_s = float(ood_cfg.get("interval_s", 0.5))
        self._send_image = bool(ood_cfg.get("send_image", True))
        self._seed = int(ood_cfg.get("seed", 42))
        self._warmup_runs = int(ood_cfg.get("warmup_runs", 1))
        if self._warmup_runs < 0:
            raise ValueError("ood.warmup_runs must be >= 0")

        geometry_cfg = ood_cfg.get("geometry") or {}
        norm_stats_path = cfg["model"].get("openpi_data", {}).get("norm_stats_path")
        if not norm_stats_path:
            raise ValueError(
                "model.openpi_data.norm_stats_path is required for geometry-aware "
                "OOD action scales"
            )
        position_scale_cfg = geometry_cfg.get("position_scales_m") or {}
        if not isinstance(position_scale_cfg, dict):
            raise ValueError("ood.geometry.position_scales_m must be a mapping")
        position_scales = np.asarray(
            [
                position_scale_cfg.get("s_x", 0.01),
                position_scale_cfg.get("s_y", 0.01),
                position_scale_cfg.get("s_z", 0.01),
            ],
            dtype=np.float64,
        )
        gripper_scale = load_gripper_scale(resolve_integration_path(norm_stats_path))
        component_weights = geometry_cfg.get("component_weights") or {}
        self._distance_config = GeometricDistanceConfig(
            position_scales=position_scales,
            rotation_scale_rad=np.deg2rad(
                float(geometry_cfg.get("rotation_scale_deg", 10.0))
            ),
            gripper_scale=gripper_scale,
            position_weight=float(component_weights.get("position", 1.0 / 3.0)),
            rotation_weight=float(component_weights.get("rotation", 1.0 / 3.0)),
            gripper_weight=float(component_weights.get("gripper", 1.0 / 3.0)),
        )
        time_weighting = str(geometry_cfg.get("time_weighting", "uniform"))
        if time_weighting != "uniform":
            raise ValueError(
                "ood.geometry.time_weighting currently must be 'uniform', "
                f"got {time_weighting!r}"
            )

        # Sources.
        source_cfg = cfg["source"]
        replay_cfg = cfg.get("replay") or {}
        self._socket_path = str(source_cfg["socket_path"])
        self._task_description = str(
            source_cfg.get("task_description", "teleoperation")
        )
        self._arm = str(source_cfg.get("arm", "left")).strip().lower()
        if self._arm not in {"left", "right"}:
            raise ValueError(f"source.arm must be 'left' or 'right', got {self._arm!r}")
        self._camera_connect_timeout_s = float(
            source_cfg.get("camera_connect_timeout_s", 0.0)
        )
        self._camera_stale_reconnect_s = float(
            source_cfg.get("camera_stale_reconnect_s", 1.0)
        )

        camera_cfgs = {
            key: CameraConfig(
                name=c["name"],
                ring_name=c["ring_name"],
                width=int(c["width"]),
                height=int(c["height"]),
                ring_slots=int(c["ring_slots"]),
                stale_reconnect_s=self._camera_stale_reconnect_s,
            )
            for key, c in source_cfg["cameras"].items()
        }
        self._camera_cfgs = camera_cfgs
        self._camera_mapping = source_cfg["model_inputs"]
        self._preview_image_key = str(
            replay_cfg.get(
                "front_camera",
                source_cfg.get(
                    "preview_image",
                    self._camera_mapping.get("extra_view_image", ""),
                ),
            )
        )
        self._preview_wrist_image_key = str(
            replay_cfg.get(
                "wrist_camera",
                self._camera_mapping.get("main_image", "wrist_left"),
            )
        )
        preview_keys = {
            self._preview_image_key,
            self._preview_wrist_image_key,
        }
        missing_preview_keys = sorted(preview_keys - set(camera_cfgs))
        if missing_preview_keys:
            raise ValueError(
                "Live preview camera keys are not configured under source.cameras: "
                + ", ".join(missing_preview_keys)
            )
        if self._preview_image_key == self._preview_wrist_image_key:
            raise ValueError("Live front and wrist preview cameras must be distinct")
        self._camera_reader = CameraSetReader(camera_cfgs)
        self._latest_snapshot: FrameSnapshot | None = None
        self._snapshot_lock = threading.Lock()
        self._camera_run_root: str | None = None

        # Model.
        self._model = Pi05ModelClient(cfg["model"])
        self._rng: torch.Generator | None = None
        self._previous_prediction: dict[str, Any] | None = None
        self._next_prediction: dict[str, Any] | None = None
        self._preview_thread: threading.Thread | None = None
        self._preview_warned_black: set[str] = set()

        # Action-rate contract. Recorder config/recorder.example.yaml emits state packets at
        # 30 Hz; OOD inference runs less often but compares a full-rate window.
        self._num_action_chunks = int(cfg["model"]["num_action_chunks"])
        self._action_dim = int(cfg["model"]["action_dim"])
        if self._action_dim != 10:
            raise ValueError(
                "The single-Franka TCP rot6d projection requires model.action_dim=10, "
                f"got {self._action_dim}"
            )
        self._action_frequency_hz = float(ood_cfg.get("action_frequency_hz", 30.0))
        self._comparison_window_s = float(ood_cfg.get("comparison_window_s", 0.5))
        self._comparison_steps = round(
            self._action_frequency_hz * self._comparison_window_s
        )
        if self._comparison_steps < 1:
            raise ValueError("action_frequency_hz * comparison_window_s must be >= 1")
        if self._comparison_steps > self._num_action_chunks:
            raise ValueError(
                "The comparison window needs more steps than the model action chunk: "
                f"{self._comparison_steps} > {self._num_action_chunks}"
            )

        # Subsystems.
        self._receiver = StateReceiver(self._socket_path)
        output_root = Path(
            str(replay_cfg.get("output_root", "./data_recorder_output"))
        ).expanduser()
        self._replay_catalog = ReplayCatalog([output_root])
        self._replay = EpisodeReplayRecorder(
            fps=float(replay_cfg.get("fps", self._action_frequency_hz)),
            front_key=str(replay_cfg.get("front_camera", self._preview_image_key)),
            wrist_key=str(
                replay_cfg.get(
                    "wrist_camera",
                    self._camera_mapping.get("main_image", "wrist_left"),
                )
            ),
            ffmpeg_bin=str(replay_cfg.get("ffmpeg_bin", "ffmpeg")),
            encoder_preference=str(
                replay_cfg.get("encoder_preference", "nvenc_then_x264")
            ),
            queue_size=int(replay_cfg.get("encoder_queue_size", 30)),
        )
        server_cfg = cfg["server"]
        static_dir = server_cfg.get("static_dir")
        self._server = OODWebServer(
            host=str(server_cfg.get("host", "0.0.0.0")),
            port=int(server_cfg.get("port", 8765)),
            static_dir=static_dir,
            replay_catalog=self._replay_catalog,
        )

        # Timing stats.
        self._inference_ms: deque[float] = deque(maxlen=100)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        logger.info("Received signal %d, shutting down.", signum)
        self._stop = True

    def _setup_signal_handler(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def start(self) -> None:
        """Load the model, connect recorder resources, and start the dashboard."""
        # Fail model/config validation before attaching recorder-owned shared
        # memory. This prevents a bad checkpoint from disturbing live capture.
        self._model.load()
        if self._warmup_runs:
            warmup_ms = self._model.warmup(
                self._build_model_warmup_observation(),
                k=self._k,
                runs=self._warmup_runs,
                seed=self._seed,
            )
            logger.info(
                "Pi0.5 K=%d warmup complete: runs=%d latency_ms=%s",
                self._k,
                self._warmup_runs,
                ",".join(f"{value:.1f}" for value in warmup_ms),
            )
        self._rng = torch.Generator(device=self._model.device)
        self._rng.manual_seed(self._seed)
        self._camera_reader.connect_all(timeout_s=self._camera_connect_timeout_s)
        self._receiver.start()
        self._server.start(daemon=True)
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            name="ood-paired-preview",
            daemon=True,
        )
        self._preview_thread.start()
        logger.info(
            "OOD monitor started: arm=%s action_dim=%d chunk=%d k=%d interval=%.2fs, "
            "compare_steps=%d action_hz=%.1f immediate_energy_threshold=%.2f, "
            "chunk_energy_threshold=%.2f",
            self._arm,
            self._action_dim,
            self._num_action_chunks,
            self._k,
            self._interval_s,
            self._comparison_steps,
            self._action_frequency_hz,
            self._threshold_immediate,
            self._threshold_chunk,
        )
        logger.info(
            "OOD geometry: xyz_scales=%s gripper_scale=%.6f rotation_scale=%.2fdeg "
            "weights=(position=%.3f rotation=%.3f gripper=%.3f) time=uniform",
            np.array2string(
                self._distance_config.position_scales,
                precision=6,
                separator=", ",
            ),
            self._distance_config.gripper_scale,
            np.rad2deg(self._distance_config.rotation_scale_rad),
            self._distance_config.position_weight,
            self._distance_config.rotation_weight,
            self._distance_config.gripper_weight,
        )
        self._publish_monitor_status(
            state="starting",
            mode="idle",
            reason="waiting_for_recorder",
            detail="Waiting for recorder state and camera frames",
        )

    def _publish_monitor_status(
        self,
        *,
        state: str,
        mode: str,
        reason: str = "",
        detail: str = "",
    ) -> None:
        status_key = (str(state), str(mode), str(reason), str(detail))
        self._server.set_monitor_status(
            state=status_key[0],
            mode=status_key[1],
            reason=status_key[2],
            detail=status_key[3],
            updated_at=time.time(),
        )
        if status_key != self._last_monitor_status:
            logger.info(
                "OOD monitor status: state=%s mode=%s reason=%s detail=%s",
                *status_key,
            )
            self._last_monitor_status = status_key

    def _pause_inference(
        self,
        *,
        mode: str,
        reason: str,
        detail: str,
        state: str = "paused",
    ) -> None:
        """Stop scoring and invalidate causal alignment across a bad interval."""
        self._previous_prediction = None
        self._next_prediction = None
        self._publish_monitor_status(
            state=state,
            mode=mode,
            reason=reason,
            detail=detail,
        )

    def _publish_inference_results(
        self,
        *,
        mode: str,
        prediction_event: OODEvent,
        finalized_event: OODEvent | None,
    ) -> None:
        """Atomically expose the result that proves inference is running."""
        detail = (
            "Inference active during episode capture"
            if mode == "recording"
            else "Inference active; idle results are live-only"
        )
        updated_at = time.time()
        self._server.put_inference_results(
            prediction_event=prediction_event,
            finalized_event=finalized_event,
            mode=mode,
            detail=detail,
            updated_at=updated_at,
        )
        status_key = ("running", str(mode), "", detail)
        if status_key != self._last_monitor_status:
            logger.info(
                "OOD monitor status: state=%s mode=%s reason=%s detail=%s",
                *status_key,
            )
            self._last_monitor_status = status_key

    @staticmethod
    def _recorder_gate_status(state: dict[str, Any]) -> tuple[bool, str, str]:
        """Decode the authoritative A-button-equivalent readiness decision."""
        ready = state.get("ood_ready")
        if ready is True:
            return True, "", ""
        if ready is None:
            return (
                False,
                "recorder_patch_outdated",
                "Recorder state packet does not contain ood_ready",
            )
        return (
            False,
            str(state.get("ood_block_reason") or "hardware_not_ready"),
            str(state.get("ood_block_detail") or "Recorder hardware gate is not ready"),
        )

    def _build_model_warmup_observation(self) -> dict[str, Any]:
        """Create one shape-accurate synthetic single-Franka observation."""
        main_key = str(self._camera_mapping["main_image"])
        main_cfg = self._camera_cfgs[main_key]
        observation: dict[str, Any] = {
            "states": np.asarray(
                [[0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            "main_images": np.zeros(
                (1, main_cfg.height, main_cfg.width, 3), dtype=np.uint8
            ),
            "task_descriptions": [self._task_description],
        }
        extra_key = self._camera_mapping.get("extra_view_image")
        if extra_key is not None:
            extra_cfg = self._camera_cfgs[str(extra_key)]
            observation["extra_view_images"] = np.zeros(
                (1, extra_cfg.height, extra_cfg.width, 3), dtype=np.uint8
            )
        return observation

    def stop(self) -> None:
        self._stop = True
        if (
            self._preview_thread is not None
            and self._preview_thread is not threading.current_thread()
            and self._preview_thread.is_alive()
        ):
            self._preview_thread.join(timeout=2.0)
        self._receiver.stop()
        self._replay.close()
        self._camera_reader.close()
        self._server.stop()

    def _publish_preview_pair(
        self,
        images: dict[str, np.ndarray],
        source_frame_ids: dict[str, int],
        captured_at: float,
    ) -> bool:
        """Encode and publish one complete front/wrist preview pair."""
        front = images.get(self._preview_image_key)
        wrist = images.get(self._preview_wrist_image_key)
        if front is None or wrist is None:
            return False

        for key, image in (
            (self._preview_image_key, front),
            (self._preview_wrist_image_key, wrist),
        ):
            if not np.count_nonzero(image):
                if key not in self._preview_warned_black:
                    logger.warning(
                        "Preview frame from %s is all zeros; check the camera "
                        "stream and recorder ring.",
                        key,
                    )
                    self._preview_warned_black.add(key)
            else:
                self._preview_warned_black.discard(key)

        try:
            front_image_b64 = _encode_image_to_b64(front)
            wrist_image_b64 = _encode_image_to_b64(wrist)
            self._server.put_preview(
                image_b64=front_image_b64,
                wrist_image_b64=wrist_image_b64,
                timestamp=captured_at,
                source_frame_ids={
                    "front": source_frame_ids[self._preview_image_key],
                    "wrist_left": source_frame_ids[self._preview_wrist_image_key],
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish paired camera preview: %s", exc)
            return False
        return True

    def _preview_loop(self) -> None:
        """Capture the paired 30 Hz frame source used by preview/model/replay."""
        interval_s = 1.0 / max(self._action_frequency_hz, 1.0)
        next_deadline = time.monotonic()
        while not self._stop:
            frames = self._camera_reader.read_frames()
            images = {key: frame.image for key, frame in frames.items()}
            source_frame_ids = {key: frame.frame_id for key, frame in frames.items()}
            captured_at = time.time()
            has_complete_pair = all(
                key in images
                for key in (
                    self._preview_image_key,
                    self._preview_wrist_image_key,
                )
            )
            if has_complete_pair:
                snapshot = self._replay.record_frame(
                    images,
                    source_frame_ids,
                    captured_at,
                )
                with self._snapshot_lock:
                    self._latest_snapshot = snapshot
                self._publish_preview_pair(
                    images,
                    source_frame_ids,
                    captured_at,
                )
            next_deadline += interval_s
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.monotonic()

    def _get_latest_snapshot(self) -> FrameSnapshot | None:
        with self._snapshot_lock:
            return self._latest_snapshot

    def _handle_lifecycle(self, lifecycle: dict[str, Any]) -> None:
        """Apply authoritative writer lifecycle transitions."""
        state = str(lifecycle.get("state") or "").strip().lower()
        run_root = lifecycle.get("run_root") or lifecycle.get("dataset_root")
        if run_root:
            self._replay_catalog.register_run_root(run_root)
        if state == "recording":
            normalized_run_root = (
                str(Path(run_root).expanduser().resolve()) if run_root else None
            )
            if (
                normalized_run_root is not None
                and self._camera_run_root is not None
                and normalized_run_root != self._camera_run_root
            ):
                self._camera_reader.reconnect_all("new recorder run")
                with self._snapshot_lock:
                    self._latest_snapshot = None
            if normalized_run_root is not None:
                self._camera_run_root = normalized_run_root
            self._recording_active = True
            self._replay.start_episode(lifecycle)
        elif state in {"saving", "discarding", "finalizing", "stopped"}:
            self._recording_active = False
            self._replay.stop_capture()
        elif state in {"discarded", "fatal"}:
            self._recording_active = False
            self._replay.discard(state)
        elif state == "waiting" and lifecycle.get("saved_frames") is not None:
            self._recording_active = False
            manifest = self._replay.finish_saved(lifecycle)
            if manifest is not None:
                self._server.put_episode_ready(manifest)

    def _build_observation(
        self,
        state: dict[str, Any],
        images: dict[str, np.ndarray],
    ) -> tuple[dict[str, Any], np.ndarray] | None:
        """Project one recorder packet into the trained single-Franka contract.

        The offline training converter uses the exact same projection:

        * state: selected arm pose9d + gripper feedback position;
        * action: selected arm pose9d + gripper position command percentage.
        """
        projected = self._project_state_action(state)
        if projected is None:
            return None
        _pose9d, obs_state, action = projected

        # ---- images ----
        main_key = self._camera_mapping.get("main_image")
        extra_key = self._camera_mapping.get("extra_view_image")

        if main_key is None or main_key not in images:
            logger.warning(
                "Main image (%s) not available, skipping OOD step.", main_key
            )
            return None
        if extra_key is not None and extra_key not in images:
            logger.warning(
                "Extra image (%s) not available, skipping OOD step.", extra_key
            )
            return None

        main_image = np.asarray(images[main_key])
        if main_image.ndim != 3 or main_image.shape[-1] != 3:
            logger.warning(
                "Main image has invalid shape %s; skipping OOD step.",
                main_image.shape,
            )
            return None
        # Model preprocessing expects [B, H, W, C] RGB uint8.
        main_image = np.expand_dims(main_image, axis=0)

        obs: dict[str, Any] = {
            "states": np.expand_dims(obs_state, axis=0),
            "main_images": main_image,
            "task_descriptions": [self._task_description],
        }

        if extra_key is not None:
            extra_image = np.asarray(images[extra_key])
            if extra_image.ndim != 3 or extra_image.shape[-1] != 3:
                logger.warning(
                    "Extra image has invalid shape %s; skipping OOD step.",
                    extra_image.shape,
                )
                return None
            obs["extra_view_images"] = np.expand_dims(extra_image, axis=0)

        return obs, action

    def _project_state_action(
        self, state: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return pose9d, observation state10, and absolute action10."""
        pose_value = state.get(f"{self._arm}_pose")
        if pose_value is None:
            logger.warning("%s pose is unavailable; skipping OOD step.", self._arm)
            return None
        pose = np.asarray(pose_value, dtype=np.float32)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            logger.warning(
                "%s pose must be a finite 4x4 matrix, got %s; skipping OOD step.",
                self._arm,
                pose.shape,
            )
            return None
        pose9d = _mat_to_pose9d(pose)

        gripper_state = state.get(f"gripper_state_{self._arm}") or {}
        state_gripper = _to_scalar(gripper_state.get("position"), default=np.nan)
        if not np.isfinite(state_gripper):
            logger.warning(
                "%s gripper feedback position is unavailable; skipping OOD step.",
                self._arm,
            )
            return None

        position_command_key = f"gripper_cmd_{self._arm}_pos_pct"
        trigger_command_key = f"gripper_cmd_{self._arm}"
        if (
            state.get(position_command_key) is None
            and state.get(trigger_command_key) is None
        ):
            logger.warning(
                "%s gripper position command is unavailable; skipping OOD step.",
                self._arm,
            )
            return None
        trigger_command = _to_scalar(state.get(trigger_command_key), default=0.0)
        action_gripper = _to_scalar(
            state.get(position_command_key), default=1.0 - trigger_command
        )
        if not np.isfinite(action_gripper):
            logger.warning(
                "%s gripper position command is non-finite; skipping OOD step.",
                self._arm,
            )
            return None

        obs_state = np.concatenate(
            [pose9d, np.array([state_gripper], dtype=np.float32)]
        ).astype(np.float32)
        action = np.concatenate(
            [pose9d, np.array([action_gripper], dtype=np.float32)]
        ).astype(np.float32)
        if obs_state.shape != (self._action_dim,) or action.shape != (
            self._action_dim,
        ):
            raise RuntimeError(
                "Single-Franka projection shape mismatch: "
                f"state={obs_state.shape} action={action.shape} "
                f"configured_dim={self._action_dim}"
            )
        return pose9d, obs_state, action

    def _build_human_chunk(
        self,
        states: list[dict[str, Any]],
        reference_pose9d: np.ndarray,
    ) -> np.ndarray | None:
        absolute_actions: list[np.ndarray] = []
        for state in states:
            projected = self._project_state_action(state)
            if projected is None:
                return None
            absolute_actions.append(projected[2])
        if not absolute_actions:
            return None
        return _to_relative_action_chunk(
            np.stack(absolute_actions, axis=0), reference_pose9d
        )

    def _evaluate(
        self,
        state: dict[str, Any],
        snapshot: FrameSnapshot,
        human_states: list[dict[str, Any]],
    ) -> tuple[OODEvent | None, OODEvent | None]:
        """Create the current prediction and finalize the preceding window.

        A finalized event always belongs to the preceding prediction anchor:
        the RGB image, video frame index, model chunk, immediate score, and
        delayed human trajectory all share the same ``prediction_id``.
        """
        images = snapshot.images
        built = self._build_observation(state, images)
        if built is None:
            return None, None
        env_obs, _human_action_absolute = built

        # Run K-sample inference.
        t0 = time.perf_counter()
        model_samples_absolute = self._model.predict_action_batch_k(
            env_obs, k=self._k, rng=self._rng
        )  # [K, T, D]
        expected_shape = (self._k, self._num_action_chunks, self._action_dim)
        if model_samples_absolute.shape != expected_shape:
            raise RuntimeError(
                "Model returned absolute actions with shape "
                f"{model_samples_absolute.shape}; expected {expected_shape}"
            )
        model_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._inference_ms.append(model_latency_ms)
        model_breakdown_ms = dict(
            getattr(self._model, "last_inference_timings_ms", {}) or {}
        )
        model_breakdown_ms.setdefault("client_total_ms", model_latency_ms)

        current_projected = self._project_state_action(state)
        if current_projected is None:
            return None, None
        current_pose9d, _obs_state, current_action_absolute = current_projected
        model_samples_current = _to_relative_action_chunk(
            model_samples_absolute, current_pose9d
        )
        current_action_relative = _to_relative_action_chunk(
            current_action_absolute[None, :], current_pose9d
        )[0]

        # Immediate OOD: T=1 geometry-aware Energy Distance.
        immediate_metrics = compute_ood_metrics(
            model_samples_current[:, :1, :],
            current_action_relative[None, :],
            self._distance_config,
        )
        immediate_score = float(immediate_metrics["energy_distance"]["score"])
        immediate_ood = is_ood(
            immediate_metrics,
            threshold=self._threshold_immediate,
        )

        state_ts = float(state.get("sys_ts") or snapshot.captured_at)
        anchor_ts = float(snapshot.captured_at)
        capture_mode = "recording" if state.get("recording") is True else "idle"
        prediction_id = (
            f"{snapshot.run_id or 'live'}:"
            f"{snapshot.episode_index if snapshot.episode_index is not None else -1}:"
            f"{snapshot.playback_frame_index}:{self._step}"
        )
        image_b64 = None
        if self._send_image and self._preview_image_key in images:
            try:
                image_b64 = _encode_image_to_b64(images[self._preview_image_key])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to encode prediction anchor image: %s", exc)

        prediction_evaluated_at = time.time()
        prediction_latency_ms = (prediction_evaluated_at - anchor_ts) * 1000.0

        current_prediction = {
            "step": self._step,
            "prediction_id": prediction_id,
            "state_ts": state_ts,
            "anchor_ts": anchor_ts,
            "reference_pose9d": current_pose9d.copy(),
            "model_samples_absolute": model_samples_absolute.copy(),
            "immediate_score": immediate_score,
            "immediate_ood": immediate_ood,
            "model_latency_ms": model_latency_ms,
            "prediction_latency_ms": prediction_latency_ms,
            "model_breakdown_ms": model_breakdown_ms,
            "image_b64": image_b64,
            "frame_index": snapshot.playback_frame_index,
            "source_frame_ids": dict(snapshot.source_frame_ids),
            "run_id": snapshot.run_id,
            "episode_index": snapshot.episode_index,
            "capture_mode": capture_mode,
        }
        self._next_prediction = current_prediction
        prediction_event = OODEvent(
            step=self._step,
            timestamp=anchor_ts,
            prediction_id=prediction_id,
            anchor_timestamp=anchor_ts,
            state_timestamp=state_ts,
            evaluated_timestamp=prediction_evaluated_at,
            model_latency_ms=model_latency_ms,
            prediction_latency_ms=prediction_latency_ms,
            model_breakdown_ms=model_breakdown_ms,
            frame_index=snapshot.playback_frame_index,
            source_frame_ids=dict(snapshot.source_frame_ids),
            run_id=snapshot.run_id,
            episode_index=snapshot.episode_index,
            capture_mode=capture_mode,
            immediate_score=immediate_score,
            immediate_ood=immediate_ood,
            chunk_score=0.0,
            chunk_ood=False,
            energy_score=0.0,
            finalized=False,
            image_b64=image_b64,
        )

        # Delayed chunk OOD: compare actions observed after the previous model
        # query against the chunk predicted at that previous query time.
        # This preserves causality and aligns model[t-15:t] with human[t-15:t].
        previous = self._previous_prediction
        if previous is None or len(human_states) != self._comparison_steps:
            return prediction_event, None

        # Live scoring is independent of idle/recording transitions. Replay is
        # stricter: only a window wholly contained in one recorded episode may
        # be persisted next to that episode's videos.
        expected_episode = previous.get("episode_index")
        replay_eligible = bool(
            previous.get("capture_mode") == "recording"
            and previous.get("run_id")
            and expected_episode is not None
            and int(previous.get("frame_index", -1)) >= 0
            and all(
                state_item.get("recording") is True
                and state_item.get("episode") is not None
                and int(state_item["episode"]) - 1 == expected_episode
                for state_item in human_states
            )
        )

        reference_pose9d = previous["reference_pose9d"]
        human_chunk = self._build_human_chunk(human_states, reference_pose9d)
        if human_chunk is None:
            return prediction_event, None
        model_chunk = _to_relative_action_chunk(
            previous["model_samples_absolute"][:, : self._comparison_steps, :],
            reference_pose9d,
        )
        chunk_metrics = compute_ood_metrics(
            model_chunk,
            human_chunk,
            self._distance_config,
        )
        chunk_score = float(chunk_metrics["energy_distance"]["score"])
        chunk_ood = is_ood(chunk_metrics, threshold=self._threshold_chunk)
        aligned_evaluated_at = time.time()
        finalized_event = OODEvent(
            step=int(previous["step"]),
            timestamp=float(previous["anchor_ts"]),
            prediction_id=str(previous["prediction_id"]),
            anchor_timestamp=float(previous["anchor_ts"]),
            state_timestamp=float(previous["state_ts"]),
            evaluated_timestamp=aligned_evaluated_at,
            model_latency_ms=previous.get("model_latency_ms"),
            prediction_latency_ms=previous.get("prediction_latency_ms"),
            aligned_latency_ms=(aligned_evaluated_at - float(previous["anchor_ts"]))
            * 1000.0,
            model_breakdown_ms=dict(previous.get("model_breakdown_ms") or {}),
            frame_index=int(previous["frame_index"]),
            source_frame_ids=previous.get("source_frame_ids"),
            run_id=previous.get("run_id"),
            episode_index=previous.get("episode_index"),
            capture_mode=str(previous.get("capture_mode") or "idle"),
            replay_eligible=replay_eligible,
            immediate_score=float(previous["immediate_score"]),
            immediate_ood=bool(previous["immediate_ood"]),
            chunk_score=chunk_score,
            chunk_ood=chunk_ood,
            energy_score=chunk_score,
            energy_components=chunk_metrics["components"],
            finalized=True,
            human_chunk=human_chunk.tolist(),
            model_mean_chunk=chunk_metrics["model_mean"].tolist(),
            model_samples_chunk=model_chunk.tolist(),
            image_b64=previous.get("image_b64"),
        )
        return prediction_event, finalized_event

    def run(self) -> None:
        """Main loop: evaluate whenever recorder hardware is ready."""
        self._setup_signal_handler()
        try:
            self.start()
            while not self._stop:
                loop_start = time.perf_counter()

                for lifecycle in self._receiver.pop_lifecycle_events():
                    self._handle_lifecycle(lifecycle)

                state = self._receiver.get_latest_state(max_age_s=1.0)
                if state is None:
                    mode = "recording" if self._recording_active else "idle"
                    self._pause_inference(
                        state="paused" if self._has_seen_state else "starting",
                        mode=mode,
                        reason=(
                            "state_stream_stale"
                            if self._has_seen_state
                            else "waiting_for_recorder"
                        ),
                        detail=(
                            "Recorder state stream is unavailable or older than 1 second"
                            if self._has_seen_state
                            else "Waiting for recorder state and camera frames"
                        ),
                    )
                    time.sleep(0.05)
                    continue
                self._has_seen_state = True
                mode = "recording" if state.get("recording") is True else "idle"
                if (
                    not self._recording_active
                    and state.get("recording") is True
                    and state.get("run_root")
                    and state.get("episode") is not None
                ):
                    # Recover safely if the one-shot writer "recording" datagram
                    # was dropped before the monitor attached to the socket.
                    self._handle_lifecycle(
                        {
                            "state": "recording",
                            "run_root": state["run_root"],
                            "episode": state["episode"],
                            "timestamp": state.get("sys_ts") or time.time(),
                        }
                    )

                gate_ready, reason, detail = self._recorder_gate_status(state)
                if not gate_ready:
                    self._pause_inference(
                        mode=mode,
                        reason=reason,
                        detail=detail,
                    )
                    time.sleep(0.05)
                    continue

                if self._previous_prediction is None:
                    human_states = []
                else:
                    window_start_ts = float(self._previous_prediction["state_ts"])
                    human_states = self._receiver.get_states_between(
                        start_ts=window_start_ts,
                        end_ts=window_start_ts + self._comparison_window_s,
                        count=self._comparison_steps,
                    )

                snapshot = self._get_latest_snapshot()
                if snapshot is None or not snapshot.images:
                    self._pause_inference(
                        mode=mode,
                        reason="camera_frames_unavailable",
                        detail="No live camera frames are available to the OOD monitor",
                    )
                    time.sleep(0.05)
                    continue
                required_image_keys = {str(self._camera_mapping["main_image"])}
                extra_image_key = self._camera_mapping.get("extra_view_image")
                if extra_image_key is not None:
                    required_image_keys.add(str(extra_image_key))
                missing_image_keys = sorted(required_image_keys - set(snapshot.images))
                if missing_image_keys:
                    self._pause_inference(
                        mode=mode,
                        reason="camera_frames_incomplete",
                        detail="Missing model camera frames: "
                        + ", ".join(missing_image_keys),
                    )
                    time.sleep(0.05)
                    continue
                state_episode = state.get("episode")
                if (
                    mode == "recording"
                    and state_episode is not None
                    and (snapshot.episode_index != int(state_episode) - 1)
                ):
                    logger.debug(
                        "Waiting for episode-matched camera frame: state=%s frame=%s",
                        int(state_episode) - 1,
                        snapshot.episode_index,
                    )
                    self._publish_monitor_status(
                        state="starting",
                        mode=mode,
                        reason="waiting_for_episode_frame",
                        detail="Waiting for the first camera frame tagged to this episode",
                    )
                    time.sleep(0.01)
                    continue

                prediction_event, finalized_event = self._evaluate(
                    state, snapshot, human_states
                )
                if prediction_event is None:
                    self._pause_inference(
                        mode=mode,
                        reason="observation_invalid",
                        detail="Required pose, gripper feedback, or model image is invalid",
                    )
                    time.sleep(0.05)
                    continue
                if self._next_prediction is not None:
                    self._previous_prediction = self._next_prediction
                    self._next_prediction = None
                self._step += 1
                self._publish_inference_results(
                    mode=mode,
                    prediction_event=prediction_event,
                    finalized_event=finalized_event,
                )
                if finalized_event is not None:
                    if finalized_event.replay_eligible:
                        self._replay.append_event(
                            self._server.event_to_dict(finalized_event)
                        )
                    logger.info(
                        "Step %-5d | immediate=%.3f%s | chunk=%.3f%s | energy=%.3f | "
                        "model=%.1fms | prediction=%.1fms | aligned=%.1fms",
                        finalized_event.step,
                        finalized_event.immediate_score,
                        " (OOD)" if finalized_event.immediate_ood else "",
                        finalized_event.chunk_score,
                        " (OOD)" if finalized_event.chunk_ood else "",
                        finalized_event.energy_score,
                        finalized_event.model_latency_ms or 0.0,
                        finalized_event.prediction_latency_ms or 0.0,
                        finalized_event.aligned_latency_ms or 0.0,
                    )
                elapsed = time.perf_counter() - loop_start
                sleep_s = max(0.0, self._interval_s - elapsed)
                if sleep_s > 0 and not self._stop:
                    time.sleep(sleep_s)
        finally:
            self.stop()
            if self._inference_ms:
                avg = sum(self._inference_ms) / len(self._inference_ms)
                logger.info(
                    "OOD monitor finished after %d steps. Average K=%d inference time: %.2f ms",
                    self._step,
                    self._k,
                    avg,
                )


def _load_config(
    path: str,
    recorder_config: str | None = None,
) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    socket_path = os.environ.get("COLLECTION_OOD_SOCKET_PATH")
    if socket_path:
        cfg.setdefault("source", {})["socket_path"] = socket_path
    if recorder_config is not None:
        with open(recorder_config, "r", encoding="utf-8") as f:
            recorder_cfg = yaml.safe_load(f) or {}
        output_dir = recorder_cfg.get("output_dir")
        if output_dir:
            cfg.setdefault("replay", {})["output_root"] = str(output_dir)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD monitor for collection teleop")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ood.yaml",
        help="Path to OOD config YAML",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level",
    )
    parser.add_argument(
        "--recorder-config",
        type=str,
        default=None,
        help="Optional recorder YAML; its output_dir is used for replay discovery",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_config(args.config, args.recorder_config)
    monitor = OODMonitor(cfg)
    monitor.run()


if __name__ == "__main__":
    main()
