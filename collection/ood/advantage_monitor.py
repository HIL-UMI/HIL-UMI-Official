# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Real-time ADVANTAGE stage-advantage OOD monitor for collection."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from ood.advantage_model_client import AdvantageModelClient
from ood.image_reader import CameraConfig, CameraSetReader
from ood.replay import EpisodeReplayRecorder, FrameSnapshot, ReplayCatalog
from ood.server import AdvantageEvent, OODWebServer
from ood.state_receiver import StateReceiver

logger = logging.getLogger(__name__)


def is_advantage_ood(score: float, threshold: float) -> bool:
    """Apply the configured strict Advantage decision boundary."""
    return float(score) < float(threshold)


def _encode_image_to_b64(image: np.ndarray, quality: int = 70) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=quality
    )
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@dataclass(frozen=True)
class FrameAnchor:
    """One unique paired camera tick with an owned historical RGB snapshot."""

    captured_at: float
    playback_frame_index: int
    source_frame_ids: dict[str, int]
    run_id: str | None
    episode_index: int | None
    images: dict[str, np.ndarray] | None = None


class AdvantageOODMonitor:
    """Compare current RGB inputs with the 50th previous unique pair."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._stop = False
        self._step = 0
        self._recording_active = False
        self._has_seen_state = False
        self._last_monitor_status: tuple[str, str, str, str] | None = None
        self._camera_run_root: str | None = None

        detection_cfg = cfg["detection"]
        self._threshold = float(detection_cfg.get("threshold", 0.0))
        self._frame_offset = int(detection_cfg.get("frame_offset", 50))
        self._interval_s = float(detection_cfg.get("interval_s", 0.5))
        self._send_anchor_images = bool(detection_cfg.get("send_anchor_images", True))
        if self._frame_offset < 1:
            raise ValueError("detection.frame_offset must be >= 1")
        if not np.isfinite(self._threshold):
            raise ValueError("detection.threshold must be finite")
        if not np.isfinite(self._interval_s) or self._interval_s <= 0:
            raise ValueError("detection.interval_s must be finite and > 0")

        source_cfg = cfg["source"]
        replay_cfg = cfg.get("replay") or {}
        self._socket_path = str(source_cfg["socket_path"])
        self._camera_frequency_hz = float(
            source_cfg.get("camera_frequency_hz", replay_cfg.get("fps", 30.0))
        )
        if not np.isfinite(self._camera_frequency_hz) or self._camera_frequency_hz <= 0:
            raise ValueError("source.camera_frequency_hz must be finite and > 0")
        self._camera_connect_timeout_s = float(
            source_cfg.get("camera_connect_timeout_s", 0.0)
        )
        stale_reconnect_s = float(source_cfg.get("camera_stale_reconnect_s", 1.0))
        self._camera_continuity_gap_s = max(0.1, stale_reconnect_s)
        camera_cfgs = {
            key: CameraConfig(
                name=str(value["name"]),
                ring_name=str(value["ring_name"]),
                width=int(value["width"]),
                height=int(value["height"]),
                ring_slots=int(value["ring_slots"]),
                stale_reconnect_s=stale_reconnect_s,
            )
            for key, value in source_cfg["cameras"].items()
        }
        model_inputs = source_cfg["model_inputs"]
        self._front_key = str(model_inputs["front"])
        self._wrist_key = str(model_inputs["wrist"])
        if self._front_key == self._wrist_key:
            raise ValueError("Advantage front and wrist cameras must be distinct")
        missing = {self._front_key, self._wrist_key} - set(camera_cfgs)
        if missing:
            raise ValueError(
                "Advantage model cameras are not configured: "
                + ", ".join(sorted(missing))
            )
        for key in (self._front_key, self._wrist_key):
            if camera_cfgs[key].ring_slots <= self._frame_offset:
                raise ValueError(
                    f"source.cameras.{key}.ring_slots must be greater than "
                    f"detection.frame_offset ({self._frame_offset})"
                )
        self._camera_cfgs = camera_cfgs
        self._camera_reader = CameraSetReader(camera_cfgs)

        self._latest_paired_snapshot: FrameSnapshot | None = None
        self._history: deque[FrameAnchor] = deque(maxlen=self._frame_offset + 1)
        self._snapshot_lock = threading.Lock()
        self._preview_thread: threading.Thread | None = None
        self._preview_warned_black: set[str] = set()

        self._model = AdvantageModelClient(cfg["model"])
        self._warmup_runs = int(cfg["model"].get("warmup_runs", 1))
        if self._warmup_runs < 0:
            raise ValueError("model.warmup_runs must be >= 0")

        self._receiver = StateReceiver(self._socket_path)
        output_root = Path(
            str(replay_cfg.get("output_root", "./data_recorder_output"))
        ).expanduser()
        self._replay_catalog = ReplayCatalog([output_root])
        self._replay = EpisodeReplayRecorder(
            fps=float(replay_cfg.get("fps", self._camera_frequency_hz)),
            front_key=str(replay_cfg.get("front_camera", self._front_key)),
            wrist_key=str(replay_cfg.get("wrist_camera", self._wrist_key)),
            ffmpeg_bin=str(replay_cfg.get("ffmpeg_bin", "ffmpeg")),
            encoder_preference=str(
                replay_cfg.get("encoder_preference", "nvenc_then_x264")
            ),
            queue_size=int(replay_cfg.get("encoder_queue_size", 30)),
            detector_mode="advantage",
            score_threshold=self._threshold,
            pretrigger_s=float(replay_cfg.get("pretrigger_s", 0.0)),
        )
        server_cfg = cfg["server"]
        self._server = OODWebServer(
            host=str(server_cfg.get("host", "0.0.0.0")),
            port=int(server_cfg.get("port", 8765)),
            static_dir=server_cfg.get("static_dir"),
            replay_catalog=self._replay_catalog,
            detector_mode="advantage",
        )
        self._inference_ms: deque[float] = deque(maxlen=100)

    @staticmethod
    def _recorder_gate_status(state: dict[str, Any]) -> tuple[bool, str, str]:
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

    @staticmethod
    def _paired_anchor_advanced(
        previous: FrameAnchor, current: FrameAnchor, keys: tuple[str, str]
    ) -> bool:
        """Count a tick only when both asynchronous cameras have advanced."""
        return all(
            current.source_frame_ids[key] > previous.source_frame_ids[key]
            for key in keys
        )

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        logger.info("Received signal %d, shutting down", signum)
        self._stop = True

    def _setup_signal_handler(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _clear_history(self, reason: str, *, reset_pretrigger: bool = False) -> None:
        with self._snapshot_lock:
            had_history = bool(self._history)
            self._history.clear()
            self._latest_paired_snapshot = None
        if had_history:
            logger.info("Advantage frame history reset (%s)", reason)
        if reset_pretrigger:
            self._replay.reset_pretrigger(reason)

    def _publish_monitor_status(
        self,
        *,
        state: str,
        mode: str,
        reason: str = "",
        detail: str = "",
    ) -> None:
        key = (str(state), str(mode), str(reason), str(detail))
        self._server.set_monitor_status(
            state=key[0],
            mode=key[1],
            reason=key[2],
            detail=key[3],
            updated_at=time.time(),
        )
        if key != self._last_monitor_status:
            logger.info(
                "Advantage monitor status: state=%s mode=%s reason=%s detail=%s",
                *key,
            )
            self._last_monitor_status = key

    def _pause_inference(
        self,
        *,
        mode: str,
        reason: str,
        detail: str,
        state: str = "paused",
        reset_history: bool = False,
        reset_pretrigger: bool = False,
    ) -> None:
        if reset_history:
            self._clear_history(reason, reset_pretrigger=reset_pretrigger)
        self._publish_monitor_status(
            state=state, mode=mode, reason=reason, detail=detail
        )

    def start(self) -> None:
        """Fail-fast on model/config, then attach the live recorder resources."""
        self._model.load()
        if self._warmup_runs:
            front_cfg = self._camera_cfgs[self._front_key]
            wrist_cfg = self._camera_cfgs[self._wrist_key]
            warmup_ms = self._model.warmup(
                runs=self._warmup_runs,
                front_shape=front_cfg.shape,
                wrist_shape=wrist_cfg.shape,
            )
            logger.info(
                "Advantage warmup complete: runs=%d latency_ms=%s",
                self._warmup_runs,
                ",".join(f"{value:.1f}" for value in warmup_ms),
            )
        self._camera_reader.connect_all(timeout_s=self._camera_connect_timeout_s)
        self._receiver.start()
        self._server.start(daemon=True)
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            name="advantage-paired-preview",
            daemon=True,
        )
        self._preview_thread.start()
        self._publish_monitor_status(
            state="starting",
            mode="idle",
            reason="waiting_for_recorder",
            detail="Waiting for recorder state and 51 unique paired camera frames",
        )
        logger.info(
            "Advantage OOD monitor started: offset=%d threshold=%.4f interval=%.3fs",
            self._frame_offset,
            self._threshold,
            self._interval_s,
        )

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

    def _publish_preview(
        self,
        images: dict[str, np.ndarray],
        source_frame_ids: dict[str, int],
        captured_at: float,
    ) -> None:
        front = images[self._front_key]
        wrist = images[self._wrist_key]
        for key, image in ((self._front_key, front), (self._wrist_key, wrist)):
            if not np.count_nonzero(image):
                if key not in self._preview_warned_black:
                    logger.warning("Preview frame from %s is all zeros", key)
                    self._preview_warned_black.add(key)
            else:
                self._preview_warned_black.discard(key)
        try:
            self._server.put_preview(
                image_b64=_encode_image_to_b64(front),
                wrist_image_b64=_encode_image_to_b64(wrist),
                timestamp=captured_at,
                source_frame_ids={
                    "front": source_frame_ids[self._front_key],
                    "wrist_left": source_frame_ids[self._wrist_key],
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish Advantage preview: %s", exc)

    def _preview_loop(self) -> None:
        interval_s = 1.0 / max(self._camera_frequency_hz, 1.0)
        next_deadline = time.monotonic()
        required = (self._front_key, self._wrist_key)
        while not self._stop:
            frames = self._camera_reader.read_frames()
            if all(key in frames for key in required):
                images = {key: frame.image for key, frame in frames.items()}
                source_frame_ids = {
                    key: frame.frame_id for key, frame in frames.items()
                }
                captured_at = time.time()
                continuity_reset_reason = ""
                with self._snapshot_lock:
                    previous = self._history[-1] if self._history else None
                    if previous is not None and any(
                        source_frame_ids[key] < previous.source_frame_ids[key]
                        for key in required
                    ):
                        continuity_reset_reason = "camera_frame_id_rollback"
                    elif (
                        previous is not None
                        and captured_at - previous.captured_at
                        > self._camera_continuity_gap_s
                    ):
                        continuity_reset_reason = "camera_frame_stream_gap"
                    if continuity_reset_reason:
                        self._history.clear()
                        self._latest_paired_snapshot = None
                if continuity_reset_reason:
                    logger.info(
                        "Advantage history reset after %s",
                        continuity_reset_reason,
                    )
                    self._replay.reset_pretrigger(continuity_reset_reason)
                snapshot = self._replay.record_frame(
                    images, source_frame_ids, captured_at
                )
                anchor = FrameAnchor(
                    captured_at=captured_at,
                    playback_frame_index=snapshot.playback_frame_index,
                    source_frame_ids={key: source_frame_ids[key] for key in required},
                    run_id=snapshot.run_id,
                    episode_index=snapshot.episode_index,
                    images={key: images[key] for key in required},
                )
                with self._snapshot_lock:
                    previous = self._history[-1] if self._history else None
                    if previous is None:
                        self._history.append(anchor)
                        self._latest_paired_snapshot = snapshot
                    elif self._paired_anchor_advanced(previous, anchor, required):
                        # A temporal tick is complete only after both cameras
                        # have advanced. This avoids counting an asynchronous
                        # half-updated pair as one of the 50 frame steps.
                        self._history.append(anchor)
                        self._latest_paired_snapshot = snapshot
                self._publish_preview(images, source_frame_ids, captured_at)
            next_deadline += interval_s
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.monotonic()

    def _handle_lifecycle(self, lifecycle: dict[str, Any]) -> None:
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
                self._clear_history("new recorder run", reset_pretrigger=True)
            if normalized_run_root is not None:
                self._camera_run_root = normalized_run_root
            self._recording_active = True
            self._replay.start_episode(lifecycle)
        elif state in {"saving", "discarding", "finalizing", "stopped"}:
            self._recording_active = False
            self._replay.stop_capture()
        elif state == "discarded":
            self._recording_active = False
            self._replay.discard(state)
        elif state == "fatal":
            self._recording_active = False
            self._replay.discard(state)
            self._clear_history(state, reset_pretrigger=True)
        elif state == "waiting" and lifecycle.get("saved_frames") is not None:
            self._recording_active = False
            manifest = self._replay.finish_saved(lifecycle)
            if manifest is not None:
                self._server.put_episode_ready(manifest)

    def _get_temporal_pair(
        self,
    ) -> tuple[FrameSnapshot | None, FrameAnchor | None, int]:
        with self._snapshot_lock:
            count = len(self._history)
            if count < self._frame_offset + 1:
                return None, None, count
            snapshot = self._latest_paired_snapshot
            current = self._history[-1]
            reference = self._history[-(self._frame_offset + 1)]
            if snapshot is None:
                return None, None, count
            required_ids = {
                key: snapshot.source_frame_ids.get(key)
                for key in (self._front_key, self._wrist_key)
            }
            if required_ids != current.source_frame_ids:
                return None, None, count
            return snapshot, reference, count

    def _evaluate(
        self,
        snapshot: FrameSnapshot,
        reference: FrameAnchor,
        mode: str,
    ) -> AdvantageEvent | None:
        current_front = snapshot.images[self._front_key]
        current_wrist = snapshot.images[self._wrist_key]
        if reference.images is not None and all(
            key in reference.images for key in (self._front_key, self._wrist_key)
        ):
            reference_front = reference.images[self._front_key]
            reference_wrist = reference.images[self._wrist_key]
        else:
            historic = self._camera_reader.read_frame_ids(reference.source_frame_ids)
            missing = {self._front_key, self._wrist_key} - set(historic)
            if missing:
                self._pause_inference(
                    mode=mode,
                    reason="reference_frames_overwritten",
                    detail=(
                        "Exact reference frame is no longer retained for: "
                        + ", ".join(sorted(missing))
                    ),
                    reset_history=True,
                    reset_pretrigger=True,
                )
                return None
            reference_front = historic[self._front_key].image
            reference_wrist = historic[self._wrist_key].image
        score, model_latency_ms = self._model.predict(
            current_front=current_front,
            current_wrist=current_wrist,
            reference_front=reference_front,
            reference_wrist=reference_wrist,
            frame_index=max(0, snapshot.playback_frame_index),
            episode_index=max(0, snapshot.episode_index or 0),
        )
        self._inference_ms.append(model_latency_ms)
        evaluated_at = time.time()
        reference_frame_index = self._replay.resolve_frame_index(
            reference.source_frame_ids, reference.captured_at
        )
        if reference_frame_index is None:
            reference_frame_index = reference.playback_frame_index
        replay_eligible = bool(
            mode == "recording"
            and snapshot.run_id
            and snapshot.episode_index is not None
            and snapshot.playback_frame_index >= 0
            and reference_frame_index >= 0
        )
        prediction_id = (
            f"{snapshot.run_id or 'live'}:"
            f"{snapshot.episode_index if snapshot.episode_index is not None else -1}:"
            f"{snapshot.playback_frame_index}:{self._step}"
        )
        event = AdvantageEvent(
            step=self._step,
            timestamp=snapshot.captured_at,
            advantage_score=score,
            threshold=self._threshold,
            ood=is_advantage_ood(score, self._threshold),
            prediction_id=prediction_id,
            reference_offset=self._frame_offset,
            frame_index=snapshot.playback_frame_index,
            reference_frame_index=reference_frame_index,
            source_frame_ids={
                "front": snapshot.source_frame_ids[self._front_key],
                "wrist_left": snapshot.source_frame_ids[self._wrist_key],
            },
            reference_source_frame_ids={
                "front": reference.source_frame_ids[self._front_key],
                "wrist_left": reference.source_frame_ids[self._wrist_key],
            },
            reference_timestamp=reference.captured_at,
            evaluated_timestamp=evaluated_at,
            model_latency_ms=model_latency_ms,
            prediction_latency_ms=(evaluated_at - snapshot.captured_at) * 1000.0,
            run_id=snapshot.run_id,
            episode_index=snapshot.episode_index,
            capture_mode=mode,
            replay_eligible=replay_eligible,
        )
        if self._send_anchor_images:
            try:
                self._server.put_advantage_anchor(
                    current_front_b64=_encode_image_to_b64(current_front),
                    current_wrist_b64=_encode_image_to_b64(current_wrist),
                    reference_front_b64=_encode_image_to_b64(reference_front),
                    reference_wrist_b64=_encode_image_to_b64(reference_wrist),
                    timestamp=snapshot.captured_at,
                    reference_timestamp=reference.captured_at,
                    source_frame_ids=event.source_frame_ids or {},
                    reference_source_frame_ids=event.reference_source_frame_ids or {},
                    frame_index=event.frame_index,
                    reference_frame_index=event.reference_frame_index,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to publish Advantage anchor images: %s", exc)
        return event

    def run(self) -> None:
        self._setup_signal_handler()
        try:
            self.start()
            while not self._stop:
                loop_started = time.perf_counter()
                for lifecycle in self._receiver.pop_lifecycle_events():
                    self._handle_lifecycle(lifecycle)

                state = self._receiver.get_latest_state(max_age_s=1.0)
                mode = "recording" if self._recording_active else "idle"
                if state is None:
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
                        reset_history=False,
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
                        reset_history=False,
                    )
                    time.sleep(0.05)
                    continue

                snapshot, reference, history_count = self._get_temporal_pair()
                if snapshot is None or reference is None:
                    self._publish_monitor_status(
                        state="starting",
                        mode=mode,
                        reason="advantage_warmup",
                        detail=(
                            f"Collecting unique paired frames: {history_count}/"
                            f"{self._frame_offset + 1}"
                        ),
                    )
                    time.sleep(0.02)
                    continue
                state_episode = state.get("episode")
                if (
                    mode == "recording"
                    and state_episode is not None
                    and (snapshot.episode_index != int(state_episode) - 1)
                ):
                    # Keep the last RUNNING status while the first post-A camera
                    # tick is attached to the new replay. This normally lasts one
                    # preview interval and must not reset continuous inference.
                    time.sleep(0.02)
                    continue

                event = self._evaluate(snapshot, reference, mode)
                if event is None:
                    time.sleep(0.05)
                    continue
                self._step += 1
                detail = (
                    "Advantage inference active during episode capture"
                    if mode == "recording"
                    else "Advantage inference active; idle results are buffered"
                )
                self._server.put_advantage_result(
                    event=event,
                    mode=mode,
                    detail=detail,
                    updated_at=time.time(),
                )
                self._last_monitor_status = ("running", mode, "", detail)
                self._replay.append_event(self._server.advantage_event_to_dict(event))
                logger.info(
                    "Step %-5d | advantage=%+.4f%s threshold=%+.4f | "
                    "current=%d reference=%d | model=%.1fms total=%.1fms",
                    event.step,
                    event.advantage_score,
                    " (OOD)" if event.ood else "",
                    event.threshold,
                    event.frame_index,
                    event.reference_frame_index,
                    event.model_latency_ms or 0.0,
                    event.prediction_latency_ms or 0.0,
                )
                sleep_s = max(
                    0.0, self._interval_s - (time.perf_counter() - loop_started)
                )
                if sleep_s > 0 and not self._stop:
                    time.sleep(sleep_s)
        finally:
            self.stop()
            if self._inference_ms:
                average = sum(self._inference_ms) / len(self._inference_ms)
                logger.info(
                    "Advantage monitor finished after %d steps; average model "
                    "latency %.2f ms",
                    self._step,
                    average,
                )


def _load_config(path: str, recorder_config: str | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    socket_path = os.environ.get("COLLECTION_OOD_SOCKET_PATH")
    if socket_path:
        cfg.setdefault("source", {})["socket_path"] = socket_path
    if recorder_config is not None:
        with open(recorder_config, "r", encoding="utf-8") as file:
            recorder_cfg = yaml.safe_load(file) or {}
        output_dir = recorder_cfg.get("output_dir")
        if output_dir:
            cfg.setdefault("replay", {})["output_root"] = str(output_dir)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADVANTAGE stage-advantage OOD monitor for collection teleop"
    )
    parser.add_argument(
        "--config",
        default="config/advantage_ood.yaml",
        help="Path to the independent Advantage OOD YAML",
    )
    parser.add_argument("--recorder-config", default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    AdvantageOODMonitor(_load_config(args.config, args.recorder_config)).run()


if __name__ == "__main__":
    main()
