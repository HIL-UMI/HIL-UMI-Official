# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Persistent, frame-addressable OOD episode replay archives."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class FrameSnapshot:
    """One paired camera tick shared by preview, inference, and replay."""

    captured_at: float
    playback_frame_index: int
    images: dict[str, np.ndarray]
    source_frame_ids: dict[str, int]
    run_id: str | None = None
    episode_index: int | None = None


@dataclass(frozen=True)
class _BufferedReplayFrame:
    """One owned preview snapshot retained for a possible future episode."""

    captured_at: float
    images: dict[str, np.ndarray]
    source_frame_ids: dict[str, int]


def _probe_nvenc(ffmpeg_bin: str) -> bool:
    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:r=1",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


class _RawVideoEncoder:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        ffmpeg_bin: str,
        codec: str,
    ) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self._process: subprocess.Popen[bytes] | None = None
        codec_args = (
            ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
            if codec == "h264_nvenc"
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        )
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{float(fps):.6f}",
            "-i",
            "-",
            "-an",
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, image: np.ndarray) -> None:
        image = np.asarray(image, dtype=np.uint8)
        if image.shape != (self.height, self.width, 3):
            raise ValueError(
                f"video frame shape {image.shape} does not match "
                f"{(self.height, self.width, 3)}"
            )
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("video encoder is closed")
        if self._process.poll() is not None:
            raise RuntimeError("video encoder exited before frame write")
        self._process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            return_code = process.wait(timeout=30.0)
        except subprocess.TimeoutExpired as exc:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError(f"video encoder did not finalize: {self.path}") from exc
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"video encoder failed ({return_code}) for {self.path}: {detail}"
            )


class EpisodeReplayRecorder:
    """Write paired H.264 videos and aligned event data for one active episode."""

    def __init__(
        self,
        *,
        fps: float = 30.0,
        front_key: str = "front",
        wrist_key: str = "wrist_left",
        ffmpeg_bin: str = "ffmpeg",
        encoder_preference: str = "nvenc_then_x264",
        queue_size: int = 30,
        detector_mode: str = "pi05",
        score_threshold: float | None = None,
        pretrigger_s: float = 0.0,
    ) -> None:
        self.fps = float(fps)
        self.front_key = front_key
        self.wrist_key = wrist_key
        self.ffmpeg_bin = ffmpeg_bin
        self.encoder_preference = encoder_preference
        self.queue_size = int(queue_size)
        if detector_mode not in {"pi05", "advantage"}:
            raise ValueError(f"Unsupported detector_mode: {detector_mode!r}")
        self.detector_mode = detector_mode
        self.score_threshold = (
            None if score_threshold is None else float(score_threshold)
        )
        self.pretrigger_s = float(pretrigger_s)
        if not math.isfinite(self.pretrigger_s) or self.pretrigger_s < 0:
            raise ValueError("pretrigger_s must be finite and >= 0")
        self._pretrigger_capacity = int(math.ceil(self.fps * self.pretrigger_s))
        self._lock = threading.Lock()
        self._active: dict[str, Any] | None = None
        self._known_run_roots: set[Path] = set()
        self._pretrigger_frames: deque[_BufferedReplayFrame] = deque(
            maxlen=self._pretrigger_capacity
        )
        self._pretrigger_events: deque[dict[str, Any]] = deque(
            maxlen=self._pretrigger_capacity
        )

    @property
    def known_run_roots(self) -> list[Path]:
        with self._lock:
            return list(self._known_run_roots)

    def active_identity(self) -> tuple[str, int] | None:
        with self._lock:
            if self._active is None:
                return None
            return self._active["run_id"], self._active["episode_index"]

    def _frame_key(
        self, source_frame_ids: dict[str, int] | None, captured_at: float | None
    ) -> tuple[int, int, float] | None:
        if source_frame_ids is None or captured_at is None:
            return None
        try:
            return (
                int(source_frame_ids[self.front_key]),
                int(source_frame_ids[self.wrist_key]),
                float(captured_at),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _trim_pretrigger_events_locked(self, newest_at: float) -> None:
        cutoff = float(newest_at) - self.pretrigger_s
        while self._pretrigger_events:
            try:
                event_at = float(self._pretrigger_events[0].get("timestamp", 0.0))
            except (TypeError, ValueError):
                event_at = 0.0
            if event_at >= cutoff:
                break
            self._pretrigger_events.popleft()

    def _buffer_frame_locked(
        self,
        images: dict[str, np.ndarray],
        source_frame_ids: dict[str, int],
        captured_at: float,
    ) -> _BufferedReplayFrame | None:
        if self._pretrigger_capacity <= 0:
            return None
        try:
            buffered_images = {
                self.front_key: images[self.front_key],
                self.wrist_key: images[self.wrist_key],
            }
            buffered_ids = {
                self.front_key: int(source_frame_ids[self.front_key]),
                self.wrist_key: int(source_frame_ids[self.wrist_key]),
            }
        except (KeyError, TypeError, ValueError):
            return None
        frame = _BufferedReplayFrame(
            captured_at=float(captured_at),
            images=buffered_images,
            source_frame_ids=buffered_ids,
        )
        self._pretrigger_frames.append(frame)
        self._trim_pretrigger_events_locked(frame.captured_at)
        return frame

    def _active_frame_index_locked(
        self,
        active: dict[str, Any],
        source_frame_ids: dict[str, int] | None,
        captured_at: float | None,
    ) -> int | None:
        key = self._frame_key(source_frame_ids, captured_at)
        if key is None:
            return None
        value = active["frame_lookup"].get(key)
        return None if value is None else int(value)

    def _normalize_event_locked(
        self,
        event: dict[str, Any],
        active: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any] | None:
        frame_index = self._active_frame_index_locked(
            active, event.get("source_frame_ids"), event.get("timestamp")
        )
        if frame_index is None:
            return None
        normalized = dict(event)
        reference_ids = event.get("reference_source_frame_ids")
        if reference_ids is not None:
            reference_index = self._active_frame_index_locked(
                active, reference_ids, event.get("reference_timestamp")
            )
            if reference_index is None:
                return None
            normalized["reference_frame_index"] = reference_index
            normalized["reference_phase"] = (
                "pretrigger"
                if reference_index < active["recording_start_frame"]
                else "recording"
            )
        source_prediction_id = str(event.get("prediction_id") or "")
        if source_prediction_id:
            normalized["source_prediction_id"] = source_prediction_id
        normalized.update(
            {
                "prediction_id": (
                    f"{active['run_id']}:{active['episode_index']}:"
                    f"{frame_index}:{event.get('step', -1)}"
                ),
                "frame_index": frame_index,
                "run_id": active["run_id"],
                "episode_index": active["episode_index"],
                "capture_mode": phase,
                "phase": phase,
                "replay_eligible": True,
            }
        )
        return normalized

    def resolve_frame_index(
        self, source_frame_ids: dict[str, int], captured_at: float
    ) -> int | None:
        """Resolve a live/pre-trigger camera tick into the active replay timeline."""
        with self._lock:
            if self._active is None:
                return None
            return self._active_frame_index_locked(
                self._active, source_frame_ids, captured_at
            )

    def reset_pretrigger(self, reason: str) -> None:
        """Forget rolling replay context after a real camera discontinuity."""
        with self._lock:
            had_context = bool(self._pretrigger_frames or self._pretrigger_events)
            self._pretrigger_frames.clear()
            self._pretrigger_events.clear()
        if had_context:
            logger.info("Replay pre-trigger buffer reset (%s)", reason)

    def start_episode(self, lifecycle: dict[str, Any]) -> None:
        run_root_raw = lifecycle.get("run_root") or lifecycle.get("dataset_root")
        episode_number = int(lifecycle.get("episode") or 0)
        if not run_root_raw or episode_number < 1:
            logger.warning("Replay start ignored: lifecycle lacks run_root/episode")
            return
        run_root = Path(str(run_root_raw)).expanduser().resolve()
        run_id = run_root.name
        episode_index = episode_number - 1
        with self._lock:
            if self._active is not None:
                current = (self._active["run_id"], self._active["episode_index"])
                if current == (run_id, episode_index):
                    return
                logger.warning("Abandoning unfinished replay episode %s", current)
                self._abort_locked("superseded")
            replay_root = run_root / "ood_replay"
            replay_root.mkdir(parents=True, exist_ok=True)
            staging = (
                replay_root / f".episode_{episode_index:06d}.partial-{os.getpid()}"
            )
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            pretrigger_frames = list(self._pretrigger_frames)
            live_queue_reserve = max(
                self.queue_size,
                self._pretrigger_capacity,
            )
            queue_capacity = len(pretrigger_frames) + live_queue_reserve
            frame_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
            recording_started_at = float(lifecycle.get("timestamp") or time.time())
            active = {
                "run_root": run_root,
                "run_id": run_id,
                "episode_index": episode_index,
                "episode_number": episode_number,
                "started_at": recording_started_at,
                "recording_started_at": recording_started_at,
                "staging": staging,
                "queue": frame_queue,
                "events": [],
                "frame_count": len(pretrigger_frames),
                "frame_lookup": {},
                "recording_start_frame": len(pretrigger_frames),
                "pretrigger_frame_count": len(pretrigger_frames),
                "pretrigger_event_count": 0,
                "pretrigger_started_at": (
                    pretrigger_frames[0].captured_at if pretrigger_frames else None
                ),
                "pretrigger_ended_at": (
                    pretrigger_frames[-1].captured_at if pretrigger_frames else None
                ),
                "capturing": True,
                "encoders": None,
                "codec": None,
                "failed": None,
            }
            for frame_index, frame in enumerate(pretrigger_frames):
                key = self._frame_key(frame.source_frame_ids, frame.captured_at)
                if key is not None:
                    active["frame_lookup"][key] = frame_index
                frame_queue.put_nowait((frame_index, frame.images))
            for event in self._pretrigger_events:
                normalized = self._normalize_event_locked(
                    event, active, phase="pretrigger"
                )
                if normalized is not None:
                    active["events"].append(normalized)
            active["pretrigger_event_count"] = len(active["events"])
            worker = threading.Thread(
                target=self._writer_loop,
                args=(active,),
                name=f"ood-replay-{episode_index:06d}",
                daemon=True,
            )
            active["worker"] = worker
            self._active = active
            self._known_run_roots.add(run_root)
            worker.start()
        logger.info(
            "Replay capture started: %s episode=%d pretrigger_frames=%d "
            "pretrigger_events=%d",
            run_id,
            episode_index,
            active["pretrigger_frame_count"],
            active["pretrigger_event_count"],
        )

    def _make_encoders(
        self,
        active: dict[str, Any],
        images: dict[str, np.ndarray],
    ) -> dict[str, _RawVideoEncoder]:
        codec = "libx264"
        if self.encoder_preference == "nvenc_then_x264" and _probe_nvenc(
            self.ffmpeg_bin
        ):
            codec = "h264_nvenc"
        active["codec"] = codec
        encoders: dict[str, _RawVideoEncoder] = {}
        for key, filename in (
            (self.front_key, "front.mp4"),
            (self.wrist_key, "wrist_left.mp4"),
        ):
            image = np.asarray(images[key])
            height, width = image.shape[:2]
            encoders[key] = _RawVideoEncoder(
                active["staging"] / filename,
                width=width,
                height=height,
                fps=self.fps,
                ffmpeg_bin=self.ffmpeg_bin,
                codec=codec,
            )
        logger.info("Replay encoder selected: %s", codec)
        return encoders

    def _writer_loop(self, active: dict[str, Any]) -> None:
        frame_queue = active["queue"]
        try:
            while True:
                item = frame_queue.get()
                if item is None:
                    break
                _index, images = item
                if active["encoders"] is None:
                    active["encoders"] = self._make_encoders(active, images)
                for key, encoder in active["encoders"].items():
                    encoder.write(images[key])
        except Exception as exc:  # noqa: BLE001
            active["failed"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Replay video writer failed")

    def record_frame(
        self,
        images: dict[str, np.ndarray],
        source_frame_ids: dict[str, int],
        captured_at: float,
    ) -> FrameSnapshot:
        with self._lock:
            buffered = self._buffer_frame_locked(images, source_frame_ids, captured_at)
            active = self._active
            if active is None or not active["capturing"]:
                return FrameSnapshot(
                    captured_at=captured_at,
                    playback_frame_index=-1,
                    images=images,
                    source_frame_ids=source_frame_ids,
                )
            frame_index = int(active["frame_count"])
            active["frame_count"] = frame_index + 1
            key = self._frame_key(source_frame_ids, captured_at)
            if key is not None:
                active["frame_lookup"][key] = frame_index
            try:
                queued_images = buffered.images if buffered is not None else images
                active["queue"].put_nowait((frame_index, queued_images))
            except queue.Full:
                active["failed"] = "frame queue overflow"
            return FrameSnapshot(
                captured_at=captured_at,
                playback_frame_index=frame_index,
                images=images,
                source_frame_ids=source_frame_ids,
                run_id=active["run_id"],
                episode_index=active["episode_index"],
            )

    def stop_capture(self) -> None:
        """Freeze the episode frame range while writer finalization completes."""
        with self._lock:
            if self._active is not None:
                self._active["capturing"] = False

    def append_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            event = dict(event)
            if self._pretrigger_capacity > 0:
                self._pretrigger_events.append(event)
                newest_at = (
                    self._pretrigger_frames[-1].captured_at
                    if self._pretrigger_frames
                    else float(event.get("timestamp") or time.time())
                )
                self._trim_pretrigger_events_locked(newest_at)
            active = self._active
            if active is None or not active["capturing"]:
                return
            if self._pretrigger_capacity > 0:
                normalized = self._normalize_event_locked(
                    event, active, phase="recording"
                )
                if normalized is not None:
                    active["events"].append(normalized)
                return
            if (
                event.get("run_id") != active["run_id"]
                or int(event.get("episode_index", -1)) != active["episode_index"]
            ):
                return
            active["events"].append(event)

    def _close_worker_locked(self, active: dict[str, Any]) -> None:
        try:
            active["queue"].put(None, timeout=2.0)
        except queue.Full:
            active["failed"] = active["failed"] or "could not stop frame writer"
        worker = active.get("worker")
        if worker is not None:
            worker.join(timeout=60.0)
            if worker.is_alive():
                active["failed"] = active["failed"] or "frame writer did not stop"
        for encoder in (active.get("encoders") or {}).values():
            try:
                encoder.close()
            except Exception as exc:  # noqa: BLE001
                active["failed"] = active["failed"] or f"{type(exc).__name__}: {exc}"

    def finish_saved(self, lifecycle: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            active = self._active
            if active is None:
                return None
            self._close_worker_locked(active)
            self._active = None
        staging: Path = active["staging"]
        if active["failed"] or active["frame_count"] < 1:
            incomplete = staging.with_name(
                f".episode_{active['episode_index']:06d}.incomplete-{int(time.time())}"
            )
            staging.replace(incomplete)
            logger.error("Replay not published: %s", active["failed"] or "no frames")
            return None

        events = sorted(active["events"], key=lambda event: event["frame_index"])
        if self.detector_mode == "advantage":
            scores = [float(event["advantage_score"]) for event in events]
        else:
            scores = [float(event["energy_score"]) for event in events]
        manifest = {
            "schema_version": (
                3
                if self.detector_mode == "advantage" and self.pretrigger_s > 0
                else 2
                if self.detector_mode == "advantage"
                else 1
            ),
            "run_id": active["run_id"],
            "episode_index": active["episode_index"],
            "episode_number": active["episode_number"],
            "started_at": active["started_at"],
            "saved_at": float(lifecycle.get("timestamp") or time.time()),
            "fps": self.fps,
            "frame_count": active["frame_count"],
            "duration_s": active["frame_count"] / self.fps,
            "replay_started_at": (
                active["pretrigger_started_at"] or active["recording_started_at"]
            ),
            "recording_started_at": active["recording_started_at"],
            "recording_start_frame": active["recording_start_frame"],
            "recording_frame_count": max(
                0, active["frame_count"] - active["recording_start_frame"]
            ),
            "pretrigger_duration_s": active["pretrigger_frame_count"] / self.fps,
            "pretrigger_wall_duration_s": (
                max(
                    0.0,
                    active["pretrigger_ended_at"] - active["pretrigger_started_at"],
                )
                if active["pretrigger_started_at"] is not None
                and active["pretrigger_ended_at"] is not None
                else 0.0
            ),
            "pretrigger_frame_count": active["pretrigger_frame_count"],
            "pretrigger_event_count": active["pretrigger_event_count"],
            "event_count": len(events),
            "codec": active["codec"],
            "videos": {"front": "front.mp4", "wrist_left": "wrist_left.mp4"},
            "events": "events.json",
        }
        if self.detector_mode == "advantage":
            manifest.update(
                {
                    "detector_mode": "advantage",
                    "score_name": "advantage_score",
                    "score_threshold": self.score_threshold,
                    "score_mean": float(np.mean(scores)) if scores else 0.0,
                    "score_min": min(scores) if scores else 0.0,
                    "score_max": max(scores) if scores else 0.0,
                    "ood_ratio": (
                        sum(bool(event.get("ood")) for event in events) / len(events)
                        if events
                        else 0.0
                    ),
                }
            )
        else:
            # Preserve the schema-v1 pi05 manifest for existing tools/replays.
            manifest.update(
                {
                    "energy_mean": float(np.mean(scores)) if scores else 0.0,
                    "energy_max": max(scores) if scores else 0.0,
                    "ood_ratio": (
                        sum(bool(event.get("chunk_ood")) for event in events)
                        / len(events)
                        if events
                        else 0.0
                    ),
                }
            )
        (staging / "events.json").write_text(
            json.dumps(events, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        final = (
            active["run_root"] / "ood_replay" / f"episode_{active['episode_index']:06d}"
        )
        if final.exists():
            backup = final.with_name(f"{final.name}.replaced-{int(time.time())}")
            final.replace(backup)
        staging.replace(final)
        logger.info("Replay published: %s", final)
        return manifest

    def discard(self, reason: str = "discarded") -> None:
        with self._lock:
            if self._active is None:
                return
            self._abort_locked(reason)

    def _abort_locked(self, reason: str) -> None:
        active = self._active
        if active is None:
            return
        self._close_worker_locked(active)
        self._active = None
        shutil.rmtree(active["staging"], ignore_errors=True)
        logger.info("Replay staging discarded (%s)", reason)

    def close(self) -> None:
        self.discard("monitor shutdown")


class ReplayCatalog:
    """Read-only episode discovery with strict path validation."""

    def __init__(self, output_roots: list[str | Path] | None = None) -> None:
        self._lock = threading.Lock()
        self._output_roots = {
            Path(root).expanduser().resolve() for root in (output_roots or [])
        }
        self._run_roots: set[Path] = set()

    def register_run_root(self, run_root: str | Path) -> None:
        with self._lock:
            self._run_roots.add(Path(run_root).expanduser().resolve())

    def _all_run_roots(self) -> set[Path]:
        with self._lock:
            roots = set(self._run_roots)
            output_roots = set(self._output_roots)
        for output_root in output_roots:
            roots.update(
                path.resolve() for path in output_root.glob("run_*") if path.is_dir()
            )
        return roots

    def list_episodes(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        for run_root in self._all_run_roots():
            for manifest_path in (run_root / "ood_replay").glob(
                "episode_*/manifest.json"
            ):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                manifest["key"] = (
                    f"{manifest.get('run_id')}:{int(manifest.get('episode_index', 0)):06d}"
                )
                episodes.append(manifest)
        return sorted(
            episodes,
            key=lambda item: (float(item.get("started_at", 0.0)), item["key"]),
            reverse=True,
        )

    def resolve_episode(self, run_id: str, episode_index: int) -> Path:
        if not _SAFE_ID.fullmatch(run_id) or episode_index < 0:
            raise FileNotFoundError("invalid episode identifier")
        for run_root in self._all_run_roots():
            if run_root.name != run_id:
                continue
            episode = run_root / "ood_replay" / f"episode_{episode_index:06d}"
            if (episode / "manifest.json").is_file():
                return episode.resolve()
        raise FileNotFoundError(f"episode not found: {run_id}/{episode_index}")
