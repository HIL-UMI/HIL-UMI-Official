# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory ring reader for camera frames produced by data_recorder."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import resource_tracker
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    ring_name: str
    width: int
    height: int
    ring_slots: int
    channels: int = 3
    dtype: str = "uint8"
    stale_reconnect_s: float = 1.0

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * self.channels * np.dtype(self.dtype).itemsize

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)


@dataclass(frozen=True)
class CameraFrame:
    """One immutable camera snapshot identified by the producer frame id."""

    frame_id: int
    captured_at: float
    image: np.ndarray


# Verified original 0.1.21 shared-memory module. It uses only NumPy and the
# standard library and parses under Python 3.11; the complete recorder remains
# in its own Python 3.12 environment. No recorder source is copied into this repo.
_RECORDER_RING_SHA256 = (
    "3e1ac2ab633085303a7785fbd5435f2d855061013361a807d391a5ee35aff4bf"
)


def recorder_ring_factory() -> Callable[..., Any]:
    """Load the installed recorder ring or an explicitly provided external file."""
    external = os.environ.get("COLLECTION_RECORDER_RING_MODULE")
    if external:
        path = Path(external).expanduser().resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != _RECORDER_RING_SHA256:
            raise RuntimeError(
                "External shared_memory_ring.py does not match the verified "
                "data_recorder 0.1.21 module. Supply the supported recorder build."
            )
        spec = importlib.util.spec_from_file_location(
            "_hil_umi_external_recorder_ring", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load external recorder ring: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SharedMemoryRing
    try:
        from data_recorder.shared_memory_ring import SharedMemoryRing
    except ImportError as exc:
        raise RuntimeError(
            "Camera reading requires the externally supplied data_recorder 0.1.21 "
            "package or COLLECTION_RECORDER_RING_MODULE. The launcher discovers "
            "that module from RECORDER_PYTHON; see README.md#collection-interface."
        ) from exc
    return SharedMemoryRing


class SharedMemoryRingReader:
    """Thin reader around ``data_recorder.shared_memory_ring.SharedMemoryRing``.

    We import the class from the installed ``data_recorder`` package so we do
    not need to vendor its implementation.
    """

    def __init__(
        self,
        cfg: CameraConfig,
        *,
        ring_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self._ring_factory = ring_factory
        self._ring: Any | None = None
        self._lock = threading.Lock()
        self._last_frame_id: int | None = None
        self._last_frame_change_at = time.monotonic()
        self._last_reconnect_attempt_at = 0.0

    def _attach_ring(self) -> Any:
        factory = self._ring_factory or recorder_ring_factory()
        ring = factory(
            name=self.cfg.ring_name,
            frame_bytes=self.cfg.frame_bytes,
            num_slots=self.cfg.ring_slots,
            create=False,
        )
        # Python <=3.12 registers even create=False attachments with this
        # process's resource_tracker. If this consumer exits, that tracker may
        # unlink the recorder-owned rings. The recorder is the owner and is
        # solely responsible for unlinking them.
        for shm in (ring.shm, ring.meta_shm):
            if shm is not None:
                resource_tracker.unregister(shm._name, "shared_memory")
        return ring

    def _reset_progress_locked(self) -> None:
        self._last_frame_id = None
        self._last_frame_change_at = time.monotonic()

    def connect(self) -> None:
        with self._lock:
            if self._ring is not None:
                return
            self._ring = self._attach_ring()
            self._reset_progress_locked()
        logger.info(
            "Connected to camera ring %s (%dx%d, slots=%d)",
            self.cfg.ring_name,
            self.cfg.width,
            self.cfg.height,
            self.cfg.ring_slots,
        )

    def reconnect(self, reason: str = "requested") -> None:
        """Attach the current named ring, replacing a stale unlinked mapping."""
        with self._lock:
            replacement = self._attach_ring()
            previous = self._ring
            self._ring = replacement
            self._reset_progress_locked()
            if previous is not None:
                try:
                    previous.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Error closing replaced ring %s: %s",
                        self.cfg.ring_name,
                        exc,
                    )
        logger.info("Reconnected camera ring %s (%s)", self.cfg.ring_name, reason)

    def _read_current_locked(self) -> CameraFrame | None:
        ring = self._ring
        if ring is None or not ring.supports_deterministic_validation:
            return None

        meta_arr = ring._meta_arr
        best_slot = -1
        best_frame_id = -1
        for slot in range(self.cfg.ring_slots):
            seq = int(meta_arr[slot, 0])
            if seq == 0 or seq & 1:
                continue  # never written, or write in progress
            frame_id = int(meta_arr[slot, 1])
            if frame_id > best_frame_id:
                best_frame_id = frame_id
                best_slot = slot

        if best_slot < 0:
            return None

        img = ring.read_verified(
            best_slot,
            self.cfg.shape,
            self.cfg.dtype,
            expected_frame_id=best_frame_id,
        )
        if img is None:
            return None
        return CameraFrame(
            frame_id=best_frame_id,
            captured_at=time.time(),
            image=img,
        )

    def read_latest_frame(self) -> CameraFrame | None:
        """Read the latest frame and recover from recorder ring replacement.

        Unlinking shared memory does not invalidate existing mappings. Without
        the frame-id watchdog, a monitor surviving a recorder restart keeps
        reading the old mapping's final frame forever.
        """
        if self._ring is None:
            self.connect()
        with self._lock:
            frame = self._read_current_locked()
            now = time.monotonic()
            if frame is not None and frame.frame_id != self._last_frame_id:
                self._last_frame_id = frame.frame_id
                self._last_frame_change_at = now
                return frame

            stale_s = max(0.1, float(self.cfg.stale_reconnect_s))
            should_reconnect = (
                now - self._last_frame_change_at >= stale_s
                and now - self._last_reconnect_attempt_at >= stale_s
            )
            if not should_reconnect:
                return frame

            self._last_reconnect_attempt_at = now
            try:
                replacement = self._attach_ring()
            except FileNotFoundError:
                # Recorder is between shutdown and creation of its new rings.
                return frame

            previous = self._ring
            self._ring = replacement
            self._reset_progress_locked()
            if previous is not None:
                try:
                    previous.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Error closing stale ring %s: %s",
                        self.cfg.ring_name,
                        exc,
                    )
            replacement_frame = self._read_current_locked()
            if replacement_frame is not None:
                self._last_frame_id = replacement_frame.frame_id
                self._last_frame_change_at = time.monotonic()
            logger.info(
                "Reattached camera ring %s after frame-id stall at %s",
                self.cfg.ring_name,
                frame.frame_id if frame is not None else "no-frame",
            )
            return replacement_frame

    def read_latest(self) -> np.ndarray | None:
        """Backward-compatible image-only wrapper."""
        frame = self.read_latest_frame()
        return frame.image if frame is not None else None

    def read_frame_id(self, frame_id: int) -> CameraFrame | None:
        """Read one exact retained producer frame, never a neighboring frame.

        The recorder writes frame ``N`` to ``N % ring_slots``.  Verified reads
        make the temporal pair safe even while the producer is overwriting the
        ring: if that slot no longer contains ``frame_id``, ``None`` is returned.
        """
        if frame_id < 0:
            return None
        if self._ring is None:
            self.connect()
        with self._lock:
            ring = self._ring
            if ring is None or not ring.supports_deterministic_validation:
                return None
            image = ring.read_verified(
                int(frame_id) % self.cfg.ring_slots,
                self.cfg.shape,
                self.cfg.dtype,
                expected_frame_id=int(frame_id),
            )
            if image is None:
                return None
            return CameraFrame(
                frame_id=int(frame_id),
                captured_at=time.time(),
                image=image,
            )

    def close(self) -> None:
        with self._lock:
            if self._ring is not None:
                try:
                    self._ring.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error closing ring %s: %s", self.cfg.ring_name, exc)
                self._ring = None
                self._reset_progress_locked()


class CameraSetReader:
    """Manage multiple SharedMemoryRingReaders by configured camera keys."""

    def __init__(
        self,
        cameras: dict[str, CameraConfig],
        *,
        ring_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._readers = {
            key: SharedMemoryRingReader(cfg, ring_factory=ring_factory)
            for key, cfg in cameras.items()
        }

    def connect_all(
        self,
        *,
        timeout_s: float = 0.0,
        retry_interval_s: float = 0.1,
    ) -> None:
        """Connect every configured ring, optionally waiting for recorder startup."""
        pending = set(self._readers)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        last_errors: dict[str, Exception] = {}
        while pending:
            for key in tuple(pending):
                try:
                    self._readers[key].connect()
                except FileNotFoundError as exc:
                    last_errors[key] = exc
                else:
                    pending.remove(key)
                    last_errors.pop(key, None)
            if not pending:
                return
            if time.monotonic() >= deadline:
                self.close()
                detail = "; ".join(
                    f"{key}: {last_errors.get(key)!s}" for key in sorted(pending)
                )
                raise FileNotFoundError(
                    f"Timed out waiting for camera shared-memory rings: {detail}"
                )
            time.sleep(max(0.01, float(retry_interval_s)))

    def read_all(self) -> dict[str, np.ndarray]:
        """Return the latest frame for every configured camera."""
        return {key: frame.image for key, frame in self.read_frames().items()}

    def read_frames(self) -> dict[str, CameraFrame]:
        """Return verified frames and producer ids for every configured camera."""
        out: dict[str, CameraFrame] = {}
        for key, reader in self._readers.items():
            try:
                frame = reader.read_latest_frame()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", reader.cfg.name, exc)
                frame = None
            if frame is not None:
                out[key] = frame
        return out

    def read_frame_ids(self, frame_ids: dict[str, int]) -> dict[str, CameraFrame]:
        """Return only exact historic frames that are still retained."""
        out: dict[str, CameraFrame] = {}
        for key, frame_id in frame_ids.items():
            reader = self._readers.get(key)
            if reader is None:
                continue
            try:
                frame = reader.read_frame_id(int(frame_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to read historic frame %s=%s: %s", key, frame_id, exc
                )
                frame = None
            if frame is not None:
                out[key] = frame
        return out

    def reconnect_all(self, reason: str = "requested") -> None:
        """Replace every camera attachment without stopping the preview loop."""
        for reader in self._readers.values():
            try:
                reader.reconnect(reason)
            except FileNotFoundError:
                logger.info(
                    "Camera ring %s is not available yet during reconnect",
                    reader.cfg.ring_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to reconnect %s: %s", reader.cfg.name, exc)

    def read_all_legacy(self) -> dict[str, np.ndarray]:
        """Deprecated alias kept for third-party callers."""
        out: dict[str, np.ndarray] = {}
        for key, reader in self._readers.items():
            try:
                img = reader.read_latest()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", reader.cfg.name, exc)
                img = None
            if img is not None:
                out[key] = img
        return out

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
