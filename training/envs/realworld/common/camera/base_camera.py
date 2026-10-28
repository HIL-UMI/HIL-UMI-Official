# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from training.utils.logging import get_logger

_logger = get_logger()


@dataclass
class CameraInfo:
    """Descriptor for a single camera device."""

    name: str
    serial_number: str
    camera_type: str = "realsense"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    enable_depth: bool = False
    crop_region: Optional[tuple[float, float, float, float]] = None


class BaseCamera(ABC):
    """Abstract base class for threaded camera capture.

    Subclasses must implement ``_read_frame`` (hardware-specific frame
    acquisition) and ``_close_device`` (hardware-specific cleanup).
    The threading, queue management, and public API (``open``, ``close``,
    ``get_frame``) are handled here.
    """

    def __init__(self, camera_info: CameraInfo):
        self._camera_info = camera_info
        self._frame_queue: queue.Queue = queue.Queue()
        self._frame_capturing_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self._frame_capturing_start = False

    @property
    def name(self) -> str:
        return self._camera_info.name

    def open(self):
        """Start the background frame-capturing thread."""
        self._frame_capturing_start = True
        self._frame_capturing_thread.start()

    def close(self):
        """Stop the capture thread and release hardware resources."""
        self._frame_capturing_start = False
        try:
            self._close_device()
        except Exception as e:
            # The capture thread may have already released the device after a
            # read failure; closing twice must not raise.
            _logger.warning(
                "[%s] _close_device raised %s: %s",
                self._camera_info.name,
                type(e).__name__,
                e,
            )
        if self._frame_capturing_thread.is_alive():
            self._frame_capturing_thread.join(timeout=2.0)

    def get_frame(self, timeout: int = 5) -> np.ndarray:
        """Return the most recent frame (blocks up to *timeout* seconds).

        Args:
            timeout: Maximum seconds to wait for a frame.
        """
        assert self._frame_capturing_start, (
            "Frame capturing is not started. Call open() first."
        )
        return self._frame_queue.get(timeout=timeout)

    # ── internal ──────────────────────────────────────────────────────

    def _capture_frames(self):
        failed = False
        while self._frame_capturing_start:
            time.sleep(1 / self._camera_info.fps)
            # close() may have stopped the device while this thread was
            # sleeping.  Do not attempt one final read on a stopped pipeline.
            if not self._frame_capturing_start:
                break
            try:
                has_frame, frame = self._read_frame()
            except Exception as e:
                # Stopping a RealSense pipeline unblocks wait_for_frames() by
                # raising.  That is an expected part of close(), not a capture
                # failure that should trigger another device close/reconnect.
                if not self._frame_capturing_start:
                    break
                _logger.error(
                    "[%s] _read_frame raised %s: %s — exiting capture thread.",
                    self._camera_info.name,
                    type(e).__name__,
                    e,
                )
                failed = True
                break
            if not has_frame:
                _logger.error(
                    "[%s] _read_frame returned (False, None) — exiting capture thread.",
                    self._camera_info.name,
                )
                failed = True
                break
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)
        if failed:
            # Release the device so a reopen doesn't hit "device busy": the
            # dead pipeline would otherwise keep the camera claimed forever.
            try:
                self._close_device()
            except Exception as e:
                _logger.warning(
                    "[%s] close after capture failure raised %s: %s",
                    self._camera_info.name,
                    type(e).__name__,
                    e,
                )

    @abstractmethod
    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera hardware.

        Returns:
            ``(success, frame)`` where *frame* is a BGR ``uint8`` numpy array,
            or ``(False, None)`` on failure.
        """
        raise NotImplementedError

    @abstractmethod
    def _close_device(self) -> None:
        """Release hardware-specific resources (pipeline, SDK handle, …)."""
        raise NotImplementedError
