# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Unix-domain-socket receiver for recorder state broadcasts."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class StateReceiver:
    """Receive JSON state packets from the patched data_recorder process.

    The recorder sends one datagram for each new Oculus sample in both idle and
    recording modes. This class buffers the latest packet and drops stale ones
    silently to keep the OOD monitor loosely coupled to the recorder.
    """

    def __init__(
        self,
        socket_path: str,
        max_packet_bytes: int = 65535,
        history_size: int = 600,
    ) -> None:
        self.socket_path = socket_path
        self.max_packet_bytes = max_packet_bytes
        self._latest: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._lifecycle: deque[dict[str, Any]] = deque(maxlen=256)
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the receiver thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("StateReceiver started on %s", self.socket_path)

    def stop(self) -> None:
        """Signal the receiver to stop."""
        self._stop_event.set()
        if self._thread is None or not self._thread.is_alive():
            return
        # Unblock the blocking socket read by sending a dummy packet to ourselves.
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.sendto(b"{}", self.socket_path)
        except Exception:  # noqa: BLE001
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_latest_state(self, max_age_s: float = 1.0) -> dict[str, Any] | None:
        """Return the most recently received state packet, or None if too old."""
        with self._latest_lock:
            state = self._latest
        if state is None:
            return None
        ts = state.get("sys_ts", 0.0)
        if max_age_s > 0 and (time.time() - ts) > max_age_s:
            return None
        return state

    def pop_lifecycle_events(self) -> list[dict[str, Any]]:
        """Atomically drain recorder lifecycle events in arrival order."""
        with self._latest_lock:
            events = list(self._lifecycle)
            self._lifecycle.clear()
        return events

    def get_recent_states(
        self,
        *,
        window_s: float,
        count: int,
        max_age_s: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Return up to ``count`` uniformly sampled recent recorder states.

        Recorder packets arrive at the configured control frequency (30 Hz in
        the field). The monitor may evaluate less often, but must retain those
        packets instead of treating one monitor tick as one action step.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        now = time.time()
        with self._latest_lock:
            history = list(self._history)
        if not history:
            return []
        latest_ts = float(history[-1].get("sys_ts") or 0.0)
        if latest_ts <= 0.0 or (max_age_s > 0 and now - latest_ts > max_age_s):
            return []
        start_ts = latest_ts - max(0.0, float(window_s))
        candidates = [
            state
            for state in history
            if start_ts <= float(state.get("sys_ts") or -1.0) <= latest_ts
        ]
        if len(candidates) <= count:
            return candidates
        # Preserve the full time span while matching the model's action count.
        if count == 1:
            return [candidates[-1]]
        indices = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
        return [candidates[index] for index in indices]

    def get_states_between(
        self,
        *,
        start_ts: float,
        end_ts: float,
        count: int,
    ) -> list[dict[str, Any]]:
        """Return states at/after ``start_ts`` and strictly before ``end_ts``.

        Including the prediction frame gives the desired ``t-15 ... t-1``
        action window when the prediction was made at ``t-15``.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        with self._latest_lock:
            candidates = [
                state
                for state in self._history
                if start_ts <= float(state.get("sys_ts") or -1.0) < end_ts
            ]
        if len(candidates) <= count:
            return candidates
        if count == 1:
            return [candidates[-1]]
        indices = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
        return [candidates[index] for index in indices]

    def _set_latest(self, state: dict[str, Any]) -> None:
        with self._latest_lock:
            self._latest = state
            self._history.append(state)

    def _dispatch_payload(self, payload: dict[str, Any]) -> None:
        message_type = str(payload.get("message_type") or "state").lower()
        if message_type == "lifecycle":
            with self._latest_lock:
                self._lifecycle.append(payload)
            return
        self._set_latest(payload)

    def _cleanup_socket(self) -> None:
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError as exc:
            logger.warning("Failed to clean up stale socket %s: %s", self.socket_path, exc)

    def _run(self) -> None:
        self._cleanup_socket()
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(self.socket_path)
            except OSError as exc:
                logger.error("Cannot bind OOD state socket %s: %s", self.socket_path, exc)
                return
            sock.settimeout(0.5)
            logger.info("Listening for recorder state on %s", self.socket_path)
            try:
                while not self._stop_event.is_set():
                    try:
                        data, _addr = sock.recvfrom(self.max_packet_bytes)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not data or data == b"{}":
                        continue
                    try:
                        payload = json.loads(data.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        logger.warning("Bad state packet: %s", exc)
                        continue
                    if not isinstance(payload, dict):
                        logger.warning("Ignoring non-object recorder payload")
                        continue
                    self._dispatch_payload(payload)
            finally:
                self._cleanup_socket()
