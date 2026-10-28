# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""aiohttp + WebSocket server for the OOD dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ood.replay import ReplayCatalog

logger = logging.getLogger(__name__)

_LIVE_HISTORY_LIMIT = 240
_DASHBOARD_BUILD = "20260905.2"


@dataclass
class OODEvent:
    """One OOD evaluation result emitted at each detection step."""

    step: int
    timestamp: float
    immediate_score: float
    immediate_ood: bool
    chunk_score: float
    chunk_ood: bool
    energy_score: float
    prediction_id: str = ""
    anchor_timestamp: float = 0.0
    state_timestamp: float = 0.0
    evaluated_timestamp: float = 0.0
    model_latency_ms: float | None = None
    prediction_latency_ms: float | None = None
    aligned_latency_ms: float | None = None
    model_breakdown_ms: dict[str, float] | None = None
    frame_index: int = -1
    source_frame_ids: dict[str, int] | None = None
    run_id: str | None = None
    episode_index: int | None = None
    capture_mode: str = "idle"
    replay_eligible: bool = False
    energy_components: dict[str, dict[str, float]] | None = None
    finalized: bool = True
    human_chunk: list[list[float]] | None = None
    model_mean_chunk: list[list[float]] | None = None
    model_samples_chunk: list[list[list[float]]] | None = None
    image_b64: str | None = None


@dataclass
class AdvantageEvent:
    """One current-versus-reference stage-advantage evaluation."""

    step: int
    timestamp: float
    advantage_score: float
    threshold: float
    ood: bool
    prediction_id: str = ""
    reference_offset: int = 50
    frame_index: int = -1
    reference_frame_index: int = -1
    source_frame_ids: dict[str, int] | None = None
    reference_source_frame_ids: dict[str, int] | None = None
    reference_timestamp: float = 0.0
    evaluated_timestamp: float = 0.0
    model_latency_ms: float | None = None
    prediction_latency_ms: float | None = None
    run_id: str | None = None
    episode_index: int | None = None
    capture_mode: str = "idle"
    replay_eligible: bool = False


@dataclass
class PreviewFrame:
    """A paired live-camera tick independent of the slower OOD event stream."""

    timestamp: float
    image_b64: str
    wrist_image_b64: str
    source_frame_ids: dict[str, int]


class OODWebServer:
    """In-process WebSocket broadcast server for OOD events.

    The monitor thread calls :meth:`put_event` from any thread; the server
    forwards events to all connected browser clients via WebSocket.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        static_dir: str | Path | None = None,
        replay_catalog: ReplayCatalog | None = None,
        detector_mode: str = "pi05",
    ):
        self.host = host
        self.port = port
        self.static_dir = Path(static_dir) if static_dir is not None else None
        self.replay_catalog = replay_catalog or ReplayCatalog()
        if detector_mode not in {"pi05", "advantage"}:
            raise ValueError(f"Unsupported detector_mode: {detector_mode!r}")
        self.detector_mode = detector_mode

        self._event_queue: queue.Queue[OODEvent | dict[str, Any] | None] = queue.Queue()
        self._preview_queue: queue.Queue[PreviewFrame] = queue.Queue(maxsize=2)
        self._clients: set[Any] = set()
        self._lock = threading.Lock()
        # Keep metrics separate from the 30 Hz preview WebSocket.  Mixing large
        # K x T x D trajectory payloads with JPEG frames caused the metric
        # stream to be starved or dropped while video continued normally.
        self._live_history: deque[dict[str, Any]] = deque(maxlen=_LIVE_HISTORY_LIMIT)
        self._live_sequence = 0
        self._monitor_status: dict[str, Any] = {
            "state": "starting",
            "mode": "idle",
            "reason": "waiting_for_recorder",
            "detail": "Waiting for recorder state and camera frames",
            "updated_at": 0.0,
        }
        self._latest_preview: dict[str, Any] | None = None
        self._latest_advantage_anchor: dict[str, Any] | None = None
        self._server_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader_task: asyncio.Task | None = None
        self._ready_event = threading.Event()
        self._start_error: BaseException | None = None
        self._shutdown = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def put_event(self, event: OODEvent) -> None:
        """Thread-safe entry point for the monitor to push an event."""
        if self._shutdown:
            return
        body = self._event_to_dict(event)
        # Remember the metric synchronously in the monitor thread.  Live metric
        # clients fetch it from /api/live; the WebSocket remains preview-only.
        with self._lock:
            self._live_sequence += 1
            body["_live_seq"] = self._live_sequence
            self._live_history.append(body)

    def put_inference_results(
        self,
        *,
        prediction_event: OODEvent,
        finalized_event: OODEvent | None,
        mode: str,
        detail: str,
        updated_at: float,
    ) -> None:
        """Publish inference results and running status as one live snapshot.

        A client must never observe ``running`` before the result proving that
        inference recovered. Otherwise the dashboard can enter a synthetic
        transition state and remain there if result delivery is interrupted.
        """
        if self._shutdown:
            return
        events = []
        if finalized_event is not None:
            events.append(self._event_to_dict(finalized_event))
        events.append(self._event_to_dict(prediction_event))
        status = {
            "state": "running",
            "mode": str(mode),
            "reason": "",
            "detail": str(detail),
            "updated_at": float(updated_at),
        }
        with self._lock:
            for body in events:
                self._live_sequence += 1
                body["_live_seq"] = self._live_sequence
                self._live_history.append(body)
            self._monitor_status = status

    def put_advantage_result(
        self,
        *,
        event: AdvantageEvent,
        mode: str,
        detail: str,
        updated_at: float,
    ) -> None:
        """Atomically publish an Advantage event and running monitor status."""
        if self._shutdown:
            return
        body = self._advantage_event_to_dict(event)
        status = {
            "state": "running",
            "mode": str(mode),
            "reason": "",
            "detail": str(detail),
            "updated_at": float(updated_at),
        }
        with self._lock:
            self._live_sequence += 1
            body["_live_seq"] = self._live_sequence
            self._live_history.append(body)
            self._monitor_status = status

    def put_advantage_anchor(
        self,
        *,
        current_front_b64: str,
        current_wrist_b64: str,
        reference_front_b64: str,
        reference_wrist_b64: str,
        timestamp: float,
        reference_timestamp: float,
        source_frame_ids: dict[str, int],
        reference_source_frame_ids: dict[str, int],
        frame_index: int,
        reference_frame_index: int,
    ) -> None:
        """Broadcast the four images used by the latest Advantage inference."""
        if self._shutdown:
            return
        body = {
            "type": "advantage_anchor",
            "timestamp": float(timestamp),
            "reference_timestamp": float(reference_timestamp),
            "current_front_b64": current_front_b64,
            "current_wrist_b64": current_wrist_b64,
            "reference_front_b64": reference_front_b64,
            "reference_wrist_b64": reference_wrist_b64,
            "source_frame_ids": dict(source_frame_ids),
            "reference_source_frame_ids": dict(reference_source_frame_ids),
            "frame_index": int(frame_index),
            "reference_frame_index": int(reference_frame_index),
        }
        with self._lock:
            self._latest_advantage_anchor = body
        self._event_queue.put(body)

    def put_episode_ready(self, manifest: dict[str, Any]) -> None:
        """Notify clients that a finalized episode became replayable."""
        if not self._shutdown:
            self._event_queue.put({"type": "episode_ready", "episode": manifest})

    def set_monitor_status(
        self,
        *,
        state: str,
        mode: str,
        reason: str = "",
        detail: str = "",
        updated_at: float,
    ) -> None:
        """Publish the current inference readiness independently of metrics."""
        body = {
            "state": str(state),
            "mode": str(mode),
            "reason": str(reason),
            "detail": str(detail),
            "updated_at": float(updated_at),
        }
        with self._lock:
            self._monitor_status = body

    def put_preview(
        self,
        *,
        image_b64: str,
        wrist_image_b64: str,
        timestamp: float,
        source_frame_ids: dict[str, int],
    ) -> None:
        """Publish the newest paired preview, dropping stale pairs if needed."""
        if self._shutdown:
            return
        frame = PreviewFrame(
            timestamp=timestamp,
            image_b64=image_b64,
            wrist_image_b64=wrist_image_b64,
            source_frame_ids=dict(source_frame_ids),
        )
        try:
            self._preview_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._preview_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._preview_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _register_client(self, ws: Any) -> None:
        with self._lock:
            self._clients.add(ws)

    def _unregister_client(self, ws: Any) -> None:
        with self._lock:
            self._clients.discard(ws)

    def _event_to_dict(self, event: OODEvent) -> dict[str, Any]:
        return {
            "type": "aligned_event" if event.finalized else "prediction",
            "step": event.step,
            "timestamp": event.timestamp,
            "immediate_score": round(event.immediate_score, 4),
            "immediate_ood": event.immediate_ood,
            "chunk_score": round(event.chunk_score, 4),
            "chunk_ood": event.chunk_ood,
            "energy_score": round(event.energy_score, 4),
            "prediction_id": event.prediction_id,
            "anchor_timestamp": event.anchor_timestamp or event.timestamp,
            "state_timestamp": event.state_timestamp or event.timestamp,
            "evaluated_timestamp": event.evaluated_timestamp or event.timestamp,
            "model_latency_ms": event.model_latency_ms,
            "prediction_latency_ms": event.prediction_latency_ms,
            "aligned_latency_ms": event.aligned_latency_ms,
            "model_breakdown_ms": event.model_breakdown_ms,
            "frame_index": event.frame_index,
            "source_frame_ids": event.source_frame_ids,
            "run_id": event.run_id,
            "episode_index": event.episode_index,
            "capture_mode": event.capture_mode,
            "energy_components": event.energy_components,
            "finalized": event.finalized,
            "human_chunk": event.human_chunk,
            "model_mean_chunk": event.model_mean_chunk,
            "model_samples_chunk": event.model_samples_chunk,
            "image_b64": event.image_b64,
        }

    @staticmethod
    def _advantage_event_to_dict(event: AdvantageEvent) -> dict[str, Any]:
        return {
            "type": "advantage_event",
            "step": event.step,
            "timestamp": event.timestamp,
            "advantage_score": round(event.advantage_score, 6),
            "threshold": event.threshold,
            "ood": event.ood,
            "prediction_id": event.prediction_id,
            "reference_offset": event.reference_offset,
            "frame_index": event.frame_index,
            "reference_frame_index": event.reference_frame_index,
            "source_frame_ids": event.source_frame_ids,
            "reference_source_frame_ids": event.reference_source_frame_ids,
            "reference_timestamp": event.reference_timestamp,
            "evaluated_timestamp": event.evaluated_timestamp,
            "model_latency_ms": event.model_latency_ms,
            "prediction_latency_ms": event.prediction_latency_ms,
            "run_id": event.run_id,
            "episode_index": event.episode_index,
            "capture_mode": event.capture_mode,
            "replay_eligible": event.replay_eligible,
        }

    def event_to_dict(self, event: OODEvent) -> dict[str, Any]:
        """Return the stable JSON wire representation for persistence/tests."""
        return self._event_to_dict(event)

    def advantage_event_to_dict(self, event: AdvantageEvent) -> dict[str, Any]:
        """Return the stable Advantage JSON representation for replay/tests."""
        return self._advantage_event_to_dict(event)

    @staticmethod
    def _preview_to_dict(frame: PreviewFrame) -> dict[str, Any]:
        return {
            "type": "preview",
            "timestamp": frame.timestamp,
            "image_b64": frame.image_b64,
            "wrist_image_b64": frame.wrist_image_b64,
            "source_frame_ids": frame.source_frame_ids,
        }

    async def _broadcast(self, event: OODEvent | dict[str, Any]) -> None:
        body = self._event_to_dict(event) if isinstance(event, OODEvent) else event
        payload = json.dumps(body)
        with self._lock:
            clients = list(self._clients)
        dead = []
        for ws in clients:
            try:
                await ws.send_str(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    async def _broadcast_preview(self, frame: PreviewFrame) -> None:
        body = self._preview_to_dict(frame)
        with self._lock:
            self._latest_preview = body
        payload = json.dumps(body)
        with self._lock:
            clients = list(self._clients)
        dead = []
        for ws in clients:
            try:
                await ws.send_str(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    async def _queue_reader(self) -> None:
        """Read events from the thread-safe queue and broadcast them."""
        while not self._shutdown:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                try:
                    frame = self._preview_queue.get(timeout=0.03)
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                await self._broadcast_preview(frame)
                continue
            if event is None:
                break
            await self._broadcast(event)

    def _run_server(self) -> None:
        from aiohttp import web

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        @web.middleware
        async def dashboard_no_cache(request, handler):
            response = await handler(request)
            if request.path == "/" or request.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-store, max-age=0"
                response.headers["Pragma"] = "no-cache"
            return response

        app = web.Application(middlewares=[dashboard_no_cache])
        app.router.add_get("/ws", self._websocket_handler)
        app.router.add_get("/api/live", self._live_handler)
        app.router.add_get("/api/episodes", self._episodes_handler)
        app.router.add_get(
            "/api/episodes/{run_id}/{episode_index}",
            self._episode_handler,
        )
        app.router.add_get(
            "/api/episodes/{run_id}/{episode_index}/video/{camera}",
            self._episode_video_handler,
        )

        static_path = self.static_dir or (Path(__file__).parent / "static")
        if static_path.is_dir():

            async def root(_request):
                index_path = static_path / "index.html"
                if index_path.exists():
                    return web.FileResponse(str(index_path))
                return web.json_response({"message": "index.html not found"})

            app.router.add_get("/", root)
            app.router.add_static("/static", str(static_path), name="static")
        else:

            async def root(_request):
                return web.json_response(
                    {"message": "OOD dashboard static files not found"}
                )

            app.router.add_get("/", root)

        runner = web.AppRunner(app, access_log=None)
        try:
            self._loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, self.host, self.port)
            self._loop.run_until_complete(site.start())
        except BaseException as exc:  # noqa: BLE001
            self._start_error = exc
            self._ready_event.set()
            logger.exception("OOD dashboard failed to bind %s:%d", self.host, self.port)
            self._loop.run_until_complete(runner.cleanup())
            self._loop.close()
            return
        self._ready_event.set()

        # Start queue reader as a background task.
        self._reader_task = self._loop.create_task(self._queue_reader())

        try:
            self._loop.run_forever()
        finally:
            if self._reader_task is not None and not self._reader_task.done():
                self._reader_task.cancel()
            self._loop.run_until_complete(runner.cleanup())
            self._loop.close()

    async def _episodes_handler(self, _request: Any) -> Any:
        from aiohttp import web

        return web.json_response({"episodes": self.replay_catalog.list_episodes()})

    async def _live_handler(self, request: Any) -> Any:
        """Return retained live metrics after a monotonically increasing cursor."""
        from aiohttp import web

        try:
            after = max(0, int(request.query.get("after", "0")))
        except ValueError:
            raise web.HTTPBadRequest(text="after must be an integer")
        with self._lock:
            latest_sequence = self._live_sequence
            monitor_status = dict(self._monitor_status)
            retained = [
                event
                for event in self._live_history
                if int(event.get("_live_seq", 0)) > after
            ]
            # A fresh page needs useful recent context, not minutes of large
            # stochastic action tensors. Incremental polls still receive every
            # event after their cursor.
            events = retained[-60:] if after == 0 else retained
        return web.json_response(
            {
                "dashboard_build": _DASHBOARD_BUILD,
                "detector_mode": self.detector_mode,
                "events": events,
                "latest_sequence": latest_sequence,
                "monitor_status": monitor_status,
            }
        )

    async def _episode_handler(self, request: Any) -> Any:
        from aiohttp import web

        try:
            episode_index = int(request.match_info["episode_index"])
            episode = self.replay_catalog.resolve_episode(
                request.match_info["run_id"], episode_index
            )
            manifest = json.loads(
                (episode / "manifest.json").read_text(encoding="utf-8")
            )
            events = json.loads(
                (episode / manifest["events"]).read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError, KeyError):
            raise web.HTTPNotFound(text="episode not found")
        return web.json_response({"manifest": manifest, "events": events})

    async def _episode_video_handler(self, request: Any) -> Any:
        from aiohttp import web

        camera = request.match_info["camera"]
        if camera not in {"front", "wrist_left"}:
            raise web.HTTPNotFound(text="camera not found")
        try:
            episode = self.replay_catalog.resolve_episode(
                request.match_info["run_id"], int(request.match_info["episode_index"])
            )
            video = episode / ("front.mp4" if camera == "front" else "wrist_left.mp4")
            if not video.is_file():
                raise FileNotFoundError(video)
        except (FileNotFoundError, ValueError):
            raise web.HTTPNotFound(text="episode video not found")
        return web.FileResponse(video)

    async def _websocket_handler(self, request: Any) -> Any:
        from aiohttp import web

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await self._send_latest_preview(ws)
        self._register_client(ws)
        try:
            async for _msg in ws:
                # The browser never sends data; receiving keeps the connection
                # lifecycle tied to aiohttp's WebSocket implementation.
                pass
        finally:
            self._unregister_client(ws)
        return ws

    async def _send_latest_preview(self, ws: Any) -> None:
        """Send only the newest complete pair when a preview client connects."""
        with self._lock:
            preview = self._latest_preview
            advantage_anchor = self._latest_advantage_anchor
        if preview is not None:
            await ws.send_str(json.dumps(preview))
        if advantage_anchor is not None:
            await ws.send_str(json.dumps(advantage_anchor))

    def start(self, daemon: bool = True) -> None:
        """Start the server in a background thread."""
        if self._server_thread is not None:
            return
        self._shutdown = False
        self._start_error = None
        self._ready_event.clear()
        self._server_thread = threading.Thread(target=self._run_server, daemon=daemon)
        self._server_thread.start()
        if not self._ready_event.wait(timeout=5.0):
            raise RuntimeError(
                f"OOD dashboard did not start within 5 seconds on {self.host}:{self.port}"
            )
        if self._start_error is not None:
            raise RuntimeError(
                f"OOD dashboard could not bind {self.host}:{self.port}; "
                "stop the stale monitor using that port"
            ) from self._start_error
        logger.info("OOD dashboard server started at %s", self.url)

    def stop(self) -> None:
        """Signal the server to shut down."""
        self._shutdown = True
        self._event_queue.put(None)

        def _cancel_and_stop() -> None:
            if self._reader_task is not None and not self._reader_task.done():
                self._reader_task.cancel()
            self._loop.stop()

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(_cancel_and_stop)
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)
