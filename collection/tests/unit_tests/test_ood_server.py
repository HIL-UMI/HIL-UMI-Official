# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from ood.server import AdvantageEvent, OODEvent, OODWebServer, PreviewFrame

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_str(self, payload: str) -> None:
        self.messages.append(payload)


def _event(step: int, *, finalized: bool) -> OODEvent:
    return OODEvent(
        step=step,
        timestamp=float(step),
        immediate_score=0.5,
        immediate_ood=False,
        chunk_score=1.5,
        chunk_ood=True,
        energy_score=1.5,
        prediction_id=f"prediction-{step}",
        finalized=finalized,
        model_latency_ms=220.0,
        prediction_latency_ms=245.0,
        aligned_latency_ms=750.0 if finalized else None,
        model_breakdown_ms={"prefix_ms": 60.0, "denoise_ms": 150.0},
    )


def test_websocket_snapshot_contains_preview_but_not_large_metric_history():
    async def exercise() -> tuple[list[dict], list[dict]]:
        server = OODWebServer()
        server.put_event(_event(0, finalized=False))
        server.put_event(_event(0, finalized=True))
        await server._broadcast_preview(
            PreviewFrame(
                timestamp=3.0,
                image_b64="latest-front",
                wrist_image_b64="latest-wrist",
                source_frame_ids={"front": 30, "wrist_left": 40},
            )
        )

        ws = _FakeWebSocket()
        await server._send_latest_preview(ws)
        messages = [json.loads(message) for message in ws.messages]
        return messages, list(server._live_history)

    messages, history = asyncio.run(exercise())

    assert messages == [
        {
            "type": "preview",
            "timestamp": 3.0,
            "image_b64": "latest-front",
            "wrist_image_b64": "latest-wrist",
            "source_frame_ids": {"front": 30, "wrist_left": 40},
        }
    ]
    assert [event["type"] for event in history] == [
        "prediction",
        "aligned_event",
    ]
    assert history[0]["_live_seq"] == 1
    assert history[1]["_live_seq"] == 2


def test_preview_queue_drops_complete_pairs_without_mixing_cameras():
    server = OODWebServer()

    for frame_id in (1, 2, 3):
        server.put_preview(
            image_b64=f"front-{frame_id}",
            wrist_image_b64=f"wrist-{frame_id}",
            timestamp=float(frame_id),
            source_frame_ids={
                "front": frame_id * 10,
                "wrist_left": frame_id * 10 + 1,
            },
        )

    queued = [
        server._preview_queue.get_nowait(),
        server._preview_queue.get_nowait(),
    ]
    assert [frame.image_b64 for frame in queued] == ["front-2", "front-3"]
    assert [frame.wrist_image_b64 for frame in queued] == ["wrist-2", "wrist-3"]
    assert [frame.source_frame_ids for frame in queued] == [
        {"front": 20, "wrist_left": 21},
        {"front": 30, "wrist_left": 31},
    ]


def test_episode_notifications_are_not_retained_as_live_metrics():
    async def exercise() -> list[dict]:
        server = OODWebServer()
        await server._broadcast({"type": "episode_ready", "episode": {}})
        ws = _FakeWebSocket()
        await server._send_latest_preview(ws)
        return [json.loads(message) for message in ws.messages]

    assert asyncio.run(exercise()) == []


def test_live_http_snapshot_is_populated_before_websocket_queue_is_consumed():
    async def exercise() -> dict:
        server = OODWebServer()
        server.put_event(_event(7, finalized=True))
        response = await server._live_handler(SimpleNamespace(query={"after": "0"}))
        return json.loads(response.text)

    payload = asyncio.run(exercise())

    assert payload["latest_sequence"] == 1
    assert payload["dashboard_build"] == "20260905.2"
    assert payload["detector_mode"] == "pi05"
    assert payload["events"][0]["step"] == 7
    assert payload["events"][0]["type"] == "aligned_event"
    assert payload["events"][0]["model_latency_ms"] == pytest.approx(220.0)
    assert payload["events"][0]["prediction_latency_ms"] == pytest.approx(245.0)
    assert payload["events"][0]["aligned_latency_ms"] == pytest.approx(750.0)
    assert payload["events"][0]["model_breakdown_ms"]["prefix_ms"] == pytest.approx(
        60.0
    )
    assert payload["events"][0]["capture_mode"] == "idle"
    assert payload["monitor_status"]["state"] == "starting"


def test_dashboard_build_matches_static_assets():
    from ood.server import _DASHBOARD_BUILD

    app_js = (REPO_ROOT / "ood" / "static" / "app.js").read_text(encoding="utf-8")
    index_html = (REPO_ROOT / "ood" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    builds = {
        "server": _DASHBOARD_BUILD,
        "app": re.search(r"dashboardBuild = '([0-9.]+)'", app_js).group(1),
        "css": re.search(r"styles\.css\?v=([0-9.]+)", index_html).group(1),
        "script": re.search(r"app\.js\?v=([0-9.]+)", index_html).group(1),
    }
    assert set(builds.values()) == {"20260905.2"}, builds
    assert (
        "<title>Robot-Free Human-in-the-loop Online Data Collection.</title>"
        in index_html
    )
    assert ">Robot-Free Human-in-the-loop Online Data Collection.</h1>" in index_html


def test_advantage_result_and_anchor_use_separate_live_channels():
    async def exercise() -> tuple[dict, list[dict]]:
        server = OODWebServer(detector_mode="advantage")
        event = AdvantageEvent(
            step=4,
            timestamp=10.0,
            advantage_score=-0.25,
            threshold=0.0,
            ood=True,
            frame_index=80,
            reference_frame_index=30,
        )
        server.put_advantage_result(
            event=event,
            mode="recording",
            detail="running",
            updated_at=11.0,
        )
        server.put_advantage_anchor(
            current_front_b64="cf",
            current_wrist_b64="cw",
            reference_front_b64="rf",
            reference_wrist_b64="rw",
            timestamp=10.0,
            reference_timestamp=8.0,
            source_frame_ids={"front": 80, "wrist_left": 81},
            reference_source_frame_ids={"front": 30, "wrist_left": 31},
            frame_index=80,
            reference_frame_index=30,
        )
        response = await server._live_handler(SimpleNamespace(query={"after": "0"}))
        ws = _FakeWebSocket()
        await server._send_latest_preview(ws)
        return json.loads(response.text), [json.loads(item) for item in ws.messages]

    payload, messages = asyncio.run(exercise())
    assert payload["detector_mode"] == "advantage"
    assert payload["events"] == [
        {
            "type": "advantage_event",
            "step": 4,
            "timestamp": 10.0,
            "advantage_score": -0.25,
            "threshold": 0.0,
            "ood": True,
            "prediction_id": "",
            "reference_offset": 50,
            "frame_index": 80,
            "reference_frame_index": 30,
            "source_frame_ids": None,
            "reference_source_frame_ids": None,
            "reference_timestamp": 0.0,
            "evaluated_timestamp": 0.0,
            "model_latency_ms": None,
            "prediction_latency_ms": None,
            "run_id": None,
            "episode_index": None,
            "capture_mode": "idle",
            "replay_eligible": False,
            "_live_seq": 1,
        }
    ]
    assert messages[0]["type"] == "advantage_anchor"
    assert messages[0]["reference_frame_index"] == 30


def test_live_http_reports_hardware_pause_without_a_metric_event():
    async def exercise() -> dict:
        server = OODWebServer()
        server.set_monitor_status(
            state="paused",
            mode="idle",
            reason="camera_offline",
            detail="cam_front offline",
            updated_at=123.0,
        )
        response = await server._live_handler(SimpleNamespace(query={"after": "0"}))
        return json.loads(response.text)

    payload = asyncio.run(exercise())

    assert payload["events"] == []
    assert payload["monitor_status"] == {
        "state": "paused",
        "mode": "idle",
        "reason": "camera_offline",
        "detail": "cam_front offline",
        "updated_at": pytest.approx(123.0),
    }


def test_running_status_is_published_atomically_with_recovery_result():
    async def exercise() -> dict:
        server = OODWebServer()
        server.set_monitor_status(
            state="paused",
            mode="idle",
            reason="camera_offline",
            detail="cam_front offline",
            updated_at=123.0,
        )
        server.put_inference_results(
            prediction_event=_event(8, finalized=False),
            finalized_event=_event(7, finalized=True),
            mode="idle",
            detail="Inference active; idle results are live-only",
            updated_at=124.0,
        )
        response = await server._live_handler(SimpleNamespace(query={"after": "0"}))
        return json.loads(response.text)

    payload = asyncio.run(exercise())

    assert [event["type"] for event in payload["events"]] == [
        "aligned_event",
        "prediction",
    ]
    assert payload["latest_sequence"] == 2
    assert payload["monitor_status"] == {
        "state": "running",
        "mode": "idle",
        "reason": "",
        "detail": "Inference active; idle results are live-only",
        "updated_at": pytest.approx(124.0),
    }


def test_server_start_propagates_background_bind_failure(monkeypatch):
    server = OODWebServer(host="127.0.0.1", port=8765)

    def fail_to_bind() -> None:
        server._start_error = OSError(98, "Address already in use")
        server._ready_event.set()

    monkeypatch.setattr(server, "_run_server", fail_to_bind)

    with pytest.raises(RuntimeError, match="stop the stale monitor"):
        server.start()
