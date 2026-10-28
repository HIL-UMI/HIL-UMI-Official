# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

from __future__ import annotations

import time

from ood.state_receiver import StateReceiver


def test_recent_states_preserves_recorder_rate_and_window():
    receiver = StateReceiver("/tmp/unused-ood-test.sock", history_size=100)
    now = time.time()
    for index in range(30):
        receiver._set_latest({"sys_ts": now - (29 - index) / 30.0, "index": index})

    recent = receiver.get_recent_states(window_s=0.5, count=15, max_age_s=1.0)

    assert len(recent) == 15
    assert recent[0]["index"] < recent[-1]["index"]
    assert recent[-1]["index"] == 29


def test_states_between_includes_prediction_frame_and_excludes_current_frame():
    receiver = StateReceiver("/tmp/unused-ood-test.sock", history_size=100)
    now = time.time()
    for index in range(20):
        receiver._set_latest({"sys_ts": now - (19 - index) / 30.0, "index": index})

    start_ts = now - 16 / 30.0
    end_ts = now - 1 / 30.0
    states = receiver.get_states_between(start_ts=start_ts, end_ts=end_ts, count=15)

    assert [state["index"] for state in states] == list(range(3, 18))


def test_lifecycle_packets_are_separate_from_state_history():
    receiver = StateReceiver("/tmp/unused-ood-test.sock", history_size=100)
    receiver._dispatch_payload(
        {"message_type": "lifecycle", "state": "recording", "episode": 1}
    )
    receiver._dispatch_payload(
        {"message_type": "state", "sys_ts": time.time(), "episode": 1}
    )

    assert receiver.get_latest_state() is not None
    assert receiver.pop_lifecycle_events() == [
        {"message_type": "lifecycle", "state": "recording", "episode": 1}
    ]
    assert receiver.pop_lifecycle_events() == []


def test_idle_hardware_ready_state_is_buffered_like_recording_state():
    receiver = StateReceiver("/tmp/unused-ood-test.sock", history_size=10)
    state = {
        "message_type": "state",
        "sys_ts": time.time(),
        "recording": False,
        "ood_ready": True,
    }

    receiver._dispatch_payload(state)

    assert receiver.get_latest_state() == state
    assert receiver.get_recent_states(window_s=0.5, count=5) == [state]
