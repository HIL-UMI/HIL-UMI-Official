# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Tests for the RLinf-native OmniPicker physical check CLI."""

import pytest

from toolkits.realworld_check import check_omnipicker_gripper as smoke_cli


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class _FakeGripper:
    def __init__(self, *, ready=True):
        self.ready = ready
        self.actions = []
        self.cleanup_calls = 0
        self.command_count = 0
        self.position = 0.7

    @staticmethod
    def rlinf_action_to_position_pct(action):
        return (float(action) + 1.0) / 2.0

    def command_rlinf_action(self, action):
        self.actions.append(float(action))
        self.command_count += 1
        self.position = self.rlinf_action_to_position_pct(action)
        return True

    def get_status(self):
        timestamp = float(self.command_count) if self.command_count else None
        return {
            "position": self.position,
            "fault_code": 0,
            "sender_alive": self.ready,
            "receiver_alive": self.ready,
            "device_online": self.ready,
            "online": self.ready,
            "online_reason": "rx_ok" if self.ready else "usb_offline",
            "last_tx_ok": True if self.command_count else None,
            "last_tx_ts": timestamp,
            "last_update_ts": timestamp,
            "disconnected": not self.ready,
        }

    def cleanup(self):
        self.cleanup_calls += 1


def _run(
    argv,
    gripper,
):
    args = smoke_cli.build_parser().parse_args(argv)
    clock = _FakeClock()
    output = []
    factory_kwargs = {}

    def factory(**kwargs):
        factory_kwargs.update(kwargs)
        return gripper

    result = smoke_cli.run_check(
        args,
        gripper_factory=factory,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
        print_fn=output.append,
    )
    return result, factory_kwargs, output


def test_passive_mode_uses_exact_right_factory_config_without_commands():
    gripper = _FakeGripper()

    result, factory_kwargs, output = _run(
        ["--ready-timeout", "0", "--monitor-seconds", "0"],
        gripper,
    )

    assert result == 0
    assert gripper.actions == []
    assert gripper.cleanup_calls == 1
    assert factory_kwargs == {
        "gripper_type": "omnipicker",
        "tx_id": 0x201,
        "rx_id": 0x301,
        "channel_index": 0,
        "is_extended_id": True,
        "accept_all_rx_ids": True,
        "rx_stale_timeout_s": 0.5,
        "online_probe_interval_s": 0.2,
        "rx_fail_probe_threshold": 3,
        "disconnect_offline_threshold": 5,
        "vel_pct": 0.2,
        "force_pct": 0.2,
        "acc_pct": 0.2,
        "dec_pct": 0.2,
        "max_width": 0.08,
    }
    assert any("no left controller" in line for line in output)
    assert any("Run again with --exercise" in line for line in output)


def test_passive_offline_returns_failure_without_commands_and_cleans_up():
    gripper = _FakeGripper(ready=False)

    result, _, _ = _run(
        ["--ready-timeout", "0", "--monitor-seconds", "0"],
        gripper,
    )

    assert result == 1
    assert gripper.actions == []
    assert gripper.cleanup_calls == 1


def test_exercise_requires_real_a_b_a_feedback_and_passes():
    gripper = _FakeGripper()

    result, _, output = _run(
        [
            "--exercise",
            "--ready-timeout",
            "0",
            "--countdown-seconds",
            "0",
            "--hold-seconds",
            "0.2",
            "--command-rate-hz",
            "10",
        ],
        gripper,
    )

    assert result == 0
    phases = []
    for action in gripper.actions:
        if not phases or phases[-1][0] != action:
            phases.append([action, 0])
        phases[-1][1] += 1
    assert [phase[0] for phase in phases] == pytest.approx([0.6, 0.2, 0.6])
    assert all(phase[1] >= 2 for phase in phases)
    assert gripper.cleanup_calls == 1
    assert any(
        "observed matching physical position feedback" in line for line in output
    )


def test_exercise_rejects_enqueue_only_false_positive():
    class NoFeedbackGripper(_FakeGripper):
        def command_rlinf_action(self, action):
            self.actions.append(float(action))
            self.command_count += 1
            return True

        def get_status(self):
            status = super().get_status()
            status["last_tx_ts"] = 1.0
            status["last_update_ts"] = 1.0
            status["position"] = 0.7
            return status

    gripper = NoFeedbackGripper()

    result, _, output = _run(
        [
            "--exercise",
            "--ready-timeout",
            "0",
            "--countdown-seconds",
            "0",
            "--hold-seconds",
            "0.1",
            "--command-rate-hz",
            "10",
        ],
        gripper,
    )

    assert result == 1
    assert gripper.cleanup_calls == 1
    assert any("timestamps did not advance" in line for line in output)
    assert any("position feedback A/B/A deltas" in line for line in output)


def test_exercise_error_still_cleans_up():
    class ErrorGripper(_FakeGripper):
        def command_rlinf_action(self, action):
            raise RuntimeError("synthetic send failure")

    gripper = ErrorGripper()

    result, _, output = _run(
        [
            "--exercise",
            "--ready-timeout",
            "0",
            "--countdown-seconds",
            "0",
        ],
        gripper,
    )

    assert result == 2
    assert gripper.cleanup_calls == 1
    assert any("synthetic send failure" in line for line in output)


def test_keyboard_interrupt_still_cleans_up():
    class InterruptGripper(_FakeGripper):
        def command_rlinf_action(self, action):
            raise KeyboardInterrupt

    gripper = InterruptGripper()
    args = smoke_cli.build_parser().parse_args(
        [
            "--exercise",
            "--ready-timeout",
            "0",
            "--countdown-seconds",
            "0",
        ]
    )
    clock = _FakeClock()

    with pytest.raises(KeyboardInterrupt):
        smoke_cli.run_check(
            args,
            gripper_factory=lambda **_kwargs: gripper,
            monotonic_fn=clock.monotonic,
            sleep_fn=clock.sleep,
            print_fn=lambda _line: None,
        )

    assert gripper.cleanup_calls == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--vel-pct", "1.1"],
        ["--action-a", "2"],
        ["--cycles", "0"],
        ["--command-rate-hz", "0"],
        ["--hold-seconds", "-1"],
    ],
)
def test_parser_rejects_unsafe_ranges(argv):
    with pytest.raises(SystemExit):
        smoke_cli.build_parser().parse_args(argv)
