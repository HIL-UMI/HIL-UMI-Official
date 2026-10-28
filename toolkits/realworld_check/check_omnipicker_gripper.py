# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Check the physical right OmniPicker through training's production backend.

Passive mode opens the CAN transport and prints status without sending motion
commands. Add ``--exercise`` to stream two conservative training action targets
and require matching position feedback before reporting success.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from typing import Any

from training.envs.realworld.common.gripper import create_gripper
from training.envs.realworld.common.gripper.omnipicker_gripper import OmniPickerGripper

_RIGHT_TX_ID = 0x201
_RIGHT_RX_ID = 0x301
_RIGHT_CHANNEL_INDEX = 0


def _bounded_float(
    value: str,
    *,
    minimum: float,
    maximum: float,
    name: str,
) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(
            f"{name} must be within [{minimum}, {maximum}], got {value!r}"
        )
    return parsed


def _fraction(value: str) -> float:
    return _bounded_float(value, minimum=0.0, maximum=1.0, name="value")


def _rlinf_action(value: str) -> float:
    return _bounded_float(value, minimum=-1.0, maximum=1.0, name="action")


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            f"value must be a non-negative finite number, got {value!r}"
        )
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"value must be a positive finite number, got {value!r}"
        )
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"value must be a positive integer, got {value!r}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the single physical right OmniPicker through training. "
            "No motion command is sent unless --exercise is provided."
        ),
        epilog=(
            "Example: .venv/bin/python -m "
            "toolkits.realworld_check.check_omnipicker_gripper --exercise"
        ),
    )
    parser.add_argument(
        "--exercise",
        action="store_true",
        help="stream two small motion targets and validate position feedback",
    )
    parser.add_argument(
        "--ready-timeout",
        type=_nonnegative_float,
        default=5.0,
        help="seconds to wait for CAN sender/receiver readiness (default: 5)",
    )
    parser.add_argument(
        "--monitor-seconds",
        type=_nonnegative_float,
        default=5.0,
        help="passive status-monitoring duration (default: 5)",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_float,
        default=0.5,
        help="status print interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--cycles",
        type=_positive_int,
        default=1,
        help="number of A/B exercise cycles (default: 1)",
    )
    parser.add_argument(
        "--action-a",
        type=_rlinf_action,
        default=0.6,
        help="first training action; 0.6 maps to pos_pct=0.8 (default: 0.6)",
    )
    parser.add_argument(
        "--action-b",
        type=_rlinf_action,
        default=0.2,
        help="second training action; 0.2 maps to pos_pct=0.6 (default: 0.2)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=_positive_float,
        default=1.5,
        help="seconds to stream each exercise target (default: 1.5)",
    )
    parser.add_argument(
        "--command-rate-hz",
        type=_positive_float,
        default=10.0,
        help="exercise command rate in Hz (default: 10)",
    )
    parser.add_argument(
        "--min-feedback-delta",
        type=_fraction,
        default=0.05,
        help="minimum normalized position change required for PASS (default: 0.05)",
    )
    parser.add_argument(
        "--feedback-tolerance",
        type=_fraction,
        default=0.15,
        help="maximum target/feedback position error for PASS (default: 0.15)",
    )
    parser.add_argument(
        "--countdown-seconds",
        type=_nonnegative_float,
        default=3.0,
        help="motion safety countdown after transport readiness (default: 3)",
    )
    parser.add_argument(
        "--vel-pct",
        type=_fraction,
        default=0.2,
        help="conservative normalized exercise velocity (default: 0.2)",
    )
    parser.add_argument(
        "--force-pct",
        type=_fraction,
        default=0.2,
        help="conservative normalized exercise force (default: 0.2)",
    )
    parser.add_argument(
        "--acc-pct",
        type=_fraction,
        default=0.2,
        help="conservative normalized exercise acceleration (default: 0.2)",
    )
    parser.add_argument(
        "--dec-pct",
        type=_fraction,
        default=0.2,
        help="conservative normalized exercise deceleration (default: 0.2)",
    )
    return parser


def _transport_ready(status: dict[str, Any]) -> bool:
    return (
        bool(status)
        and not bool(status.get("disconnected"))
        and bool(status.get("sender_alive"))
        and bool(status.get("receiver_alive"))
        and bool(status.get("device_online") or status.get("online"))
    )


def _feedback_position(status: dict[str, Any]) -> float | None:
    value = status.get("position")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp(status: dict[str, Any], key: str) -> float | None:
    value = status.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_status(status: dict[str, Any]) -> str:
    position = _feedback_position(status)
    position_text = "-" if position is None else f"{position:.3f}"
    return (
        f"ready={_transport_ready(status)} "
        f"device_online={bool(status.get('device_online'))} "
        f"online={bool(status.get('online'))} "
        f"sender_alive={bool(status.get('sender_alive'))} "
        f"receiver_alive={bool(status.get('receiver_alive'))} "
        f"last_tx_ok={status.get('last_tx_ok')} "
        f"position={position_text} "
        f"fault_code={status.get('fault_code')} "
        f"disconnected={bool(status.get('disconnected'))} "
        f"reason={status.get('online_reason') or '-'}"
    )


def _wait_for_ready(
    gripper: OmniPickerGripper,
    *,
    timeout_s: float,
    poll_interval_s: float,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    print_fn: Callable[[str], None],
) -> dict[str, Any]:
    deadline = monotonic_fn() + max(0.0, timeout_s)
    interval = max(0.05, poll_interval_s)
    while True:
        status = gripper.get_status()
        print_fn("[STATUS] " + _format_status(status))
        if _transport_ready(status) or monotonic_fn() >= deadline:
            return status
        sleep_fn(min(interval, max(0.0, deadline - monotonic_fn())))


def _monitor(
    gripper: OmniPickerGripper,
    *,
    duration_s: float,
    poll_interval_s: float,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    print_fn: Callable[[str], None],
) -> dict[str, Any]:
    deadline = monotonic_fn() + max(0.0, duration_s)
    interval = max(0.05, poll_interval_s)
    status = gripper.get_status()
    while monotonic_fn() < deadline:
        print_fn("[STATUS] " + _format_status(status))
        sleep_fn(min(interval, max(0.0, deadline - monotonic_fn())))
        status = gripper.get_status()
    print_fn("[STATUS] " + _format_status(status))
    return status


def _motion_countdown(
    duration_s: float,
    *,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    print_fn: Callable[[str], None],
) -> None:
    deadline = monotonic_fn() + max(0.0, duration_s)
    while monotonic_fn() < deadline:
        remaining = max(0.0, deadline - monotonic_fn())
        print_fn(f"[WARNING] Motion starts in {math.ceil(remaining)} second(s)...")
        sleep_fn(min(1.0, remaining))


def _stream_action(
    gripper: OmniPickerGripper,
    action: float,
    *,
    duration_s: float,
    command_rate_hz: float,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    print_fn: Callable[[str], None],
) -> dict[str, Any]:
    target = gripper.rlinf_action_to_position_pct(action)
    print_fn(f"[MOVE] training action={action:.3f} -> pos_pct={target:.3f}")
    deadline = monotonic_fn() + max(0.0, duration_s)
    interval = 1.0 / max(0.1, command_rate_hz)
    first_command = True
    while first_command or monotonic_fn() < deadline:
        first_command = False
        gripper.command_rlinf_action(action)
        remaining = deadline - monotonic_fn()
        if remaining > 0.0:
            sleep_fn(min(interval, remaining))
    status = gripper.get_status()
    print_fn("[STATUS] " + _format_status(status))
    return status


def _exercise_failure_reasons(
    status_a_before: dict[str, Any],
    status_b: dict[str, Any],
    status_a_after: dict[str, Any],
    *,
    target_a: float,
    target_b: float,
    min_feedback_delta: float,
    feedback_tolerance: float,
) -> list[str]:
    reasons: list[str] = []
    statuses = (status_a_before, status_b, status_a_after)
    if not all(_transport_ready(status) for status in statuses):
        reasons.append("CAN transport is not ready after commands")
    if not all(status.get("online") is True for status in statuses):
        reasons.append("protocol feedback did not stay online during the exercise")
    if not all(status.get("last_tx_ok") is True for status in statuses):
        reasons.append("vendor sender did not report last_tx_ok=True for each target")
    if any(int(status.get("fault_code") or 0) != 0 for status in statuses):
        reasons.append("the gripper reported a non-zero fault_code")

    tx_timestamps = [_timestamp(status, "last_tx_ts") for status in statuses]
    feedback_timestamps = [_timestamp(status, "last_update_ts") for status in statuses]
    if any(timestamp is None for timestamp in tx_timestamps):
        reasons.append("CAN transmit timestamps are missing")
    elif not tx_timestamps[0] < tx_timestamps[1] < tx_timestamps[2]:
        reasons.append("CAN transmit timestamps did not advance for A/B/A")
    if any(timestamp is None for timestamp in feedback_timestamps):
        reasons.append("feedback timestamps are missing")
    elif not feedback_timestamps[0] < feedback_timestamps[1] < feedback_timestamps[2]:
        reasons.append("feedback timestamps did not advance for A/B/A")

    position_a_before = _feedback_position(status_a_before)
    position_b = _feedback_position(status_b)
    position_a_after = _feedback_position(status_a_after)
    if position_a_before is None or position_b is None or position_a_after is None:
        reasons.append("normalized position feedback is missing")
    else:
        required_delta = max(0.0, min_feedback_delta)
        deltas = (
            abs(position_b - position_a_before),
            abs(position_a_after - position_b),
        )
        if any(delta < required_delta for delta in deltas):
            reasons.append(
                "position feedback A/B/A deltas were "
                f"{deltas[0]:.3f} and {deltas[1]:.3f}; each must be at least "
                f"{required_delta:.3f}"
            )
        tolerance = max(0.0, feedback_tolerance)
        errors = (
            abs(position_a_before - target_a),
            abs(position_b - target_b),
            abs(position_a_after - target_a),
        )
        if any(error > tolerance for error in errors):
            reasons.append(
                "target/feedback errors for A/B/A were "
                f"{errors[0]:.3f}, {errors[1]:.3f}, and {errors[2]:.3f}; "
                f"each must be at most {tolerance:.3f}"
            )
    return reasons


def run_check(
    args: argparse.Namespace,
    *,
    gripper_factory: Callable[..., OmniPickerGripper] = create_gripper,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    print_fn: Callable[[str], None] = print,
) -> int:
    """Run the passive or active training hardware check."""
    print_fn(
        "[INFO] training right-only OmniPicker: "
        "channel=0 tx=0x201 rx=0x301; no left controller will be created."
    )
    print_fn(
        "[INFO] Stop the recorder, training, evaluation, and any other "
        "OmniPicker process before running this check."
    )
    print_fn("[INFO] This tool does not start or command the Franka arm.")
    if args.exercise:
        print_fn(
            "[WARNING] Motion is enabled. Clear the jaws and keep the emergency "
            "stop accessible."
        )
    else:
        print_fn("[INFO] Passive mode: no gripper motion command will be sent.")

    try:
        gripper = gripper_factory(
            gripper_type="omnipicker",
            tx_id=_RIGHT_TX_ID,
            rx_id=_RIGHT_RX_ID,
            channel_index=_RIGHT_CHANNEL_INDEX,
            is_extended_id=True,
            accept_all_rx_ids=True,
            rx_stale_timeout_s=0.5,
            online_probe_interval_s=0.2,
            rx_fail_probe_threshold=3,
            disconnect_offline_threshold=5,
            vel_pct=args.vel_pct,
            force_pct=args.force_pct,
            acc_pct=args.acc_pct,
            dec_pct=args.dec_pct,
            max_width=0.08,
        )
    except Exception as exc:
        print_fn(f"[FAIL] training could not initialize the right OmniPicker: {exc}")
        return 2

    exit_code = 2
    cleanup_error: Exception | None = None
    try:
        ready_status = _wait_for_ready(
            gripper,
            timeout_s=args.ready_timeout,
            poll_interval_s=args.poll_interval,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            print_fn=print_fn,
        )
        if not _transport_ready(ready_status):
            print_fn("[FAIL] The training OmniPicker transport did not become ready.")
            exit_code = 1
        elif not args.exercise:
            final_status = _monitor(
                gripper,
                duration_s=args.monitor_seconds,
                poll_interval_s=args.poll_interval,
                monotonic_fn=monotonic_fn,
                sleep_fn=sleep_fn,
                print_fn=print_fn,
            )
            if _transport_ready(final_status):
                print_fn(
                    "[PASS] training opened and monitored the right CAN transport. "
                    "Run again with --exercise to prove physical motion."
                )
                exit_code = 0
            else:
                print_fn("[FAIL] The CAN transport went offline while monitoring.")
                exit_code = 1
        else:
            _motion_countdown(
                args.countdown_seconds,
                monotonic_fn=monotonic_fn,
                sleep_fn=sleep_fn,
                print_fn=print_fn,
            )
            status_a_before: dict[str, Any] = {}
            status_b: dict[str, Any] = {}
            status_a_after: dict[str, Any] = {}
            for cycle in range(max(1, args.cycles)):
                print_fn(f"[INFO] Exercise cycle {cycle + 1}/{max(1, args.cycles)}")
                status_a_before = _stream_action(
                    gripper,
                    args.action_a,
                    duration_s=args.hold_seconds,
                    command_rate_hz=args.command_rate_hz,
                    monotonic_fn=monotonic_fn,
                    sleep_fn=sleep_fn,
                    print_fn=print_fn,
                )
                status_b = _stream_action(
                    gripper,
                    args.action_b,
                    duration_s=args.hold_seconds,
                    command_rate_hz=args.command_rate_hz,
                    monotonic_fn=monotonic_fn,
                    sleep_fn=sleep_fn,
                    print_fn=print_fn,
                )
                status_a_after = _stream_action(
                    gripper,
                    args.action_a,
                    duration_s=args.hold_seconds,
                    command_rate_hz=args.command_rate_hz,
                    monotonic_fn=monotonic_fn,
                    sleep_fn=sleep_fn,
                    print_fn=print_fn,
                )
            target_a = gripper.rlinf_action_to_position_pct(args.action_a)
            target_b = gripper.rlinf_action_to_position_pct(args.action_b)
            reasons = _exercise_failure_reasons(
                status_a_before,
                status_b,
                status_a_after,
                target_a=target_a,
                target_b=target_b,
                min_feedback_delta=args.min_feedback_delta,
                feedback_tolerance=args.feedback_tolerance,
            )
            if reasons:
                for reason in reasons:
                    print_fn(f"[FAIL] {reason}")
                exit_code = 1
            else:
                print_fn(
                    "[PASS] training sent actions through its OmniPicker backend "
                    "and observed matching physical position feedback."
                )
                exit_code = 0
    except Exception as exc:
        print_fn(f"[FAIL] training OmniPicker check raised an error: {exc}")
        exit_code = 2
    finally:
        try:
            gripper.cleanup()
        except Exception as exc:  # pragma: no cover - hardware defensive path
            cleanup_error = exc

    if cleanup_error is not None:
        print_fn(f"[FAIL] OmniPicker cleanup failed: {cleanup_error}")
        return 2
    print_fn("[INFO] training released the OmniPicker CAN threads and device.")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    try:
        return run_check(args)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; OmniPicker cleanup was attempted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
