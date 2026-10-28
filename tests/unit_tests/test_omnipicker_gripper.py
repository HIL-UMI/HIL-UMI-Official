# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the single-channel OmniPicker gripper adapter."""

import sys
from types import SimpleNamespace

import pytest

from training.envs.realworld.common.gripper import create_gripper
from training.envs.realworld.common.gripper.omnipicker_gripper import (
    OmniPickerGripper,
)
from training.envs.realworld.franka.end_effectors import (
    EndEffectorType,
    normalize_end_effector_type,
)


@pytest.fixture
def fake_vendor_controller(monkeypatch):
    """Install a fake vendor module so tests never touch the CAN device."""

    class FakeOmniPickerGripperController:
        instances = []
        fail_at = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.events = []
            self.commands = []
            self.state = {}
            type(self).instances.append(self)

        def _record_lifecycle_event(self, name):
            self.events.append(name)
            if self.fail_at == name:
                raise RuntimeError(f"{name} failed")

        def open_device(self):
            self._record_lifecycle_event("open_device")

        def init_channel(self):
            self._record_lifecycle_event("init_channel")

        def start_threads(self):
            self._record_lifecycle_event("start_threads")

        def send_command(self, **kwargs):
            self.commands.append(kwargs)

        def get_state(self):
            return self.state

        def cleanup(self):
            self.events.append("cleanup")

    fake_module = SimpleNamespace(
        OmniPickerGripperController=FakeOmniPickerGripperController
    )
    monkeypatch.setitem(sys.modules, "ominipicker_controller", fake_module)
    return FakeOmniPickerGripperController


def _make_gripper(**overrides):
    kwargs = {
        "tx_id": 0x201,
        "rx_id": 0x301,
        "channel_index": 0,
        "is_extended_id": True,
        "accept_all_rx_ids": True,
        "rx_stale_timeout_s": 0.5,
        "online_probe_interval_s": 0.2,
        "rx_fail_probe_threshold": 3,
        "disconnect_offline_threshold": 5,
        "vel_pct": 0.9,
        "force_pct": 0.8,
        "acc_pct": 0.7,
        "dec_pct": 0.6,
        "max_width": 0.08,
    }
    kwargs.update(overrides)
    return OmniPickerGripper(**kwargs)


def test_initializes_exactly_one_right_channel_in_vendor_order(
    fake_vendor_controller,
):
    gripper = _make_gripper()

    assert len(fake_vendor_controller.instances) == 1
    controller = fake_vendor_controller.instances[0]
    assert controller.kwargs == {
        "tx_id": 0x201,
        "rx_id": 0x301,
        "channel_index": 0,
        "is_extended_id": True,
        "accept_all_rx_ids": True,
        "rx_stale_timeout_s": 0.5,
        "online_probe_interval_s": 0.2,
        "rx_fail_probe_threshold": 3,
        "disconnect_offline_threshold": 5,
    }
    assert controller.events == ["open_device", "init_channel", "start_threads"]

    gripper.cleanup()


def test_continuous_commands_repeat_and_forward_all_gains(fake_vendor_controller):
    gripper = _make_gripper()
    controller = fake_vendor_controller.instances[0]

    gripper.command_position(0.25)
    gripper.command_position(0.25)

    expected = {
        "pos_pct": 0.25,
        "vel_pct": 0.9,
        "force_pct": 0.8,
        "acc_pct": 0.7,
        "dec_pct": 0.6,
    }
    assert controller.commands == [expected, expected]


def test_rlinf_action_mapping_uses_the_production_command_path(
    fake_vendor_controller,
):
    gripper = _make_gripper()
    controller = fake_vendor_controller.instances[0]

    results = [
        gripper.command_rlinf_action(action) for action in (-1.0, -1.0, 0.0, 1.0)
    ]

    assert [command["pos_pct"] for command in controller.commands] == pytest.approx(
        [0.0, 0.0, 0.5, 1.0]
    )
    assert results == [False, False, True, False]


def test_legacy_move_inverts_opening_fraction(fake_vendor_controller):
    gripper = _make_gripper()
    controller = fake_vendor_controller.instances[0]

    gripper.move(0, speed=0.4)
    gripper.move(255, speed=0.4)

    assert [command["pos_pct"] for command in controller.commands] == [1.0, 0.0]
    assert [command["vel_pct"] for command in controller.commands] == [0.4, 0.4]


def test_feedback_is_reported_in_metres_and_ready_tracks_disconnect(
    fake_vendor_controller,
):
    gripper = _make_gripper(max_width=0.08)
    controller = fake_vendor_controller.instances[0]
    controller.state = {
        "position": 0.75,
        "sender_alive": True,
        "receiver_alive": True,
        "device_online": True,
        "disconnected": False,
    }

    assert gripper.position == pytest.approx(0.06)
    assert not gripper.is_open
    assert gripper.is_ready()
    assert gripper.get_status() == controller.state

    # Feedback may lag or pass through the threshold while the gripper moves.
    # Openness therefore follows the last command, while position stays live.
    gripper.command_position(0.75)
    controller.state["position"] = 0.1
    assert gripper.position == pytest.approx(0.008)
    assert gripper.is_open

    controller.state["disconnected"] = True
    assert not gripper.is_ready()


def test_cleanup_is_idempotent(fake_vendor_controller):
    gripper = _make_gripper()
    controller = fake_vendor_controller.instances[0]

    gripper.cleanup()
    gripper.cleanup()

    assert controller.events.count("cleanup") == 1


def test_initialization_failure_cleans_up_vendor_controller(
    fake_vendor_controller,
):
    fake_vendor_controller.fail_at = "init_channel"

    with pytest.raises(RuntimeError, match="init_channel failed"):
        _make_gripper()

    assert len(fake_vendor_controller.instances) == 1
    assert fake_vendor_controller.instances[0].events == [
        "open_device",
        "init_channel",
        "cleanup",
    ]


@pytest.mark.parametrize("gripper_type", ["omnipicker", "ominipicker"])
def test_factory_accepts_canonical_name_and_vendor_alias(
    fake_vendor_controller,
    gripper_type,
):
    gripper = create_gripper(
        gripper_type=gripper_type,
        tx_id=0x201,
        rx_id=0x301,
        max_width=0.08,
    )

    assert isinstance(gripper, OmniPickerGripper)
    assert len(fake_vendor_controller.instances) == 1


@pytest.mark.parametrize("gripper_type", ["omnipicker", "ominipicker"])
def test_end_effector_normalization_accepts_omnipicker_alias(gripper_type):
    assert (
        normalize_end_effector_type("franka_gripper", gripper_type)
        == EndEffectorType.OMNIPICKER_GRIPPER
    )
