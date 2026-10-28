# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""OmniPicker gripper controlled through a single USB-CAN-FD channel.

The vendor controller uses a normalised opening fraction: ``0`` is fully
closed and ``1`` is fully open. RLinf's legacy ``move`` API uses the opposite
raw convention (``0`` open, ``255`` closed), so this adapter performs the
conversion at the hardware boundary.
"""

from typing import Any

import numpy as np

from training.utils.logging import get_logger

from .base_gripper import BaseGripper


class OmniPickerGripper(BaseGripper):
    """Control one OmniPicker gripper on one USB-CAN-FD channel.

    This class intentionally owns exactly one controller. For the single-arm
    Franka setup, pass the parameters of the recorder's ``right`` channel; no
    ``left`` controller or shared-device-handle path is created.

    Args:
        tx_id: CAN-FD command identifier.
        rx_id: CAN-FD feedback identifier.
        channel_index: USB-CAN channel index.
        is_extended_id: Whether commands use a CAN extended identifier.
        accept_all_rx_ids: Whether the vendor receiver may auto-detect a
            feedback identifier that differs from ``rx_id``.
        rx_stale_timeout_s: Feedback staleness threshold used by the vendor
            receiver.
        online_probe_interval_s: USB device probe interval.
        rx_fail_probe_threshold: Receive failures before probing the device.
        disconnect_offline_threshold: Non-USB failures before disconnect is
            latched.
        vel_pct: Default normalised command speed.
        force_pct: Default normalised command force/torque limit.
        acc_pct: Default normalised command acceleration.
        dec_pct: Default normalised command deceleration.
        max_width: Physical opening width in metres. Feedback is scaled by
            this value to preserve RLinf's existing gripper-state units.
        open_threshold: Command opening fraction classified as open.
    """

    def __init__(
        self,
        tx_id: int = 0x201,
        rx_id: int = 0x301,
        channel_index: int = 0,
        is_extended_id: bool = True,
        accept_all_rx_ids: bool = True,
        rx_stale_timeout_s: float = 0.5,
        online_probe_interval_s: float = 0.2,
        rx_fail_probe_threshold: int = 3,
        disconnect_offline_threshold: int = 5,
        vel_pct: float = 1.0,
        force_pct: float = 1.0,
        acc_pct: float = 1.0,
        dec_pct: float = 1.0,
        max_width: float = 0.08,
        open_threshold: float = 0.5,
    ) -> None:
        self._logger = get_logger()
        self._vel_pct = self._clip_pct(vel_pct)
        self._force_pct = self._clip_pct(force_pct)
        self._acc_pct = self._clip_pct(acc_pct)
        self._dec_pct = self._clip_pct(dec_pct)
        self._open_threshold = self._clip_pct(open_threshold)
        self._max_width = float(max_width)
        if not np.isfinite(self._max_width) or self._max_width <= 0.0:
            raise ValueError(
                f"max_width must be a positive finite value, got {max_width!r}."
            )
        self._cached_opening_fraction = 0.0
        self._is_open_flag = False
        self._controller: Any | None = None

        try:
            from ominipicker_controller import OmniPickerGripperController
        except (ImportError, OSError) as exc:
            raise ImportError(
                "OmniPicker support requires the 'ominipicker-controller' "
                "vendor wheel, including its platform-specific CAN-FD library."
            ) from exc

        controller = OmniPickerGripperController(
            tx_id=int(tx_id),
            rx_id=int(rx_id),
            channel_index=int(channel_index),
            is_extended_id=bool(is_extended_id),
            accept_all_rx_ids=bool(accept_all_rx_ids),
            rx_stale_timeout_s=float(rx_stale_timeout_s),
            online_probe_interval_s=float(online_probe_interval_s),
            rx_fail_probe_threshold=int(rx_fail_probe_threshold),
            disconnect_offline_threshold=int(disconnect_offline_threshold),
        )
        self._controller = controller
        try:
            controller.open_device()
            controller.init_channel()
            controller.start_threads()
        except Exception:
            try:
                self.cleanup()
            except Exception as cleanup_exc:  # pragma: no cover - defensive
                self._logger.warning(
                    "OmniPicker cleanup after initialization failure failed: %s",
                    cleanup_exc,
                )
            raise

        self._logger.info(
            "OmniPicker gripper initialized on right CAN channel %d (tx=0x%X, rx=0x%X)",
            channel_index,
            tx_id,
            rx_id,
        )

    @staticmethod
    def _clip_pct(value: float) -> float:
        """Clamp a normalised command value to ``[0, 1]``."""
        return float(np.clip(float(value), 0.0, 1.0))

    def _send(
        self,
        position_pct: float,
        *,
        vel_pct: float,
        force_pct: float,
    ) -> None:
        if self._controller is None:
            raise RuntimeError("OmniPicker gripper has already been cleaned up.")
        position_pct = self._clip_pct(position_pct)
        self._controller.send_command(
            pos_pct=position_pct,
            vel_pct=self._clip_pct(vel_pct),
            force_pct=self._clip_pct(force_pct),
            acc_pct=self._acc_pct,
            dec_pct=self._dec_pct,
        )
        self._cached_opening_fraction = position_pct
        self._is_open_flag = position_pct >= self._open_threshold

    def command_position(self, position_pct: float) -> None:
        """Send a normalised target using the configured five command gains.

        Args:
            position_pct: Opening fraction, where ``0`` is closed and ``1`` is
                open.
        """
        self._send(
            position_pct,
            vel_pct=self._vel_pct,
            force_pct=self._force_pct,
        )

    def command_rlinf_action(self, action: float) -> bool:
        """Map and send one RLinf scalar gripper action.

        Args:
            action: RLinf gripper action in ``[-1, 1]``. ``-1`` maps to fully
                closed and ``1`` maps to fully open.

        Returns:
            Whether the command crossed the configured open/closed threshold.
        """
        position_pct = self.rlinf_action_to_position_pct(action)
        was_open = self._is_open_flag
        self.command_position(position_pct)
        return was_open != self._is_open_flag

    @staticmethod
    def rlinf_action_to_position_pct(action: float) -> float:
        """Convert an RLinf action from ``[-1, 1]`` to ``pos_pct``."""
        value = float(np.clip(float(action), -1.0, 1.0))
        return (value + 1.0) / 2.0

    def get_status(self) -> dict[str, Any]:
        """Return the latest vendor status snapshot for diagnostics."""
        return self._read_state()

    def open(self, speed: float = 0.3) -> None:
        """Fully open the gripper."""
        self._send(1.0, vel_pct=speed, force_pct=self._force_pct)

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        """Fully close the gripper."""
        force_pct = min(force / 255.0, 1.0) if force > 1.0 else force
        self._send(0.0, vel_pct=speed, force_pct=force_pct)

    def move(self, position: float, speed: float = 0.3) -> None:
        """Move using RLinf's legacy ``0=open, 255=closed`` convention."""
        position_pct = 1.0 - float(np.clip(position, 0.0, 255.0)) / 255.0
        self._send(
            position_pct,
            vel_pct=speed,
            force_pct=self._force_pct,
        )

    def _read_state(self) -> dict[str, Any]:
        if self._controller is None:
            return {}
        try:
            state = self._controller.get_state()
        except Exception as exc:
            self._logger.warning("OmniPicker state read failed: %s", exc)
            return {}
        return state if isinstance(state, dict) else {}

    @property
    def position(self) -> float:
        """Current opening width in metres (consistent with other backends)."""
        state = self._read_state()
        value = state.get("position")
        if value is not None:
            try:
                self._cached_opening_fraction = self._clip_pct(float(value))
            except (TypeError, ValueError):
                pass
        return self._max_width * self._cached_opening_fraction

    @property
    def is_open(self) -> bool:
        return self._is_open_flag

    def is_ready(self) -> bool:
        """Return whether the CAN transport is alive and not disconnected."""
        state = self._read_state()
        if not state or state.get("disconnected"):
            return False
        sender_alive = bool(state.get("sender_alive"))
        receiver_alive = bool(state.get("receiver_alive"))
        device_online = bool(state.get("device_online") or state.get("online"))
        return sender_alive and receiver_alive and device_online

    def cleanup(self) -> None:
        """Stop vendor threads and release the CAN channel/device handle."""
        controller = self._controller
        self._controller = None
        if controller is not None:
            controller.cleanup()
