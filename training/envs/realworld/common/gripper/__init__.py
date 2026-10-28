# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

from typing import Optional

from .base_gripper import BaseGripper

__all__ = [
    "BaseGripper",
    "create_gripper",
]


def create_gripper(
    gripper_type: str = "franka",
    ros=None,
    port: Optional[str] = None,
    **kwargs,
) -> BaseGripper:
    """Factory that instantiates the right gripper backend.

    Args:
        gripper_type: ``"franka"`` (ROS-based), ``"robotiq"`` (Modbus RTU),
            or ``"omnipicker"`` (USB-CAN-FD).
        ros: :class:`ROSController` instance — required for ``"franka"``.
        port: Serial device path (e.g. ``"/dev/ttyUSB0"``) — required for
            ``"robotiq"``.
        **kwargs: Forwarded to the gripper constructor (e.g. ``max_width``,
            ``baudrate``, ``slave_id``).
    """
    gt = gripper_type.lower()
    if gt == "robotiq":
        if port is None:
            raise ValueError(
                "gripper_connection (serial port) must be specified "
                "for Robotiq grippers."
            )
        from .robotiq_gripper import RobotiqGripper

        return RobotiqGripper(port=port, **kwargs)
    if gt in ("omnipicker", "ominipicker"):
        from .omnipicker_gripper import OmniPickerGripper

        return OmniPickerGripper(**kwargs)
    if gt == "franka":
        if ros is None:
            raise ValueError(
                "ROSController instance must be provided for Franka gripper."
            )
        from .franka_gripper import FrankaGripper

        return FrankaGripper(ros=ros, **kwargs)
    raise ValueError(
        f"Unsupported gripper_type={gripper_type!r}. "
        f"Supported types: 'franka', 'robotiq', 'omnipicker'."
    )
