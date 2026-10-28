# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

from .base_camera import BaseCamera, CameraInfo
from .realsense_camera import RealSenseCamera

__all__ = [
    "BaseCamera",
    "CameraInfo",
    "RealSenseCamera",
    "create_camera",
]


def create_camera(camera_info: CameraInfo) -> BaseCamera:
    """Factory that instantiates the right camera backend from *camera_info*.

    Supported ``camera_info.camera_type`` values:

    * ``"realsense"`` / ``"rs"`` — Intel RealSense (requires ``pyrealsense2``)
    * ``"zed"`` — Stereolabs ZED (requires the ZED SDK / ``pyzed``)
    * ``"lumos"`` — LUMOS V4L2 USB camera (requires ``opencv-python``)
    """
    camera_type = camera_info.camera_type.lower()
    if camera_type == "zed":
        from .zed_camera import ZEDCamera

        return ZEDCamera(camera_info)
    if camera_type in ("realsense", "rs"):
        return RealSenseCamera(camera_info)
    if camera_type == "lumos":
        from .lumos_camera import LumosCamera

        return LumosCamera(camera_info)
    raise ValueError(
        f"Unsupported camera_type={camera_type!r}. Supported types: 'realsense', 'zed', 'lumos'."
    )
