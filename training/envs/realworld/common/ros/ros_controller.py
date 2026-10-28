# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys
import time
from typing import Callable, Optional

import rosgraph
import rospy
from filelock import FileLock

from training.envs.realworld.common.process_utils import (
    start_process_group,
    stop_process_group,
)
from training.utils.logging import get_logger


class ROSController:
    """Controller for ROS communication. A controller is used for managing one robot."""

    def __init__(self, ros_version: int = 1):
        """Initialize the ROS controller."""
        self._logger = get_logger()
        self._ros_version = ros_version
        assert self._ros_version == 1, "Currently only ROS 1 is supported."
        self._ros_core = None
        self._owns_ros_core = False
        self._closed = False

        # ROS is a global service on the node
        # When there are multiple controllers, concurrency control is needed
        ros_lock_file = "/tmp/.ros.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(ros_lock_file)):
            ros_lock_file = os.path.join(pathlib.Path.home(), ".ros.lock")
        self._ros_lock = FileLock(ros_lock_file)

        try:
            if self._ros_version == 1:
                # roscore is removed in ROS 2
                with self._ros_lock:
                    if not rosgraph.is_master_online():
                        self._ros_core = start_process_group(
                            ["roscore"],
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                        )
                        self._owns_ros_core = True
                        if not self._wait_for_master():
                            raise RuntimeError(
                                "roscore did not become ready within 10s"
                            )

            # Initialize ros node
            rospy.init_node("franka_controller", anonymous=True)
        except Exception:
            self.close()
            raise

        # ROS channels
        self._output_channels: dict[str, rospy.Publisher] = {}
        self._input_channels: dict[str, rospy.Subscriber] = {}
        self._input_channel_status: dict[str, bool] = {}

    @staticmethod
    def _wait_for_master(timeout_s: float = 10.0) -> bool:
        """Wait until the ROS 1 master responds."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if rosgraph.is_master_online():
                return True
            time.sleep(0.1)
        return rosgraph.is_master_online()

    def get_input_channel_status(self, name: str) -> bool:
        """Get the status of a ROS input channel.

        Args:
            name: The name of the ROS input channel.

        Returns:
            bool: The status of the ROS input channel.
        """
        if name not in self._input_channel_status:
            return False
        return self._input_channel_status.get(name, False)

    def create_ros_channel(
        self, name: str, data_class: rospy.Message, queue_size: Optional[int] = None
    ):
        """Create a ROS Publisher channel for communication.

        Args:
            name: The name of the ROS channel.
            data_class: The message data class for the ROS channel.
            queue_size: The size of the queue for the ROS channel. Same as common channel, queue_size 0 means an infinite queue. However, queue_size being None means the channel becomes blocking.
        """
        self._output_channels[name] = rospy.Publisher(
            name, data_class, queue_size=queue_size
        )

    def connect_ros_channel(
        self, name: str, data_class: rospy.Message, callback: Callable
    ):
        """Connect a ROS Subscriber channel for communication.

        Args:
            name: The name of the ROS channel.
            data_class: The message data class for the ROS channel.
            callback: The callback function to handle incoming messages.
        """

        def callback_wrapper(*args, **kwargs):
            # When the callback is called, mark the channel as active
            self._input_channel_status[name] = True
            return callback(*args, **kwargs)

        self._input_channel_status[name] = False
        self._input_channels[name] = rospy.Subscriber(
            name, data_class, callback_wrapper
        )

    def put_channel(self, name: str, data: rospy.Message):
        """Put data into a ROS Publisher channel.

        Args:
            name: The name of the ROS channel.
            data: The data to publish on the ROS channel.
        """
        if name in self._output_channels:
            assert isinstance(data, self._output_channels[name].data_class), (
                f"Invalid data type for ROS channel '{name}'. Expected {self._output_channels[name].data_class}, got {type(data)}."
            )
            self._output_channels[name].publish(data)
        else:
            self._logger.warning(f"ROS channel '{name}' is not created.")

    def close(self) -> None:
        """Unregister channels and stop the roscore launched by this controller."""
        if self._closed:
            return
        self._closed = True

        input_channels = getattr(self, "_input_channels", {})
        output_channels = getattr(self, "_output_channels", {})
        for channel in (*input_channels.values(), *output_channels.values()):
            try:
                channel.unregister()
            except Exception as exc:  # pragma: no cover - rospy implementation detail
                self._logger.warning("Failed to unregister a ROS channel: %s", exc)
        input_channels.clear()
        output_channels.clear()
        getattr(self, "_input_channel_status", {}).clear()

        ros_core = getattr(self, "_ros_core", None)
        owns_ros_core = getattr(self, "_owns_ros_core", False)
        self._ros_core = None
        self._owns_ros_core = False
        if owns_ros_core:
            stop_process_group(ros_core, name="roscore", logger=self._logger)
