# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FrankaRobotState:
    """Full state of the Franka robot arm and its end-effector.

    The state covers the arm kinematics (pose, velocity, force/torque)
    as well as the end-effector.  For the built-in Franka gripper the
    scalar ``gripper_position`` / ``gripper_open`` fields are used.  For
    dexterous hands such as Ruiyan, the 6-D ``hand_position`` array is
    used instead.
    """

    # https://docs.ros.org/en/kinetic/api/libfranka/html/structfranka_1_1RobotState.html
    tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.O_T_EE
    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    arm_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.q
    arm_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.dq
    tcp_force: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # FrankaState.K_F_ext_hat_K[0:3]
    tcp_torque: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # FrankaState.K_F_ext_hat_K[3:6]
    arm_jacobian: np.ndarray = field(
        default_factory=lambda: np.zeros((6, 7))
    )  # ZeroJacobian.zero_jacobian

    # -- Parallel gripper ------------------------------------------------
    gripper_position: float = 0.0
    gripper_open: bool = False

    # -- Dexterous hand --------------------------------------------------
    hand_position: Optional[np.ndarray] = None  # 6-D normalised [0, 1]

    def to_dict(self):
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)
