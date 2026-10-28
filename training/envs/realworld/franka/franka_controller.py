# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from typing import Optional

import numpy as np
import psutil
from scipy.spatial.transform import Rotation as R

from training.envs.realworld.common.process_utils import (
    start_process_group,
    stop_process_group,
)
from training.scheduler import Cluster, NodePlacementStrategy, Worker
from training.utils.logging import get_logger

from .end_effectors import (
    EndEffector,
    EndEffectorType,
    normalize_end_effector_type,
)
from .franka_robot_state import FrankaRobotState


class FrankaController(Worker):
    """Franka robot arm controller."""

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        node_group_label: Optional[str] = None,
        worker_rank: int = 0,
        ros_pkg: str = "serl_franka_controllers",
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        """Launch a FrankaController on the specified worker's node."""
        cluster = Cluster()
        placement_node_rank = node_rank
        if node_group_label is not None:
            node_group = cluster.get_node_group(node_group_label)
            if node_group is None or node_rank not in node_group.node_ranks:
                raise ValueError(
                    f"Controller node rank {node_rank} is not in node group "
                    f"{node_group_label!r}."
                )
            placement_node_rank = node_group.node_ranks.index(node_rank)
        placement = NodePlacementStrategy(
            node_ranks=[placement_node_rank],
            node_group_label=node_group_label,
        )
        return FrankaController.create_group(
            robot_ip,
            ros_pkg,
            end_effector_type,
            end_effector_config or {},
            gripper_type,
            gripper_connection,
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"FrankaController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        ros_pkg: str = "serl_franka_controllers",
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        super().__init__()
        self._logger = get_logger()
        # A remote arm's IP may not reach the env worker; resolve it from this
        # node's hardware infos instead.
        if not robot_ip:
            robot_ip = self._resolve_robot_ip_from_node()
        if not robot_ip:
            raise ValueError(
                "Franka 'robot_ip' is not set and could not be resolved from "
                f"node rank {self._cluster_node_rank}'s hardware infos. Provide "
                "it in the env config, the Franka hardware config, or set the "
                "'ROBOT_IP' environment variable on the controller's node."
            )
        self._robot_ip = robot_ip
        self._ros_pkg = ros_pkg
        self._end_effector_type = normalize_end_effector_type(
            end_effector_type,
            gripper_type,
        )

        # Lazy-import ROS packages so the module can be imported on non-ROS nodes.
        import geometry_msgs.msg as geom_msg
        import rospy
        from dynamic_reconfigure.client import Client as ReconfClient
        from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
        from serl_franka_controllers.msg import ZeroJacobian

        self._geom_msg = geom_msg
        self._rospy = rospy
        self._ErrorRecoveryActionGoal = ErrorRecoveryActionGoal
        self._FrankaState = FrankaState
        self._ZeroJacobian = ZeroJacobian
        self._ReconfClient = ReconfClient

        self._state = FrankaRobotState()
        self._end_effector: EndEffector | None = None
        self._gripper = None
        self._impedance: psutil.Popen | None = None
        self._joint: psutil.Popen | None = None
        self._cleaned_up = False

        from training.envs.realworld.common.ros import ROSController

        self._ros = None
        try:
            self._ros = ROSController()
            self._init_ros_channels()
            self._init_end_effector(end_effector_config or {}, gripper_connection)

            self.start_impedance()
            self._reconf_client = self._ReconfClient(
                "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
            )
        except Exception:
            self.cleanup()
            raise

    def _resolve_robot_ip_from_node(self) -> Optional[str]:
        """Return the first ``robot_ip`` in this node's hardware infos, if any.

        The controller's node enumerates the arm and resolves ``robot_ip`` from
        its local ``ROBOT_IP``, so the value is available here even when the env
        worker on another node could not detect it.
        """
        try:
            node_info = Cluster().get_node_info(self._cluster_node_rank)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.warning(
                "Could not access node info to resolve robot_ip: %s", exc
            )
            return None
        for resource in node_info.hardware_resources:
            for info in resource.infos:
                robot_ip = getattr(getattr(info, "config", None), "robot_ip", None)
                if robot_ip:
                    return robot_ip
        return None

    def _init_end_effector(
        self,
        end_effector_config: dict,
        gripper_connection: Optional[str],
    ) -> None:
        from training.envs.realworld.common.gripper import create_gripper

        self._gripper = create_gripper(
            gripper_type=self._end_effector_type.gripper_backend,
            ros=self._ros,
            port=gripper_connection,
            **end_effector_config,
        )
        self._logger.info(
            "Gripper initialised: end_effector=%s",
            self._end_effector_type.value,
        )
        return

    def _init_ros_channels(self):
        """Initialize ROS channels for arm communication."""
        self._arm_equilibrium_channel = (
            "/cartesian_impedance_controller/equilibrium_pose"
        )
        self._arm_reset_channel = "/franka_control/error_recovery/goal"
        self._arm_jacobian_channel = "/cartesian_impedance_controller/franka_jacobian"
        self._arm_state_channel = "franka_state_controller/franka_states"

        self._ros.create_ros_channel(
            self._arm_equilibrium_channel,
            self._geom_msg.PoseStamped,
            queue_size=10,
        )
        self._ros.create_ros_channel(
            self._arm_reset_channel,
            self._ErrorRecoveryActionGoal,
            queue_size=1,
        )
        self._ros.connect_ros_channel(
            self._arm_jacobian_channel,
            self._ZeroJacobian,
            self._on_arm_jacobian_msg,
        )
        self._ros.connect_ros_channel(
            self._arm_state_channel,
            self._FrankaState,
            self._on_arm_state_msg,
        )

    def _on_arm_jacobian_msg(self, msg):
        self._state.arm_jacobian = np.array(list(msg.zero_jacobian)).reshape(
            (6, 7), order="F"
        )

    def _on_arm_state_msg(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3].copy())
        self._state.tcp_pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])

        self._state.arm_joint_velocity = np.array(list(msg.dq)).reshape((7,))
        self._state.arm_joint_position = np.array(list(msg.q)).reshape((7,))
        self._state.tcp_force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self._state.tcp_torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        try:
            self._state.tcp_vel = (
                self._state.arm_jacobian @ self._state.arm_joint_velocity
            )
        except Exception as exc:
            self._state.tcp_vel = np.zeros(6)
            self._logger.warning(
                "Jacobian not set, end-effector velocity temporarily unavailable: %s",
                exc,
            )

    def reconfigure_compliance_params(self, params: dict[str, float]):
        self._reconf_client.update_configuration(params)
        self.log_debug(f"Reconfigure compliance parameters: {params}")

    def is_robot_up(self) -> bool:
        """Check whether the arm and active end-effector are ready."""
        arm_ok = self._ros.get_input_channel_status(self._arm_state_channel)
        return arm_ok and self._gripper.is_ready()

    def get_state(self) -> FrankaRobotState:
        """Get the current state of the Franka robot."""
        self._state.gripper_position = self._gripper.position
        self._state.gripper_open = self._gripper.is_open
        self._state.hand_position = None
        return self._state

    def start_impedance(self):
        """Start the impedance controller."""
        load_gripper = (
            "true"
            if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER
            else "false"
        )
        self._impedance = start_process_group(
            [
                "roslaunch",
                self._ros_pkg,
                "impedance.launch",
                "robot_ip:=" + self._robot_ip,
                f"load_gripper:={load_gripper}",
            ],
            stdout=sys.stdout,
            stderr=sys.stdout,
        )

        self._wait_robot()
        self.log_debug(f"Start Impedance controller: {self._impedance.status()}")

    def stop_impedance(self):
        if self._stop_controller_process("_impedance", "impedance controller"):
            self._wait_robot()
        self.log_debug("Stop Impedance controller")

    def _stop_controller_process(self, attribute: str, name: str) -> bool:
        """Stop one owned ROS launch tree and clear its process handle."""
        process = getattr(self, attribute, None)
        if process is None:
            return False
        setattr(self, attribute, None)
        stop_process_group(process, name=name, logger=self._logger)
        return True

    def clear_errors(self):
        self._ros.put_channel(self._arm_reset_channel, self._ErrorRecoveryActionGoal())

    def reset_joint(self, reset_pos: list[float]):
        """Reset the joint positions of the robot arm."""
        self.stop_impedance()
        self.clear_errors()
        self._wait_robot()
        self.clear_errors()

        assert len(reset_pos) == 7, (
            f"Invalid reset position, expected 7 dimensions but got {len(reset_pos)}"
        )

        load_gripper = (
            "true"
            if self._end_effector_type == EndEffectorType.FRANKA_GRIPPER
            else "false"
        )
        self._rospy.set_param("/target_joint_positions", reset_pos)
        try:
            self._joint = start_process_group(
                [
                    "roslaunch",
                    self._ros_pkg,
                    "joint.launch",
                    "robot_ip:=" + self._robot_ip,
                    f"load_gripper:={load_gripper}",
                ],
                stdout=sys.stdout,
            )
            self._wait_robot()
            self._logger.debug("Joint reset begins")
            self.clear_errors()
            self._wait_for_joint(reset_pos)
        finally:
            self._stop_controller_process("_joint", "joint controller")
            self._wait_robot()
            self.clear_errors()
            self.start_impedance()

    def move_arm(self, position: np.ndarray):
        """Move the robot arm to the desired position."""
        assert len(position) == 7, (
            f"Invalid position, expected 7 dimensions but got {len(position)}"
        )
        pose_msg = self._geom_msg.PoseStamped()
        pose_msg.header.frame_id = "0"
        pose_msg.header.stamp = self._rospy.Time.now()
        pose_msg.pose.position = self._geom_msg.Point(
            position[0], position[1], position[2]
        )
        pose_msg.pose.orientation = self._geom_msg.Quaternion(
            position[3], position[4], position[5], position[6]
        )

        self._ros.put_channel(self._arm_equilibrium_channel, pose_msg)
        self.log_debug(f"Move arm to position: {position}")

    def command_end_effector(self, action: np.ndarray) -> bool:
        """Send an action to the active end-effector."""
        if self._end_effector_type == EndEffectorType.OMNIPICKER_GRIPPER:
            value = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            changed = self._gripper.command_rlinf_action(value)
            self._state.gripper_open = self._gripper.is_open
            return changed

        value = float(np.asarray(action).reshape(-1)[0])
        if value <= -0.5 and self._state.gripper_open:
            self.close_gripper()
            return True
        if value >= 0.5 and not self._state.gripper_open:
            self.open_gripper()
            return True
        return False

    def reset_end_effector(self, target_state: np.ndarray | None = None) -> None:
        """Reset the end-effector to a target or default state."""
        if target_state is not None:
            self.command_end_effector(np.asarray(target_state))
        return

    def open_gripper(self):
        if self._end_effector_type == EndEffectorType.OMNIPICKER_GRIPPER:
            self._gripper.command_position(1.0)
        else:
            self._gripper.open()
        self._state.gripper_open = True
        self.log_debug("Open gripper")

    def close_gripper(self):
        if self._end_effector_type == EndEffectorType.OMNIPICKER_GRIPPER:
            self._gripper.command_position(0.0)
        else:
            self._gripper.close()
        self._state.gripper_open = False
        self.log_debug("Close gripper")

    def move_gripper(self, position: int, speed: float = 0.3):
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        self._gripper.move(position, speed)
        self.log_debug(f"Move gripper to position: {position}")

    def _wait_robot(self, sleep_time: int = 1):
        time.sleep(sleep_time)

    def _wait_for_joint(self, target_pos: list[float], timeout: int = 30):
        wait_time = 0.01
        waited_time = 0.0
        target_pos = np.array(target_pos)

        while (
            not np.allclose(
                target_pos,
                self._state.arm_joint_position,
                atol=1e-2,
                rtol=1e-2,
            )
            and waited_time < timeout
        ):
            time.sleep(wait_time)
            waited_time += wait_time

        if waited_time >= timeout:
            self._logger.warning("Joint position wait timeout exceeded")
        else:
            self._logger.debug(
                f"Joint position reached {self._state.arm_joint_position}"
            )

    def cleanup(self) -> None:
        """Release all ROS, controller, and end-effector resources once."""
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True

        cleanup_errors: list[tuple[str, Exception]] = []

        def run_cleanup(name: str, callback) -> None:
            try:
                callback()
            except Exception as exc:  # Continue releasing independent resources.
                self._logger.warning("Failed to clean up %s: %s", name, exc)
                cleanup_errors.append((name, exc))

        run_cleanup(
            "joint controller",
            lambda: self._stop_controller_process("_joint", "joint controller"),
        )
        run_cleanup(
            "impedance controller",
            lambda: self._stop_controller_process("_impedance", "impedance controller"),
        )

        gripper = getattr(self, "_gripper", None)
        self._gripper = None
        if gripper is not None:
            run_cleanup("gripper", gripper.cleanup)

        end_effector = getattr(self, "_end_effector", None)
        self._end_effector = None
        if end_effector is not None:
            run_cleanup("end-effector", end_effector.shutdown)

        ros = getattr(self, "_ros", None)
        self._ros = None
        if ros is not None:
            run_cleanup("ROS controller", ros.close)

        if cleanup_errors:
            names = ", ".join(name for name, _ in cleanup_errors)
            self._logger.error("Cleanup completed with failures: %s", names)
