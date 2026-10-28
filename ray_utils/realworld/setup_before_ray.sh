#!/usr/bin/env bash
# Modified for HIL-UMI.
# Source this helper before starting Ray so workers inherit your ROS environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script from Bash after selecting the robot-control environment." >&2
    exit 2
fi
if [[ ! "${TRAINING_NODE_RANK:-}" =~ ^[0-9]+$ ]]; then
    echo "Set TRAINING_NODE_RANK explicitly before sourcing this helper." >&2
    return 2
fi
for setup_file in "${ROS_NOETIC_SETUP:-/opt/ros/noetic/setup.bash}" "${FRANKA_CATKIN_SETUP:-}"; do
    if [[ ! -f "$setup_file" ]]; then
        echo "Set ROS_NOETIC_SETUP and FRANKA_CATKIN_SETUP to installed setup files." >&2
        return 2
    fi
    source "$setup_file"
done
