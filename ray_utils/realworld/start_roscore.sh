#!/usr/bin/env bash
# Modified for HIL-UMI.

# Start one persistent local ROS 1 master and wait until it is responsive.
# This script is intentionally executed by bash so callers using zsh do not
# need to source ROS's setup.bash themselves.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"

ROS_NOETIC_SETUP="${ROS_NOETIC_SETUP:-/opt/ros/noetic/setup.bash}"
FRANKA_CATKIN_SETUP="${FRANKA_CATKIN_SETUP:-${HOME}/catkin_ws_franka/devel/setup.bash}"
for setup_file in "${ROS_NOETIC_SETUP}" "${FRANKA_CATKIN_SETUP}"; do
    if [ ! -f "${setup_file}" ]; then
        echo "ERROR: required ROS setup file does not exist: ${setup_file}" >&2
        exit 1
    fi
done

set +u
# shellcheck disable=SC1090
source "${ROS_NOETIC_SETUP}"
# shellcheck disable=SC1090
source "${FRANKA_CATKIN_SETUP}"
set -u

# This deployment runs every ROS node on one machine. Explicit loopback
# addresses keep roscore and local clients on the same interface.
export ROS_MASTER_URI="${TRAINING_ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_HOSTNAME="${TRAINING_ROS_HOSTNAME:-127.0.0.1}"
unset ROS_IP

if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to start roscore safely." >&2
    exit 1
fi
ROSCORE_SUPERVISOR="${SCRIPT_DIR}/run_roscore.sh"
if [ ! -x "${ROSCORE_SUPERVISOR}" ]; then
    echo "ERROR: ROS core supervisor is not executable: ${ROSCORE_SUPERVISOR}" >&2
    exit 1
fi

master_is_online() {
    timeout 2 /usr/bin/python3 -c \
        'import socket; socket.setdefaulttimeout(1); import rosgraph; raise SystemExit(0 if rosgraph.is_master_online() else 1)' \
        >/dev/null 2>&1
}

core_is_online() {
    timeout 2 /usr/bin/python3 -c \
        'import socket, xmlrpc.client; socket.setdefaulttimeout(1); master = xmlrpc.client.ServerProxy(__import__("os").environ["ROS_MASTER_URI"]); state = master.getSystemState("/rlinf_core_probe")[2]; nodes = {node for group in state for _topic, names in group for node in names}; raise SystemExit(0 if "/rosout" in nodes else 1)' \
        >/dev/null 2>&1
}

# Serialize the check-and-start sequence so concurrent launchers cannot race
# and create two masters on port 11311.
exec 9>/tmp/training-roscore-start.lock
flock 9

if master_is_online; then
    # An existing core may be between starting the master and registering
    # /rosout. Wait for that startup instead of racing it with another core.
    for _ in $(seq 1 60); do
        if core_is_online; then
            echo "ROS core is already ready at ${ROS_MASTER_URI}."
            exit 0
        fi
        if ! master_is_online; then
            break
        fi
        sleep 0.5
    done
    if master_is_online; then
        echo "ERROR: a ROS master is online at ${ROS_MASTER_URI}, but /rosout did not register within 30 seconds." >&2
        exit 1
    fi
fi

ROSCORE_LOG_DIR="${TRAINING_ROSCORE_LOG_DIR:-${REPO_PATH}/logs/roscore}"
mkdir -p "${ROSCORE_LOG_DIR}"
ROSCORE_LOG_FILE="${ROSCORE_LOG_DIR}/roscore-$(date +'%Y%m%d-%H%M%S').log"
ROSCORE_SYSTEMD_UNIT="training-roscore.service"
ROSCORE_START_MODE="direct"
ROSCORE_PID=""
ROSCORE_EXISTING_SERVICE=false

# Prefer the user service manager so the ROS core stays outside the evaluation
# process group. run_roscore.sh starts rosmaster before rosout and avoids
# roslaunch's fixed master-start deadline. Fall back to a detached child on
# systems that do not provide a user manager.
if command -v systemd-run >/dev/null 2>&1 && \
    systemctl --user show-environment >/dev/null 2>&1; then
    if systemctl --user --quiet is-active "${ROSCORE_SYSTEMD_UNIT}"; then
        ROSCORE_START_MODE="systemd"
        ROSCORE_EXISTING_SERVICE=true
    else
        systemctl --user reset-failed "${ROSCORE_SYSTEMD_UNIT}" >/dev/null 2>&1 || true
        # The service's bash process expands the variables in its command string.
        # shellcheck disable=SC2016
        if systemd-run \
            --user \
            --quiet \
            --collect \
            --unit="${ROSCORE_SYSTEMD_UNIT%.service}" \
            --property=Nice=0 \
            --setenv="ROS_MASTER_URI=${ROS_MASTER_URI}" \
            --setenv="ROS_HOSTNAME=${ROS_HOSTNAME}" \
            --setenv="ROS_NOETIC_SETUP=${ROS_NOETIC_SETUP}" \
            --setenv="FRANKA_CATKIN_SETUP=${FRANKA_CATKIN_SETUP}" \
            --setenv="TRAINING_ROSCORE_LOG_FILE=${ROSCORE_LOG_FILE}" \
            --setenv="TRAINING_ROSCORE_SUPERVISOR=${ROSCORE_SUPERVISOR}" \
            /bin/bash -c \
            'exec /bin/bash "${TRAINING_ROSCORE_SUPERVISOR}" >>"${TRAINING_ROSCORE_LOG_FILE}" 2>&1'; then
            ROSCORE_START_MODE="systemd"
        fi
    fi
fi

if [ "${ROSCORE_START_MODE}" = "direct" ]; then
    nohup setsid /bin/bash "${ROSCORE_SUPERVISOR}" >"${ROSCORE_LOG_FILE}" 2>&1 </dev/null &
    ROSCORE_PID=$!
fi

for _ in $(seq 1 60); do
    if core_is_online; then
        if [ "${ROSCORE_START_MODE}" = "systemd" ]; then
            ROSCORE_PID="$(systemctl --user show --property=MainPID --value "${ROSCORE_SYSTEMD_UNIT}")"
        fi
        echo "ROS core is ready at ${ROS_MASTER_URI} (pid=${ROSCORE_PID})."
        if [ "${ROSCORE_EXISTING_SERVICE}" = false ]; then
            echo "ROS core log: ${ROSCORE_LOG_FILE}"
        fi
        exit 0
    fi
    if [ "${ROSCORE_START_MODE}" = "systemd" ]; then
        if ! systemctl --user --quiet is-active "${ROSCORE_SYSTEMD_UNIT}"; then
            echo "ERROR: ${ROSCORE_SYSTEMD_UNIT} exited before ROS became ready." >&2
            echo "Last roscore log lines:" >&2
            tail -n 40 "${ROSCORE_LOG_FILE}" >&2 || true
            exit 1
        fi
    elif ! kill -0 "${ROSCORE_PID}" 2>/dev/null; then
        echo "ERROR: roscore exited before becoming ready." >&2
        echo "Last roscore log lines:" >&2
        tail -n 40 "${ROSCORE_LOG_FILE}" >&2 || true
        exit 1
    fi
    sleep 0.5
done

echo "ERROR: ROS core did not become ready within 30 seconds." >&2
echo "Last roscore log lines:" >&2
tail -n 40 "${ROSCORE_LOG_FILE}" >&2 || true
if [ "${ROSCORE_START_MODE}" = "systemd" ]; then
    systemctl --user stop "${ROSCORE_SYSTEMD_UNIT}" >/dev/null 2>&1 || true
elif [ -n "${ROSCORE_PID}" ]; then
    kill -- "-${ROSCORE_PID}" >/dev/null 2>&1 || true
fi
exit 1
