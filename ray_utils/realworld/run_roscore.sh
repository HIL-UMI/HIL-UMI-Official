#!/usr/bin/env bash
# Modified for HIL-UMI.

# Run the ROS 1 core services without roslaunch's fixed master-start deadline.
# On a busy deployment host, stock `roscore` can kill a healthy rosmaster before
# its XML-RPC thread becomes responsive. Starting and probing the master first
# avoids that race while providing the same master, parameter server, and rosout
# services as roscore.

set -Eeuo pipefail

ROS_NOETIC_SETUP="${ROS_NOETIC_SETUP:-/opt/ros/noetic/setup.bash}"
FRANKA_CATKIN_SETUP="${FRANKA_CATKIN_SETUP:-${HOME}/catkin_ws_franka/devel/setup.bash}"

set +u
# shellcheck disable=SC1090
source "${ROS_NOETIC_SETUP}"
# shellcheck disable=SC1090
source "${FRANKA_CATKIN_SETUP}"
set -u

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-127.0.0.1}"
unset ROS_IP

for command_name in rosmaster rosparam rosrun rosversion uuidgen; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: ${command_name} is required to run the ROS core." >&2
        exit 1
    fi
done

ROS_MASTER_PORT="$(
    /usr/bin/python3 -c \
        'import os; from urllib.parse import urlparse; uri = urlparse(os.environ["ROS_MASTER_URI"]); assert uri.hostname in {"127.0.0.1", "localhost", "::1"}, f"ROS master is not local: {uri.hostname}"; print(uri.port or 11311)'
)"
ROSMASTER_NUM_WORKERS="${TRAINING_ROSMASTER_NUM_WORKERS:-3}"
MASTER_PID=""
ROSOUT_PID=""

master_is_online() {
    timeout 2 /usr/bin/python3 -c \
        'import socket; socket.setdefaulttimeout(1); import rosgraph; raise SystemExit(0 if rosgraph.is_master_online() else 1)' \
        >/dev/null 2>&1
}

rosout_is_registered() {
    timeout 2 /usr/bin/python3 -c \
        'import socket, xmlrpc.client; socket.setdefaulttimeout(1); master = xmlrpc.client.ServerProxy(__import__("os").environ["ROS_MASTER_URI"]); state = master.getSystemState("/rlinf_core_probe")[2]; nodes = {node for group in state for _topic, names in group for node in names}; raise SystemExit(0 if "/rosout" in nodes else 1)' \
        >/dev/null 2>&1
}

stop_children() {
    trap - EXIT INT TERM
    for child_pid in "${ROSOUT_PID}" "${MASTER_PID}"; do
        if [ -n "${child_pid}" ] && kill -0 "${child_pid}" 2>/dev/null; then
            kill "${child_pid}" 2>/dev/null || true
        fi
    done
    for child_pid in "${ROSOUT_PID}" "${MASTER_PID}"; do
        if [ -n "${child_pid}" ]; then
            wait "${child_pid}" 2>/dev/null || true
        fi
    done
}
trap stop_children EXIT INT TERM

if master_is_online; then
    echo "ERROR: a ROS master is already running at ${ROS_MASTER_URI}." >&2
    exit 1
fi

echo "Starting ROS master at ${ROS_MASTER_URI}."
rosmaster --core -p "${ROS_MASTER_PORT}" -w "${ROSMASTER_NUM_WORKERS}" &
MASTER_PID=$!

for _ in $(seq 1 120); do
    if master_is_online; then
        break
    fi
    if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
        wait "${MASTER_PID}" || true
        echo "ERROR: ROS master exited before becoming ready." >&2
        exit 1
    fi
    sleep 0.5
done
if ! master_is_online; then
    echo "ERROR: ROS master did not become ready within 60 seconds." >&2
    exit 1
fi

# These are the parameters created by /opt/ros/noetic/etc/ros/roscore.xml.
rosparam set /run_id "${TRAINING_ROS_RUN_ID:-$(uuidgen)}"
rosparam set /rosversion "$(rosversion roslaunch)"
rosparam set /rosdistro "$(rosversion -d)"

start_rosout() {
    echo "Starting /rosout."
    rosrun rosout rosout __name:=rosout &
    ROSOUT_PID=$!
}
start_rosout
for _ in $(seq 1 60); do
    if rosout_is_registered; then
        break
    fi
    if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
        echo "ERROR: ROS master exited while /rosout was starting." >&2
        exit 1
    fi
    if ! kill -0 "${ROSOUT_PID}" 2>/dev/null; then
        wait "${ROSOUT_PID}" || true
        echo "WARNING: /rosout exited during startup; restarting it." >&2
        start_rosout
    fi
    sleep 0.5
done
if ! rosout_is_registered; then
    echo "ERROR: /rosout did not register within 30 seconds." >&2
    exit 1
fi
echo "ROS core is ready at ${ROS_MASTER_URI} (master pid=${MASTER_PID})."

# roscore declares rosout with respawn=true. Match that behavior, but stop the
# whole service if the master itself exits.
while kill -0 "${MASTER_PID}" 2>/dev/null; do
    if ! kill -0 "${ROSOUT_PID}" 2>/dev/null; then
        wait "${ROSOUT_PID}" || true
        echo "WARNING: /rosout exited; restarting it." >&2
        start_rosout
    fi
    sleep 1
done

wait "${MASTER_PID}" || MASTER_STATUS=$?
echo "ERROR: ROS master exited with status ${MASTER_STATUS:-0}." >&2
exit "${MASTER_STATUS:-1}"
