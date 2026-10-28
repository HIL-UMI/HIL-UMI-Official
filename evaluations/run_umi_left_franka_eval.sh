#! /bin/bash
# Modified for HIL-UMI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${SCRIPT_DIR}")"

MODEL_PATH=""
NORM_STATS_PATH="${UMI_LEFT_FRANKA_NORM_STATS:-}"
ROBOT_IP="${FRANKA_ROBOT_IP:-}"
WRIST_CAMERA="${FRANKA_WRIST_CAMERA_SERIAL:-}"
FRONT_CAMERA="${FRANKA_FRONT_CAMERA_SERIAL:-}"
KEYBOARD_DEVICE="${TRAINING_KEYBOARD_DEVICE:-/dev/input/by-id/KEYBOARD-event-kbd}"
# OPENPI_PYTHON is accepted only as a legacy interpreter-selector alias.
export POLICY_PYTHON="${POLICY_PYTHON:-${OPENPI_PYTHON:-${REPO_PATH}/.venv-policy/bin/python}}"
export FRANKA_PYTHON="${FRANKA_PYTHON:-${REPO_PATH}/.venv-franka/bin/python}"
DUMMY_MODE=false
DRY_RUN=false
SCENE_REFERENCE=""
SCENE_REFERENCE_ROOT="${TRAINING_SCENE_REFERENCE_ROOT:-${REPO_PATH}/assets/scene_references}"
SCENE_REFERENCE_DIR=""
SCENE_CACHE_DIR=""
SCENE_EPOCH_OFFSET="0"
SCENE_ALIGNMENT_HOST="127.0.0.1"
SCENE_ALIGNMENT_PORT="8765"
SCENE_REFERENCE_OPACITY="0.35"
SAVE_POSE_TRACE=false
POSE_TRACE_DIR=""
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 --model_path DIR [options]"
    echo "Options:"
    echo "  --robot_ip IP             Franka control IP (required unless --dummy)"
    echo "  --norm_stats_path DIR     Directory containing norm_stats.json"
    echo "  --wrist_camera SERIAL     Wrist camera serial (required unless --dummy)"
    echo "  --front_camera SERIAL     Front camera serial (required unless --dummy)"
    echo "  --keyboard_device PATH    Keyboard event device used for a/b/c controls"
    echo "  --policy_python PATH      Policy Python (default: .venv-policy)"
    echo "  --openpi_python PATH      Legacy alias for --policy_python"
    echo "  --franka_python PATH      Franka Python (default: .venv-franka)"
    echo "  --scene_reference NAME    Use a named reference preset, e.g. stack_cubes"
    echo "  --scene_reference_dir DIR Recorded seed_0 video directory for browser alignment"
    echo "  --scene_cache_dir DIR     Shared directory for saved reference images/metadata"
    echo "  --scene_epoch_offset N    Number local video 0 as global epoch N (default: 0)"
    echo "  --scene_alignment_host IP Browser UI bind host (default: 127.0.0.1)"
    echo "  --scene_alignment_port N  Browser UI port (default: 8765)"
    echo "  --scene_reference_opacity Alpha of the recorded frame (default: 0.35)"
    echo "  --save_pose_trace         Record async predicted/commanded/actual TCP poses"
    echo "  --pose_trace_dir DIR      Override the pose trace output directory"
    echo "  --dummy                   Run the full policy path without robot/cameras"
    echo "  --dry-run                 Print the command without ROS, devices, models or logs"
    echo "  remaining args            Forwarded as Hydra overrides"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model_path|--norm_stats_path|--robot_ip|--wrist_camera|--front_camera|--keyboard_device|--policy_python|--openpi_python|--franka_python|--scene_reference|--scene_reference_dir|--scene_cache_dir|--scene_epoch_offset|--scene_alignment_host|--scene_alignment_port|--scene_reference_opacity|--pose_trace_dir)
            [[ $# -ge 2 && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; } ;;
    esac
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model_path=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        --norm_stats_path)
            NORM_STATS_PATH="$2"
            shift 2
            ;;
        --norm_stats_path=*)
            NORM_STATS_PATH="${1#*=}"
            shift
            ;;
        --robot_ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        --robot_ip=*)
            ROBOT_IP="${1#*=}"
            shift
            ;;
        --wrist_camera)
            WRIST_CAMERA="$2"
            shift 2
            ;;
        --wrist_camera=*)
            WRIST_CAMERA="${1#*=}"
            shift
            ;;
        --front_camera)
            FRONT_CAMERA="$2"
            shift 2
            ;;
        --front_camera=*)
            FRONT_CAMERA="${1#*=}"
            shift
            ;;
        --keyboard_device)
            KEYBOARD_DEVICE="$2"
            shift 2
            ;;
        --keyboard_device=*)
            KEYBOARD_DEVICE="${1#*=}"
            shift
            ;;
        --policy_python|--openpi_python)
            export POLICY_PYTHON="$2"
            shift 2
            ;;
        --policy_python=*|--openpi_python=*)
            export POLICY_PYTHON="${1#*=}"
            shift
            ;;
        --franka_python)
            export FRANKA_PYTHON="$2"
            shift 2
            ;;
        --franka_python=*)
            export FRANKA_PYTHON="${1#*=}"
            shift
            ;;
        --scene_reference)
            SCENE_REFERENCE="$2"
            shift 2
            ;;
        --scene_reference=*)
            SCENE_REFERENCE="${1#*=}"
            shift
            ;;
        --scene_reference_dir)
            SCENE_REFERENCE_DIR="$2"
            shift 2
            ;;
        --scene_reference_dir=*)
            SCENE_REFERENCE_DIR="${1#*=}"
            shift
            ;;
        --scene_cache_dir)
            SCENE_CACHE_DIR="$2"
            shift 2
            ;;
        --scene_cache_dir=*)
            SCENE_CACHE_DIR="${1#*=}"
            shift
            ;;
        --scene_epoch_offset)
            SCENE_EPOCH_OFFSET="$2"
            shift 2
            ;;
        --scene_epoch_offset=*)
            SCENE_EPOCH_OFFSET="${1#*=}"
            shift
            ;;
        --scene_alignment_host)
            SCENE_ALIGNMENT_HOST="$2"
            shift 2
            ;;
        --scene_alignment_host=*)
            SCENE_ALIGNMENT_HOST="${1#*=}"
            shift
            ;;
        --scene_alignment_port)
            SCENE_ALIGNMENT_PORT="$2"
            shift 2
            ;;
        --scene_alignment_port=*)
            SCENE_ALIGNMENT_PORT="${1#*=}"
            shift
            ;;
        --scene_reference_opacity)
            SCENE_REFERENCE_OPACITY="$2"
            shift 2
            ;;
        --scene_reference_opacity=*)
            SCENE_REFERENCE_OPACITY="${1#*=}"
            shift
            ;;
        --save_pose_trace)
            SAVE_POSE_TRACE=true
            shift
            ;;
        --pose_trace_dir)
            SAVE_POSE_TRACE=true
            POSE_TRACE_DIR="$2"
            shift 2
            ;;
        --pose_trace_dir=*)
            SAVE_POSE_TRACE=true
            POSE_TRACE_DIR="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --dummy)
            DUMMY_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: --model_path is required." >&2
    usage >&2
    exit 1
fi
if [[ -z "$NORM_STATS_PATH" ]]; then
    echo "ERROR: --norm_stats_path (or UMI_LEFT_FRANKA_NORM_STATS) is required." >&2
    exit 2
fi
if [[ "$DRY_RUN" == false && ! -f "${MODEL_PATH}/model.safetensors" ]]; then
    echo "ERROR: model.safetensors not found under ${MODEL_PATH}." >&2
    exit 1
fi
if [[ "$DRY_RUN" == false && ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: config.json not found under ${MODEL_PATH}." >&2
    exit 1
fi
if [[ "$DRY_RUN" == false && ! -f "${NORM_STATS_PATH}/norm_stats.json" ]]; then
    echo "ERROR: norm_stats.json not found under ${NORM_STATS_PATH}." >&2
    exit 1
fi
if [[ "$DRY_RUN" == false && "$DUMMY_MODE" == false ]] && \
    { [[ -z "$ROBOT_IP" ]] || [[ -z "$WRIST_CAMERA" ]] || [[ -z "$FRONT_CAMERA" ]]; }; then
    echo "ERROR: --robot_ip, --wrist_camera and --front_camera are required." >&2
    exit 1
fi
if [ -n "${SCENE_REFERENCE}" ] && [ -n "${SCENE_REFERENCE_DIR}" ]; then
    echo "ERROR: use --scene_reference or --scene_reference_dir, not both." >&2
    exit 1
fi
if [ -n "${SCENE_REFERENCE}" ]; then
    if [[ ! "${SCENE_REFERENCE}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
        echo "ERROR: invalid --scene_reference name: ${SCENE_REFERENCE}" >&2
        exit 1
    fi
    SCENE_REFERENCE_SET_DIR="${SCENE_REFERENCE_ROOT}/${SCENE_REFERENCE}"
    if [ ! -f "${SCENE_REFERENCE_SET_DIR}/manifest.json" ]; then
        echo "ERROR: unknown scene reference preset: ${SCENE_REFERENCE}" >&2
        echo "Expected ${SCENE_REFERENCE_SET_DIR}/manifest.json" >&2
        exit 1
    fi
    EXTRA_ARGS+=(
        "env.eval.override_cfg.scene_alignment_enabled=true"
        "env.eval.override_cfg.scene_alignment_reference_set_dir=${SCENE_REFERENCE_SET_DIR}"
        "env.eval.override_cfg.scene_alignment_host=${SCENE_ALIGNMENT_HOST}"
        "env.eval.override_cfg.scene_alignment_port=${SCENE_ALIGNMENT_PORT}"
        "env.eval.override_cfg.scene_alignment_reference_opacity=${SCENE_REFERENCE_OPACITY}"
    )
fi
if [ -n "${SCENE_REFERENCE_DIR}" ]; then
    if [ ! -d "${SCENE_REFERENCE_DIR}" ]; then
        echo "ERROR: scene reference directory does not exist: ${SCENE_REFERENCE_DIR}" >&2
        exit 1
    fi
    if [ ! -f "${SCENE_REFERENCE_DIR}/0_wrist_2.mp4" ]; then
        echo "ERROR: 0_wrist_2.mp4 not found under ${SCENE_REFERENCE_DIR}." >&2
        exit 1
    fi
    EXTRA_ARGS+=(
        "env.eval.override_cfg.scene_alignment_enabled=true"
        "env.eval.override_cfg.scene_alignment_reference_dir=${SCENE_REFERENCE_DIR}"
        "env.eval.override_cfg.scene_alignment_epoch_offset=${SCENE_EPOCH_OFFSET}"
        "env.eval.override_cfg.scene_alignment_host=${SCENE_ALIGNMENT_HOST}"
        "env.eval.override_cfg.scene_alignment_port=${SCENE_ALIGNMENT_PORT}"
        "env.eval.override_cfg.scene_alignment_reference_opacity=${SCENE_REFERENCE_OPACITY}"
    )
    if [ -n "${SCENE_CACHE_DIR}" ]; then
        EXTRA_ARGS+=(
            "env.eval.override_cfg.scene_alignment_cache_dir=${SCENE_CACHE_DIR}"
        )
    fi
fi
if [ "${SAVE_POSE_TRACE}" = true ]; then
    EXTRA_ARGS+=("env.eval.override_cfg.save_pose_trace=true")
    if [ -n "${POSE_TRACE_DIR}" ]; then
        EXTRA_ARGS+=("env.eval.override_cfg.pose_trace_dir=${POSE_TRACE_DIR}")
    fi
fi
if [ "${DUMMY_MODE}" = true ]; then
    WRIST_CAMERA="${WRIST_CAMERA:-dummy_wrist}"
    FRONT_CAMERA="${FRONT_CAMERA:-dummy_front}"
    EXTRA_ARGS+=(
        "env.eval.override_cfg.is_dummy=true"
        "env.eval.override_cfg.enable_camera_player=false"
        "env.eval.keyboard_reward_wrapper=null"
        "env.eval.video_cfg.save_video=false"
        "env.eval.max_episode_steps=20"
        "env.eval.override_cfg.max_num_steps=20"
        "env.eval.rollout_epoch=1"
    )
fi

# Select the top-level evaluator directly, without changing PATH. ROS tools
# such as xacro must continue to resolve the system `python3` interpreter.
if [[ "$DRY_RUN" == false && ! -x "$POLICY_PYTHON" ]]; then
    echo "ERROR: --policy_python is not executable: ${POLICY_PYTHON}" >&2
    exit 1
fi
if [[ "$DRY_RUN" == false && ! -x "$FRANKA_PYTHON" ]]; then
    echo "ERROR: --franka_python is not executable: ${FRANKA_PYTHON}" >&2
    exit 1
fi
if [[ "$DRY_RUN" == false && -n "$SCENE_REFERENCE" ]]; then
    PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}" "${FRANKA_PYTHON}" -c \
        'import sys; from training.envs.realworld.common.scene_reference_set import SceneReferenceSet; preset = SceneReferenceSet.load(sys.argv[1]); preset.validate_saved_references()' \
        "${SCENE_REFERENCE_SET_DIR}"
fi
if [[ "$DRY_RUN" == false && "$DUMMY_MODE" == false && ! -e "$KEYBOARD_DEVICE" ]]; then
    echo "ERROR: keyboard device does not exist: ${KEYBOARD_DEVICE}" >&2
    exit 1
fi
export TRAINING_EVAL_PYTHON="${POLICY_PYTHON}"
export TRAINING_KEYBOARD_DEVICE="${KEYBOARD_DEVICE}"

export PI05_DEPLOY_CKPT="${MODEL_PATH}"
export UMI_LEFT_FRANKA_NORM_STATS="${NORM_STATS_PATH}"
export FRANKA_ROBOT_IP="${ROBOT_IP}"
export FRANKA_WRIST_CAMERA_SERIAL="${WRIST_CAMERA}"
export FRANKA_FRONT_CAMERA_SERIAL="${FRONT_CAMERA}"
EXTRA_ARGS+=(
    "rollout.model.model_path=${MODEL_PATH}"
    "rollout.model.openpi_data.norm_stats_path=${NORM_STATS_PATH}"
)

# Start ROS before Ray and the rollout model compete for CPU. The helper also
# gives every local ROS process one consistent address and verifies that the
# master responds before hardware workers are launched.
if [[ "$DRY_RUN" == false && "$DUMMY_MODE" != true ]]; then
    [[ -n "$ROBOT_IP" && -n "$WRIST_CAMERA" && -n "$FRONT_CAMERA" ]] || { echo "Set the robot IP and both camera serials explicitly." >&2; exit 2; }
# Load ROS and the Franka catkin overlay before Ray starts.  Ray workers inherit
# these variables and need the generated franka_msgs Python package from this
# workspace.  Temporarily disable nounset because ROS setup scripts are not
# guaranteed to be compatible with `set -u`.
ROS_NOETIC_SETUP="${ROS_NOETIC_SETUP:-/opt/ros/noetic/setup.bash}"
FRANKA_CATKIN_SETUP="${FRANKA_CATKIN_SETUP:-${HOME}/catkin_ws_franka/devel/setup.bash}"
for setup_file in "${ROS_NOETIC_SETUP}" "${FRANKA_CATKIN_SETUP}"; do
    if [ ! -f "${setup_file}" ]; then
        echo "ERROR: required ROS setup file does not exist: ${setup_file}" >&2
        exit 1
    fi
done
SCRIPT_ARGS=("$@")
set --
set +u
# shellcheck disable=SC1090
source "${ROS_NOETIC_SETUP}"
# shellcheck disable=SC1090
source "${FRANKA_CATKIN_SETUP}"
set -u
set -- "${SCRIPT_ARGS[@]}"
unset SCRIPT_ARGS
fi

export ROS_MASTER_URI="${TRAINING_ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_HOSTNAME="${TRAINING_ROS_HOSTNAME:-127.0.0.1}"
unset ROS_IP
if [[ "$DRY_RUN" == false && "$DUMMY_MODE" == false ]]; then
    bash "${REPO_PATH}/ray_utils/realworld/start_roscore.sh"
fi

if "$DRY_RUN"; then EXTRA_ARGS+=(--dry-run); fi
exec bash "${SCRIPT_DIR}/run_eval.sh" \
    realworld \
    umi_left_franka_tcp_rot6d_eval \
    "${EXTRA_ARGS[@]}"
