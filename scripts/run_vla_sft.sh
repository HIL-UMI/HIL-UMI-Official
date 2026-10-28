#! /bin/bash
# Modified for HIL-UMI.

export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_PATH}"
export EMBODIED_PATH="${REPO_PATH}/examples/sft"
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "$1" ]; then
    CONFIG_NAME="umi_left_franka_sft_openpi_pytorch_tcp_rot6d"
else
    CONFIG_NAME=$1
    shift
fi

# Parse optional flags:
#   --task_name <name>  names the run directory (logs/<task_name>-<n>)
#                       so checkpoints of different tasks stay separated
#                       without needing a dedicated yaml per task.
#   --data_path <path>  overrides data.train_data_paths (merged LeRobot
#                       dataset directory or repo id) for this run.
#   --post-train <true|false>
#                       when true, train batches from data.train_data_paths
#                       and data.post_train_data_path using the configured
#                       post-train ratios.
#                       runner.resume_dir is only for recovering an interrupted
#                       post-train run, not for switching from stage-1 SFT.
#   --post-train-data-path <path>
#                       LeRobot-format online correction dataset used when
#                       --post-train true.
#   --post-train-model-path <path>
#                       new-format stage-1 checkpoint containing
#                       model.safetensors. This starts a weights-only post-train
#                       run: Adam and the LR scheduler are initialized fresh,
#                       and runner.resume_dir must not be set. Unless overridden,
#                       the safe post-train schedule is 20k steps, peak LR
#                       2.5e-6, 500 warmup steps, and min LR 2.5e-7.
#   --post-train-demo-ratio <float>
#                       relative weight for original demonstration data during
#                       post-train. Default: 0.5.
#   --post-train-correction-ratio <float>
#                       relative weight for collection/correction data during
#                       post-train. Default: 0.5.
#                       Aliases:
#                       --post-train-original-ratio maps to demo ratio;
#                       --post-train-collection-ratio maps to correction ratio.
#   --norm_stats_path <path>  selects the per-task norm stats directory produced
#                       by calculate_norm_stats.py --task-name. The override
#                       keys are selected for legacy OpenPI or openpi_pytorch.
#   --fast-norm-stats   selects the no-padding GPU stats directory named
#                       <task_name>_no_padding_gpu. It cannot be combined with
#                       --norm_stats_path.
#   --max_steps <N>     overrides runner.max_steps and
#                       actor.optim.total_training_steps (kept in sync so the
#                       lr schedule matches the actual run length).
#   --lr <value>        overrides actor.optim.lr.
#   --min_lr <value>    overrides actor.optim.min_lr.
#   --lr_warmup_steps <N>
#                       overrides actor.optim.lr_warmup_steps.
#   --save_interval <N> overrides runner.save_interval (checkpoint frequency).
#   --gpus <range>      overrides cluster.component_placement.actor, e.g.
#                       "0-7" (all 8 GPUs) or "0-3" (4 GPUs). Note
#                       global_batch_size must be divisible by
#                       micro_batch_size * <number of GPUs>.
#   --dry-run           print the resolved command without creating logs or
#                       starting training.
# Any remaining arguments are forwarded to the python entry as Hydra
# overrides, e.g. actor.global_batch_size=64.
TASK_NAME=""
DATA_PATH=""
POST_TRAIN=""
POST_TRAIN_DATA_PATH=""
POST_TRAIN_MODEL_PATH=""
POST_TRAIN_DEMO_RATIO=""
POST_TRAIN_CORRECTION_RATIO=""
NORM_STATS_PATH=""
FAST_NORM_STATS="false"
MAX_STEPS=""
LEARNING_RATE=""
MIN_LEARNING_RATE=""
LR_WARMUP_STEPS=""
SAVE_INTERVAL=""
GPUS=""
DRY_RUN="false"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --task_name)
            TASK_NAME="$2"
            shift 2
            ;;
        --task_name=*)
            TASK_NAME="${1#*=}"
            shift
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --data_path=*)
            DATA_PATH="${1#*=}"
            shift
            ;;
        --post-train)
            POST_TRAIN="$2"
            shift 2
            ;;
        --post-train=*)
            POST_TRAIN="${1#*=}"
            shift
            ;;
        --post_train)
            POST_TRAIN="$2"
            shift 2
            ;;
        --post_train=*)
            POST_TRAIN="${1#*=}"
            shift
            ;;
        --post-train-data-path)
            POST_TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --post-train-data-path=*)
            POST_TRAIN_DATA_PATH="${1#*=}"
            shift
            ;;
        --post_train_data_path)
            POST_TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --post_train_data_path=*)
            POST_TRAIN_DATA_PATH="${1#*=}"
            shift
            ;;
        --post-train-model-path|--post_train_model_path)
            POST_TRAIN_MODEL_PATH="$2"
            shift 2
            ;;
        --post-train-model-path=*|--post_train_model_path=*)
            POST_TRAIN_MODEL_PATH="${1#*=}"
            shift
            ;;
        --post-train-demo-ratio|--post-train-original-ratio)
            POST_TRAIN_DEMO_RATIO="$2"
            shift 2
            ;;
        --post-train-demo-ratio=*|--post-train-original-ratio=*)
            POST_TRAIN_DEMO_RATIO="${1#*=}"
            shift
            ;;
        --post_train_demo_ratio|--post_train_original_ratio)
            POST_TRAIN_DEMO_RATIO="$2"
            shift 2
            ;;
        --post_train_demo_ratio=*|--post_train_original_ratio=*)
            POST_TRAIN_DEMO_RATIO="${1#*=}"
            shift
            ;;
        --post-train-correction-ratio|--post-train-collection-ratio)
            POST_TRAIN_CORRECTION_RATIO="$2"
            shift 2
            ;;
        --post-train-correction-ratio=*|--post-train-collection-ratio=*)
            POST_TRAIN_CORRECTION_RATIO="${1#*=}"
            shift
            ;;
        --post_train_correction_ratio|--post_train_collection_ratio)
            POST_TRAIN_CORRECTION_RATIO="$2"
            shift 2
            ;;
        --post_train_correction_ratio=*|--post_train_collection_ratio=*)
            POST_TRAIN_CORRECTION_RATIO="${1#*=}"
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
        --fast-norm-stats|--fast_norm_stats)
            FAST_NORM_STATS="true"
            shift
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max_steps=*)
            MAX_STEPS="${1#*=}"
            shift
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lr=*)
            LEARNING_RATE="${1#*=}"
            shift
            ;;
        --min_lr|--min-lr)
            MIN_LEARNING_RATE="$2"
            shift 2
            ;;
        --min_lr=*|--min-lr=*)
            MIN_LEARNING_RATE="${1#*=}"
            shift
            ;;
        --lr_warmup_steps|--lr-warmup-steps)
            LR_WARMUP_STEPS="$2"
            shift 2
            ;;
        --lr_warmup_steps=*|--lr-warmup-steps=*)
            LR_WARMUP_STEPS="${1#*=}"
            shift
            ;;
        --save_interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --save_interval=*)
            SAVE_INTERVAL="${1#*=}"
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"

get_hydra_override() {
    local key="$1"
    local value=""
    for ARG in "${EXTRA_ARGS[@]}"; do
        if [[ "${ARG}" == "${key}="* || "${ARG}" == "+${key}="* ]]; then
            value="${ARG#*=}"
        fi
    done
    printf "%s" "${value}"
}

strip_yaml_scalar_quotes() {
    local value="$1"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    printf "%s" "${value}"
}

read_config_scalar() {
    local key="$1"
    local value=""
    if [ -f "${CONFIG_FILE}" ]; then
        value=$(sed -n "s/^[[:space:]]*${key}:[[:space:]]*//p" "${CONFIG_FILE}" | tail -n 1)
        strip_yaml_scalar_quotes "${value}"
    fi
}

is_truthy() {
    case "$(printf "%s" "$1" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|y|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if is_truthy "${FAST_NORM_STATS}" && [ -n "${NORM_STATS_PATH}" ]; then
    echo "ERROR: --fast-norm-stats cannot be combined with --norm_stats_path."
    exit 1
fi
if is_truthy "${FAST_NORM_STATS}" && [ -z "${TASK_NAME}" ]; then
    echo "ERROR: --fast-norm-stats requires --task_name."
    exit 1
fi

format_post_train_ratio_suffix() {
    awk -v demo="$1" -v correction="$2" '
        function fmt(value, text) {
            text = sprintf("%.6f", value)
            sub(/0+$/, "", text)
            sub(/\.$/, "", text)
            gsub(/\./, "p", text)
            return text
        }
        BEGIN {
            if (demo <= 0 || correction <= 0) {
                exit 1
            }
            total = demo + correction
            print fmt(demo / total * 100) "_" fmt(correction / total * 100)
        }
    '
}

format_post_train_stats_ratio_suffix() {
    awk -v demo="$1" -v correction="$2" '
        function fmt(value, text) {
            text = sprintf("%.6f", value)
            sub(/0+$/, "", text)
            sub(/\.$/, "", text)
            return text
        }
        BEGIN {
            if (demo <= 0 || correction <= 0) {
                exit 1
            }
            print fmt(demo) "_" fmt(correction)
        }
    '
}

format_post_train_experiment_ratio_suffix() {
    awk -v demo="$1" -v correction="$2" '
        function fmt(value, text, numeric) {
            text = sprintf("%.6f", value)
            sub(/0+$/, "", text)
            sub(/\.$/, "", text)
            if (text ~ /^[0-9]+$/) {
                numeric = text + 0
                if (numeric >= 10 && numeric % 10 == 0) {
                    text = sprintf("%d", numeric / 10)
                }
            }
            gsub(/\./, "p", text)
            return text
        }
        BEGIN {
            if (demo <= 0 || correction <= 0) {
                exit 1
            }
            total = demo + correction
            print fmt(demo / total * 100) "-" fmt(correction / total * 100)
        }
    '
}

# A stage-1 model path denotes a new, weights-only post-train run. Keep these
# defaults local to this explicit mode so ordinary SFT and full-state resume
# retain their existing behavior.
if [ -n "${POST_TRAIN_MODEL_PATH}" ]; then
    case "$(printf "%s" "${POST_TRAIN}" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|n|off)
            echo "ERROR: --post-train-model-path conflicts with --post-train false."
            exit 1
            ;;
    esac
    POST_TRAIN="true"

    if [ ! -f "${POST_TRAIN_MODEL_PATH}/model.safetensors" ]; then
        echo "ERROR: weights-only post-train model not found at ${POST_TRAIN_MODEL_PATH}/model.safetensors"
        echo "Convert the stage-1 checkpoint first with the OpenPI sft2new converter."
        exit 1
    fi

    RESUME_OVERRIDE=$(get_hydra_override "runner.resume_dir")
    if [ -n "${RESUME_OVERRIDE}" ] && [ "${RESUME_OVERRIDE}" != "null" ]; then
        echo "ERROR: --post-train-model-path cannot be combined with runner.resume_dir."
        echo "Weights-only post-train must initialize a fresh optimizer and LR scheduler."
        exit 1
    fi

    EXTRA_ARGS+=(
        "actor.model.model_path=${POST_TRAIN_MODEL_PATH}"
        "runner.resume_dir=null"
    )
    if [ -z "${MAX_STEPS}" ] && [ -z "$(get_hydra_override "runner.max_steps")" ]; then
        MAX_STEPS="20000"
    fi
    if [ -z "${LEARNING_RATE}" ] && [ -z "$(get_hydra_override "actor.optim.lr")" ]; then
        LEARNING_RATE="2.5e-6"
    fi
    if [ -z "${MIN_LEARNING_RATE}" ] && [ -z "$(get_hydra_override "actor.optim.min_lr")" ]; then
        MIN_LEARNING_RATE="2.5e-7"
    fi
    if [ -z "${LR_WARMUP_STEPS}" ] && [ -z "$(get_hydra_override "actor.optim.lr_warmup_steps")" ]; then
        LR_WARMUP_STEPS="500"
    fi
fi

RUN_NAME="${TASK_NAME:-${CONFIG_NAME}}"
if [ -n "${DATA_PATH}" ]; then
    EXTRA_ARGS+=("data.train_data_paths=${DATA_PATH}")
fi
if [ -n "${POST_TRAIN}" ]; then
    EXTRA_ARGS+=("data.post_train=${POST_TRAIN}")
fi
if [ -n "${POST_TRAIN_DATA_PATH}" ]; then
    EXTRA_ARGS+=("data.post_train_data_path=${POST_TRAIN_DATA_PATH}")
fi
if [ -n "${POST_TRAIN_DEMO_RATIO}" ]; then
    EXTRA_ARGS+=("data.post_train_demo_ratio=${POST_TRAIN_DEMO_RATIO}")
fi
if [ -n "${POST_TRAIN_CORRECTION_RATIO}" ]; then
    EXTRA_ARGS+=("data.post_train_correction_ratio=${POST_TRAIN_CORRECTION_RATIO}")
fi
if [ -n "${MAX_STEPS}" ]; then
    EXTRA_ARGS+=("runner.max_steps=${MAX_STEPS}" "actor.optim.total_training_steps=${MAX_STEPS}")
fi
if [ -n "${LEARNING_RATE}" ]; then
    EXTRA_ARGS+=("actor.optim.lr=${LEARNING_RATE}")
fi
if [ -n "${MIN_LEARNING_RATE}" ]; then
    EXTRA_ARGS+=("actor.optim.min_lr=${MIN_LEARNING_RATE}")
fi
if [ -n "${LR_WARMUP_STEPS}" ]; then
    EXTRA_ARGS+=("actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}")
fi
if [ -n "${SAVE_INTERVAL}" ]; then
    EXTRA_ARGS+=("runner.save_interval=${SAVE_INTERVAL}")
fi
if [ -n "${GPUS}" ]; then
    EXTRA_ARGS+=("cluster.component_placement.actor=${GPUS}")
fi

POST_TRAIN_DEMO_RATIO_EFFECTIVE=$(get_hydra_override "data.post_train_demo_ratio")
if [ -z "${POST_TRAIN_DEMO_RATIO_EFFECTIVE}" ]; then
    POST_TRAIN_DEMO_RATIO_EFFECTIVE=$(read_config_scalar "post_train_demo_ratio")
fi
if [ -z "${POST_TRAIN_DEMO_RATIO_EFFECTIVE}" ] || [ "${POST_TRAIN_DEMO_RATIO_EFFECTIVE}" = "null" ]; then
    POST_TRAIN_DEMO_RATIO_EFFECTIVE="0.5"
fi
POST_TRAIN_CORRECTION_RATIO_EFFECTIVE=$(get_hydra_override "data.post_train_correction_ratio")
if [ -z "${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}" ]; then
    POST_TRAIN_CORRECTION_RATIO_EFFECTIVE=$(read_config_scalar "post_train_correction_ratio")
fi
if [ -z "${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}" ] || [ "${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}" = "null" ]; then
    POST_TRAIN_CORRECTION_RATIO_EFFECTIVE="0.5"
fi
POST_TRAIN_RATIO_SUFFIX=$(format_post_train_ratio_suffix "${POST_TRAIN_DEMO_RATIO_EFFECTIVE}" "${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}")
POST_TRAIN_STATS_RATIO_SUFFIX=$(format_post_train_stats_ratio_suffix "${POST_TRAIN_DEMO_RATIO_EFFECTIVE}" "${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}")
POST_TRAIN_EXPERIMENT_RATIO_SUFFIX=$(format_post_train_experiment_ratio_suffix "${POST_TRAIN_DEMO_RATIO_EFFECTIVE}" "${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}")
if [ -z "${POST_TRAIN_RATIO_SUFFIX}" ] || [ -z "${POST_TRAIN_STATS_RATIO_SUFFIX}" ] || [ -z "${POST_TRAIN_EXPERIMENT_RATIO_SUFFIX}" ]; then
    echo "ERROR: invalid post-train ratios: demo=${POST_TRAIN_DEMO_RATIO_EFFECTIVE}, correction=${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}"
    exit 1
fi
POST_TRAIN_DATA_PATH_EFFECTIVE="${POST_TRAIN_DATA_PATH}"
if [ -z "${POST_TRAIN_DATA_PATH_EFFECTIVE}" ]; then
    POST_TRAIN_DATA_PATH_EFFECTIVE=$(get_hydra_override "data.post_train_data_path")
fi
if [ -z "${POST_TRAIN_DATA_PATH_EFFECTIVE}" ]; then
    POST_TRAIN_DATA_PATH_EFFECTIVE=$(read_config_scalar "post_train_data_path")
fi
if [ "${POST_TRAIN_DATA_PATH_EFFECTIVE}" = "null" ]; then
    POST_TRAIN_DATA_PATH_EFFECTIVE=""
fi
POST_TRAIN_EXPERIMENT_KIND_SUFFIX=""
if [ -n "${POST_TRAIN_MODEL_PATH}" ]; then
    POST_TRAIN_EXPERIMENT_KIND_SUFFIX="-weights-only"
fi

# Per-task norm stats conventions used by calculate_norm_stats.py. The
# openpi_pytorch single-Franka path has its own assets tree; all legacy configs
# keep the existing pi05_realworld convention.
if [ -z "${NORM_STATS_PATH}" ] && [ -n "${TASK_NAME}" ]; then
    case "${CONFIG_NAME}" in
        umi_left_franka_sft_openpi_pytorch_tcp_rot6d)
            SINGLE_FRANKA_ASSETS_DIR="${UMI_LEFT_FRANKA_ASSETS_DIR:-${REPO_PATH}/assets/pi05_singlefranka_tcp_rot6d}"
            DEFAULT_STATS_DIR="${SINGLE_FRANKA_ASSETS_DIR}/umi_left_franka_tcp_rot6d/${TASK_NAME}"
            ;;
        *)
            DEFAULT_STATS_DIR="${REPO_PATH}/assets/pi05_realworld/realworld_franka_bin_relocation/${TASK_NAME}"
            ;;
    esac
    if is_truthy "${FAST_NORM_STATS}"; then
        if [ "${CONFIG_NAME}" != \
            "umi_left_franka_sft_openpi_pytorch_tcp_rot6d" ]; then
            echo "ERROR: --fast-norm-stats currently supports" \
                "umi_left_franka_sft_openpi_pytorch_tcp_rot6d only."
            exit 1
        fi
        FAST_STATS_DIR="${DEFAULT_STATS_DIR}_no_padding_gpu"
        if [ ! -f "${FAST_STATS_DIR}/norm_stats.json" ]; then
            echo "ERROR: fast no-padding norm stats not found at" \
                "${FAST_STATS_DIR}/norm_stats.json"
            echo "Compute them with calculate_norm_stats_parquet_fast.py" \
                "--output-task-name ${TASK_NAME}_no_padding_gpu."
            exit 1
        fi
        NORM_STATS_PATH="${FAST_STATS_DIR}"
        echo "Using fast no-padding norm stats: ${NORM_STATS_PATH}"
    else
        POST_TRAIN_FOR_STATS="${POST_TRAIN}"
        if [ -z "${POST_TRAIN_FOR_STATS}" ]; then
            POST_TRAIN_FOR_STATS=$(get_hydra_override "data.post_train")
        fi
        if [ -z "${POST_TRAIN_FOR_STATS}" ]; then
            POST_TRAIN_FOR_STATS=$(read_config_scalar "post_train")
        fi
        POST_TRAIN_LOWER=$(printf "%s" "${POST_TRAIN_FOR_STATS}" | tr '[:upper:]' '[:lower:]')
        if [[ "${POST_TRAIN_LOWER}" =~ ^(1|true|yes|y|on)$ ]]; then
            if [ -z "${POST_TRAIN_DATA_PATH_EFFECTIVE}" ]; then
                echo "ERROR: post-train norm stats lookup requires --post-train-data-path or data.post_train_data_path."
                exit 1
            fi
            POST_TRAIN_DATA_NAME=$(basename "${POST_TRAIN_DATA_PATH_EFFECTIVE%/}")
            if [ -z "${POST_TRAIN_DATA_NAME}" ]; then
                echo "ERROR: --post-train-data-path must have a non-empty final path component."
                exit 1
            fi
            POST_TRAIN_STATS_DIR="${DEFAULT_STATS_DIR}_${POST_TRAIN_STATS_RATIO_SUFFIX}_${POST_TRAIN_DATA_NAME}"
        fi
        if [[ "${POST_TRAIN_LOWER}" =~ ^(1|true|yes|y|on)$ ]] && [ -f "${POST_TRAIN_STATS_DIR}/norm_stats.json" ]; then
            NORM_STATS_PATH="${POST_TRAIN_STATS_DIR}"
            echo "Using post-train norm stats (${POST_TRAIN_DEMO_RATIO_EFFECTIVE}:${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}): ${NORM_STATS_PATH}"
        elif [[ "${POST_TRAIN_LOWER}" =~ ^(1|true|yes|y|on)$ ]]; then
            echo "ERROR: post-train norm stats not found at ${POST_TRAIN_STATS_DIR}/norm_stats.json"
            echo "Compute them first with calculate_norm_stats.py --post-train true --post-train-demo-ratio ${POST_TRAIN_DEMO_RATIO_EFFECTIVE} --post-train-correction-ratio ${POST_TRAIN_CORRECTION_RATIO_EFFECTIVE}, or pass --norm_stats_path explicitly."
            exit 1
        fi
        if [ -z "${NORM_STATS_PATH}" ] && [ -f "${DEFAULT_STATS_DIR}/norm_stats.json" ]; then
            NORM_STATS_PATH="${DEFAULT_STATS_DIR}"
            echo "Using per-task norm stats: ${NORM_STATS_PATH}"
        elif [ -z "${NORM_STATS_PATH}" ]; then
            echo "WARNING: no norm stats found at ${DEFAULT_STATS_DIR}; falling back to the default assets logic."
        fi
    fi
fi
if [ -n "${NORM_STATS_PATH}" ]; then
    if [ "$(basename "${NORM_STATS_PATH}")" = "norm_stats.json" ]; then
        NORM_STATS_PATH=$(dirname "${NORM_STATS_PATH}")
    fi
    if [ "${CONFIG_NAME}" = "umi_left_franka_sft_openpi_pytorch_tcp_rot6d" ]; then
        NORM_STATS_ASSETS_DIR=$(dirname "${NORM_STATS_PATH}")
        NORM_STATS_ASSET_ID=$(basename "${NORM_STATS_PATH}")
        EXTRA_ARGS+=(
            "actor.model.openpi.assets_dir=${NORM_STATS_ASSETS_DIR}"
            "actor.model.openpi.asset_id=${NORM_STATS_ASSET_ID}"
        )
    else
        EXTRA_ARGS+=("actor.model.openpi_data.norm_stats_path=${NORM_STATS_PATH}")
    fi
fi

echo "Using Python at $(which python)"
LOG_OFFLOAD_ROOT="${SFT_LOG_OFFLOAD_DIR:-${REPO_PATH}/outputs/checkpoint_offload}"
CHECKPOINT_OFFLOAD_ROOT=$(get_hydra_override "runner.checkpoint_offload_dir")
if [ -z "${CHECKPOINT_OFFLOAD_ROOT}" ]; then
    CHECKPOINT_OFFLOAD_ROOT=$(read_config_scalar "checkpoint_offload_dir")
fi
if [ "${CHECKPOINT_OFFLOAD_ROOT}" = "null" ]; then
    CHECKPOINT_OFFLOAD_ROOT=""
fi

POST_TRAIN_EFFECTIVE=$(get_hydra_override "data.post_train")
if [ -z "${POST_TRAIN_EFFECTIVE}" ]; then
    POST_TRAIN_EFFECTIVE=$(read_config_scalar "post_train")
fi
RESUME_DIR=$(get_hydra_override "runner.resume_dir")
if [ -z "${RESUME_DIR}" ]; then
    RESUME_DIR=$(read_config_scalar "resume_dir")
fi
if is_truthy "${POST_TRAIN_EFFECTIVE}" && [ -n "${RESUME_DIR}" ] && [ "${RESUME_DIR}" != "null" ]; then
    echo "WARNING: post-train runner.resume_dir restores model, Adam, and LR scheduler state."
    echo "Use it only to recover the same post-train schedule; keep max_steps and total_training_steps unchanged."
    echo "For the stage-1 SFT -> collection transition, use --post-train-model-path instead."
fi

POST_TRAIN_RUN_DIR_NAME=""
if is_truthy "${POST_TRAIN_EFFECTIVE}"; then
    if [ -n "${RESUME_DIR}" ] && [ "${RESUME_DIR}" != "null" ]; then
        RESUME_DIR="${RESUME_DIR%/}"
        RESUME_CHECKPOINTS_DIR=$(dirname "${RESUME_DIR}")
        RESUME_EXPERIMENT_DIR=$(dirname "${RESUME_CHECKPOINTS_DIR}")
        POST_TRAIN_RUN_DIR_NAME=$(basename "$(dirname "${RESUME_EXPERIMENT_DIR}")")
    fi
    if [ -z "$(get_hydra_override "runner.logger.experiment_name")" ]; then
        BASE_EXPERIMENT_NAME=$(read_config_scalar "experiment_name")
        if [ -z "${BASE_EXPERIMENT_NAME}" ]; then
            BASE_EXPERIMENT_NAME="${CONFIG_NAME}"
        fi
        case "${BASE_EXPERIMENT_NAME}" in
            *_sft)
                POST_TRAIN_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME%_sft}_post-train${POST_TRAIN_EXPERIMENT_KIND_SUFFIX}${POST_TRAIN_EXPERIMENT_RATIO_SUFFIX}"
                ;;
            *)
                POST_TRAIN_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_post-train${POST_TRAIN_EXPERIMENT_KIND_SUFFIX}${POST_TRAIN_EXPERIMENT_RATIO_SUFFIX}"
                ;;
        esac
        EXTRA_ARGS+=("runner.logger.experiment_name=${POST_TRAIN_EXPERIMENT_NAME}")
        echo "Using post-train experiment name: ${POST_TRAIN_EXPERIMENT_NAME}"
    fi
fi

run_dir_exists() {
    local dir_name="$1"
    if [ -e "${REPO_PATH}/logs/${dir_name}" ]; then
        return 0
    fi
    if [ -n "${CHECKPOINT_OFFLOAD_ROOT}" ] && [ -e "${CHECKPOINT_OFFLOAD_ROOT}/${dir_name}" ]; then
        return 0
    fi
    if [ -d "${LOG_OFFLOAD_ROOT}" ] && [ -e "${LOG_OFFLOAD_ROOT}/${dir_name}" ]; then
        return 0
    fi
    return 1
}

if is_truthy "${DRY_RUN}"; then
    RUN_DIR_NAME="${POST_TRAIN_RUN_DIR_NAME:-${RUN_NAME}-dry-run}"
    LOG_DIR="${REPO_PATH}/logs/${RUN_DIR_NAME}"
else
    mkdir -p "${REPO_PATH}/logs"
    if [ -n "${POST_TRAIN_RUN_DIR_NAME}" ]; then
        RUN_DIR_NAME="${POST_TRAIN_RUN_DIR_NAME}"
        LOG_DIR="${REPO_PATH}/logs/${RUN_DIR_NAME}"
        mkdir -p "${LOG_DIR}"
    else
        RUN_INDEX=1
        while true; do
            RUN_DIR_NAME="${RUN_NAME}-${RUN_INDEX}"
            LOG_DIR="${REPO_PATH}/logs/${RUN_DIR_NAME}"
            if run_dir_exists "${RUN_DIR_NAME}"; then
                RUN_INDEX=$((RUN_INDEX + 1))
                continue
            fi
            if mkdir "${LOG_DIR}" 2>/dev/null; then
                break
            fi
            RUN_INDEX=$((RUN_INDEX + 1))
        done
    fi
fi
echo "Using run directory: ${RUN_DIR_NAME}"
if is_truthy "${POST_TRAIN_EFFECTIVE}"; then
    MEGA_LOG_FILE="${LOG_DIR}/run_embodiment_post-train.log"
else
    MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
fi
CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "${EXTRA_ARGS[@]}"
)
if is_truthy "${DRY_RUN}"; then
    printf "Resolved command:"
    printf " %q" "${CMD[@]}"
    printf "\n"
    exit 0
fi
printf "%q " "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf "\n" >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
TRAIN_STATUS=${PIPESTATUS[0]}

# 训练结束后：把日志目录转存到配置的目标目录，再删除本地副本。
# 目标目录不存在或转存失败时保留本地副本。
# （checkpoint 在训练过程中已由 runner 的 checkpoint_offload_dir 边存边转，
#  这里转的是 tensorboard、run_embodiment.log 等剩余文件，目标结构与之一致。）
DEST_ROOT="${LOG_OFFLOAD_ROOT}"
if [ -d "${DEST_ROOT}" ]; then
    DEST_DIR="${DEST_ROOT}/$(basename "${LOG_DIR}")"
    echo "Transferring logs: ${LOG_DIR} -> ${DEST_DIR}"
    mkdir -p "${DEST_DIR}"
    if rsync -a --remove-source-files "${LOG_DIR}/" "${DEST_DIR}/"; then
        find "${LOG_DIR}" -depth -type d -empty -delete
        echo "Transfer complete. Local copy removed: ${LOG_DIR}"
    else
        echo "WARNING: rsync transfer failed; local copy kept at ${LOG_DIR}"
    fi
else
    echo "WARNING: ${DEST_ROOT} does not exist; logs kept at ${LOG_DIR}"
fi

exit ${TRAIN_STATUS}
