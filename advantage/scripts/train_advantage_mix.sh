#!/usr/bin/env bash
# Modified for HIL-UMI.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train_advantage_mix.sh EXP_NAME GROUP_A_CSV GROUP_B_CSV [train_pytorch options]

Examples:
  bash scripts/train_advantage_mix.sh towel_base_stage1_mix \
      "fold_towel/base" \
      "fold_towel/stage1"

  bash scripts/train_advantage_mix.sh towel_multi_mix \
      "fold_towel/base,fold_towel/stage2" \
      "fold_towel/stage1,fold_towel/stage3" \
      --num-train-steps=20000

Relative entries are resolved below data/. If an entry is a stage directory,
its origin/ child is selected automatically. Absolute dataset paths are also accepted.
EOF
}

if (( $# < 3 )); then
    usage >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/source_runtime.sh"
advantage_init_runtime "${REPO_ROOT}"
readonly EXP_NAME="$1"
readonly GROUP_A_CSV="$2"
readonly GROUP_B_CSV="$3"
shift 3

resolve_group() {
    local csv="$1"
    local output_name="$2"
    local -n output_ref="${output_name}"
    local -a entries=()
    local raw spec candidate

    IFS=',' read -r -a entries <<< "${csv}"
    for raw in "${entries[@]}"; do
        spec="${raw#"${raw%%[![:space:]]*}"}"
        spec="${spec%"${spec##*[![:space:]]}"}"
        if [[ -z "${spec}" ]]; then
            echo "Error: dataset groups cannot contain empty entries: ${csv}" >&2
            exit 2
        fi

        if [[ "${spec}" = /* ]]; then
            candidate="${spec}"
        else
            candidate="${ADVANTAGE_DATA_ROOT}/${spec}"
        fi

        if [[ -f "${candidate}/meta/info.json" ]]; then
            :
        elif [[ -f "${candidate}/origin/meta/info.json" ]]; then
            candidate="${candidate}/origin"
        else
            echo "Error: not a LeRobot dataset (meta/info.json missing): ${spec}" >&2
            exit 2
        fi
        output_ref+=("$(realpath "${candidate}")")
    done

    if (( ${#output_ref[@]} == 0 )); then
        echo "Error: each advantage mix group must contain at least one dataset" >&2
        exit 2
    fi
}

group_a=()
group_b=()
resolve_group "${GROUP_A_CSV}" group_a
resolve_group "${GROUP_B_CSV}" group_b

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export JAX_PLATFORMS="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"

readonly NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
readonly RESUME="${RESUME:-0}"
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NPROC_PER_NODE must be a positive integer, got ${NPROC_PER_NODE}" >&2
    exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
    echo "Error: RESUME must be 0 or 1, got ${RESUME}" >&2
    exit 2
fi

cd "${REPO_ROOT}"

resume_args=()
if [[ "${RESUME}" == "1" ]]; then
    resume_args+=(--resume)
fi

echo "Advantage mix training"
echo "  experiment: ${EXP_NAME}"
echo "  group A (${#group_a[@]} datasets, probability 0.5): ${group_a[*]}"
echo "  group B (${#group_b[@]} datasets, probability 0.5): ${group_b[*]}"

exec "${ADVANTAGE_PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/train_pytorch.py \
    ADVANTAGE_TORCH_TOWEL_FOLD \
    --exp-name="${EXP_NAME}" \
    --advantage-dataset-group-a "${group_a[@]}" \
    --advantage-dataset-group-b "${group_b[@]}" \
    "${resume_args[@]}" \
    "$@"
