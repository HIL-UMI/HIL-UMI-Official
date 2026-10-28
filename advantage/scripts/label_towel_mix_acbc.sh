#!/usr/bin/env bash
# Modified for HIL-UMI.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/source_runtime.sh"
advantage_init_runtime "${REPO_ROOT}"

exec bash "${SCRIPT_DIR}/label_advantage_datasets.sh" \
    --checkpoint \
    "${ADVANTAGE_MODEL_CHECKPOINT:-${ADVANTAGE_CHECKPOINT_ROOT}/advantage/ADVANTAGE_TORCH_TOWEL_FOLD/towel_base_stage1_mix}" \
    --output-root "${ADVANTAGE_DATA_ROOT}/fold_towel/stage1/labeled" \
    --dataset "base=${ADVANTAGE_DATA_ROOT}/fold_towel/base/origin" \
    --dataset "stage1=${ADVANTAGE_DATA_ROOT}/fold_towel/stage1/origin" \
    "$@"
