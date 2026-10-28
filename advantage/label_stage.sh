#!/usr/bin/env bash
# Modified for HIL-UMI.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
    echo "Usage: bash label_stage.sh TASK_NAME STAGE_X [extra_args...]"
    echo "Example: bash label_stage.sh fold_towel stage1"
    echo "Example: bash label_stage.sh fold_towel stage1 --dry-run"
    exit 1
fi

readonly TASK_NAME="$1"
readonly STAGE_NAME="$2"
shift 2

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/source_runtime.sh"
advantage_init_runtime "${SCRIPT_DIR}"
readonly PYTHON="${ADVANTAGE_PYTHON}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Error: Python interpreter not found at ${PYTHON}" >&2
    exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/scripts/label_stage.py" \
    --task-name "${TASK_NAME}" \
    --stage-name "${STAGE_NAME}" \
    "$@"
