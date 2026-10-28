#!/usr/bin/env bash
# Modified for HIL-UMI.
set -euo pipefail
export EVALUATIONS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$EVALUATIONS_PATH")"
export HIL_UMI_ROOT="${HIL_UMI_ROOT:-$REPO_PATH}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
EVAL_PYTHON="${EVAL_PYTHON:-${TRAINING_EVAL_PYTHON:-${POLICY_PYTHON:-${REPO_PATH}/.venv-policy/bin/python}}}"
DRY_RUN=false
ARGS=()
usage() {
    echo "Usage: $0 [--dry-run] realworld umi_left_franka_tcp_rot6d_eval [Hydra overrides]"
    echo "POLICY_PYTHON defaults to .venv-policy; EVAL_PYTHON or TRAINING_EVAL_PYTHON overrides it."
    echo "EVAL_LOG_DIR overrides the per-run log directory. --dry-run only prints the command."
}
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h) usage; exit 0 ;;
        *) ARGS+=("$arg") ;;
    esac
done
if [[ ${#ARGS[@]} -lt 2 || "${ARGS[0]:-}" != realworld || "${ARGS[1]:-}" != umi_left_franka_tcp_rot6d_eval ]]; then
    usage >&2
    exit 2
fi
LOG_DIR="${EVAL_LOG_DIR:-${TRAINING_EVAL_LOG_DIR:-${REPO_PATH}/logs/eval/$(date +'%Y%m%d-%H%M%S')-umi_left_franka_tcp_rot6d_eval}}"
CMD=("$EVAL_PYTHON" "${EVALUATIONS_PATH}/eval_embodied_agent.py"
    --config-path "${EVALUATIONS_PATH}/realworld/"
    --config-name umi_left_franka_tcp_rot6d_eval
    "runner.logger.log_path=${LOG_DIR}" "${ARGS[@]:2}")
if "$DRY_RUN"; then
    printf 'Dry run (command only): '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
command -v "$EVAL_PYTHON" >/dev/null 2>&1 || { echo "Evaluator Python is not executable: $EVAL_PYTHON" >&2; exit 2; }
mkdir -p "$LOG_DIR"
printf '%q ' "${CMD[@]}" | tee "${LOG_DIR}/eval_embodiment.log"
printf '\n' | tee -a "${LOG_DIR}/eval_embodiment.log"
"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/eval_embodiment.log"
