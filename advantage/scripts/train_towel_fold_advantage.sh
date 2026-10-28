#!/usr/bin/env bash
# Modified for HIL-UMI.
set -euo pipefail
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
exec bash "${REPO_ROOT}/advantage" train-base "${EXP_NAME:-towel_fold_advantage_v1}" "$@"
