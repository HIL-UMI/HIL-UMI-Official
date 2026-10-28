#!/usr/bin/env bash
# Modified for HIL-UMI.
set -euo pipefail
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/source_runtime.sh"
advantage_init_runtime "${REPO_ROOT}"
cd "${REPO_ROOT}"
exec "${ADVANTAGE_PYTHON}" scripts/label_advantage_datasets.py "$@"
