#!/usr/bin/env bash
# Modified for HIL-UMI.
# Build a derivative of a user-supplied data_recorder 0.1.21 wheel.
# The source is never overwritten, downloaded, or installed by this script.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 /absolute/path/data_recorder-0.1.21-py3-none-any.whl [output-directory]" >&2
    echo "Output defaults to collection/.local/patched-wheels (ignored by Git)." >&2
    exit 2
fi
PATCH_PYTHON="${PATCH_PYTHON:-${RECORDER_PYTHON:-python3}}"
OUTPUT_DIR="${2:-${COLLECTION_ROOT}/.local/patched-wheels}"
"${PATCH_PYTHON}" "${SCRIPT_DIR}/patch_recorder_for_ood.py" --wheel "$1" --output-dir "${OUTPUT_DIR}"
