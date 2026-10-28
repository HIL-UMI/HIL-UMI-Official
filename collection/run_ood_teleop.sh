#!/usr/bin/env bash
# Modified for HIL-UMI.
# Launch the external data_recorder 0.1.21 with one installed HIL-UMI monitor.
# Usage: run_ood_teleop.sh [--mode pi05|advantage] [--ood-config FILE]
#                         [--check] RECORDER_CONFIG [record arguments ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HIL_UMI_ROOT="${HIL_UMI_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export COLLECTION_OOD_SOCKET_PATH="${COLLECTION_OOD_SOCKET_PATH:-/tmp/collection_ood_state.sock}"
MODE="pi05"
CHECK_ONLY=0
OOD_CONFIG=""
usage() {
    echo "Usage: $0 [--mode pi05|advantage] [--ood-config FILE] [--check] RECORDER_CONFIG [record args...]"
    echo "Set RECORDER_PYTHON and RECORD_BIN to your external recorder environment."
    echo "Set POLICY_PYTHON to .venv-policy/bin/python for pi05 mode."
    echo "Set ADVANTAGE_PYTHON to .venv-advantage/bin/python for advantage mode."
    echo "Optional OOD_PYTHON explicitly overrides the interpreter for the selected mode."
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode|--ood-config)
            if [[ $# -lt 2 ]]; then usage >&2; exit 2; fi
            if [[ "$1" == --mode ]]; then MODE="$2"; else OOD_CONFIG="$2"; fi
            shift 2 ;;
        --mode=*) MODE="${1#--mode=}"; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "Unknown launcher option: $1" >&2; usage >&2; exit 2 ;;
        *) break ;;
    esac
done
if [[ "${MODE}" != pi05 && "${MODE}" != advantage ]]; then
    echo "Unsupported OOD mode: ${MODE} (expected pi05 or advantage)." >&2
    exit 2
fi
if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
if [[ -z "${RECORDER_PYTHON:-}" || -z "${RECORD_BIN:-}" ]]; then
    echo "External recorder required: set RECORDER_PYTHON and RECORD_BIN; no recorder is bundled." >&2
    exit 2
fi
if [[ -z "${OOD_PYTHON:-}" ]]; then
    if [[ "${MODE}" == advantage ]]; then
        OOD_PYTHON="${ADVANTAGE_PYTHON:-}"
        MODEL_ENVIRONMENT="ADVANTAGE_PYTHON (.venv-advantage)"
    else
        OOD_PYTHON="${POLICY_PYTHON:-}"
        MODEL_ENVIRONMENT="POLICY_PYTHON (.venv-policy)"
    fi
fi
if [[ -z "${OOD_PYTHON:-}" ]]; then
    echo "Set ${MODEL_ENVIRONMENT} for ${MODE} mode, or supply an explicit OOD_PYTHON override." >&2
    exit 2
fi
echo "Selected ${MODE} monitor interpreter: ${OOD_PYTHON}"
for EXECUTABLE in "${RECORDER_PYTHON}" "${RECORD_BIN}" "${OOD_PYTHON}"; do
    if [[ ! -x "${EXECUTABLE}" ]]; then
        echo "External executable not found or not executable: ${EXECUTABLE}" >&2
        exit 2
    fi
done
RECORDER_CONFIG="$1"
if [[ ! -f "${RECORDER_CONFIG}" ]]; then
    echo "Recorder config not found: ${RECORDER_CONFIG}" >&2
    exit 2
fi
if [[ "${MODE}" == advantage ]]; then
    OOD_CONFIG="${OOD_CONFIG:-${SCRIPT_DIR}/config/advantage_ood.yaml}"
    OOD_MODULE="ood.advantage_monitor"
    export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
else
    OOD_CONFIG="${OOD_CONFIG:-${SCRIPT_DIR}/config/ood.yaml}"
    OOD_MODULE="ood.monitor"
fi

# Read package files without importing the hardware-facing recorder module.
RECORDER_RING_MODULE="$(env -u PYTHONPATH "${RECORDER_PYTHON}" -c '
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
try:
    if version("data-recorder") != "0.1.21":
        raise RuntimeError("only data_recorder 0.1.21 is supported")
    spec = find_spec("data_recorder")
    root = Path(spec.origin).parent
    recorder = (root / "recorder.py").read_text(encoding="utf-8")
    camera = (root / "camera_worker.py").read_text(encoding="utf-8")
    for marker in ("_ood_lifecycle_payload", "\"message_type\": \"state\"", "\"ood_ready\": bool(_ood_ready)", "if is_new_oculus_sample:"):
        if marker not in recorder:
            raise RuntimeError("installed recorder lacks the continuous OOD patch")
    if "capture_active = bool(capture_event.is_set())" not in camera:
        raise RuntimeError("installed recorder lacks continuous RGB shared memory")
    print(root / "shared_memory_ring.py")
except Exception as exc:
    raise SystemExit(f"Recorder preflight failed: {exc}. See README.md#collection-interface.")
')"
export COLLECTION_RECORDER_RING_MODULE="${COLLECTION_RECORDER_RING_MODULE:-${RECORDER_RING_MODULE}}"
env -u PYTHONPATH "${OOD_PYTHON}" -m ood.preflight \
    --mode "${MODE}" --config "${OOD_CONFIG}" \
    --recorder-config "${RECORDER_CONFIG}" --source-root "${SCRIPT_DIR}"
if [[ "${CHECK_ONLY}" == 1 ]]; then
    echo "Preflight passed. Hardware capture and model inference were not started."
    exit 0
fi

env -u PYTHONPATH "${OOD_PYTHON}" -m "${OOD_MODULE}" \
    --config "${OOD_CONFIG}" --recorder-config "${RECORDER_CONFIG}" &
OOD_PID=$!
_cleanup() {
    kill "${OOD_PID}" 2>/dev/null || true
    wait "${OOD_PID}" 2>/dev/null || true
}
trap _cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Translate only at the external subprocess boundary for historical recorder
# patches; current patches use the public COLLECTION_* variable directly.
env -u PYTHONPATH EGODAGGER_OOD_SOCKET_PATH="${COLLECTION_OOD_SOCKET_PATH}" \
    "${RECORD_BIN}" "$@"
