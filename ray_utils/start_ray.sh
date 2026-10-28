#!/usr/bin/env bash
# Modified for HIL-UMI.
# Start an explicitly configured Ray node; no shared IP file or memory override.
set -euo pipefail
DRY_RUN=false
usage() {
    echo "Usage: TRAINING_NODE_RANK=0 [RAY_PORT=6379] bash $0 [--dry-run]"
    echo "Worker nodes additionally require RAY_ADDRESS=<head-host>:<port>."
    echo "Optional RAY_NODE_IP_ADDRESS is forwarded as --node-ip-address."
}
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unsupported argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "${TRAINING_NODE_RANK:-}" =~ ^[0-9]+$ ]] || { echo "Set TRAINING_NODE_RANK to a nonnegative integer before starting Ray." >&2; exit 2; }
RAY_PORT="${RAY_PORT:-6379}"
[[ "$RAY_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && ((10#$RAY_PORT <= 65535)) || { echo "RAY_PORT must be 1..65535." >&2; exit 2; }
CMD=(ray start)
if [[ "$TRAINING_NODE_RANK" =~ ^0+$ ]]; then
    CMD+=(--head "--port=$RAY_PORT")
else
    [[ "${RAY_ADDRESS:-}" =~ ^([A-Za-z0-9._-]+|\[[a-fA-F0-9:]+\]):([1-9][0-9]{0,4})$ ]] || { echo "Worker nodes require RAY_ADDRESS=<head-host>:<port>." >&2; exit 2; }
    [[ "${BASH_REMATCH[2]}" -le 65535 ]] || { echo "RAY_ADDRESS port must be 1..65535." >&2; exit 2; }
    CMD+=("--address=$RAY_ADDRESS")
fi
if [[ -n "${RAY_NODE_IP_ADDRESS:-}" ]]; then CMD+=("--node-ip-address=$RAY_NODE_IP_ADDRESS"); fi
printf '%q ' "${CMD[@]}"
printf '\n'
if "$DRY_RUN"; then exit 0; fi
command -v ray >/dev/null 2>&1 || { echo "Ray is not installed in the selected environment." >&2; exit 2; }
exec "${CMD[@]}"
