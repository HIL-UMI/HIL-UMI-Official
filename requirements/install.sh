#!/usr/bin/env bash
# Modified for HIL-UMI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="embodied"
MODEL="openpi"
ENV_NAME="franka"
VENV=""
PYTHON_VERSION="3.11"
DRY_RUN=false
usage() {
  cat <<'EOF'
Usage:
  bash requirements/install.sh embodied --model openpi --env franka --venv .venv-policy
  bash requirements/install.sh franka --venv .venv-franka
Options: --python VERSION (default 3.11), --dry-run, --help
Training requires Linux x86_64 and NVIDIA CUDA. ROS/Franka controllers and
hardware SDKs are external prerequisites; this script does not install them.
Policy uses .venv-policy (POLICY_PYTHON); Advantage uses .venv-advantage
(ADVANTAGE_PYTHON), installed separately from advantage/ with uv.
The policy, Advantage and Franka environments must remain separate.
EOF
}
while (($#)); do
  case "$1" in
    embodied|franka) TARGET="$1"; shift ;;
    --model|--env|--venv|--python)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in --model) MODEL="$2";; --env) ENV_NAME="$2";; --venv) VENV="$2";; --python) PYTHON_VERSION="$2";; esac
      shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unsupported option or target: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$MODEL" == openpi && "$ENV_NAME" == franka ]] || { echo "Only openpi + franka is included." >&2; exit 2; }
if [[ -z "$VENV" ]]; then
  if [[ "$TARGET" == franka ]]; then VENV=.venv-franka; else VENV=.venv-policy; fi
fi
[[ "$VENV" == /* ]] || VENV="$ROOT/$VENV"
if [[ "$TARGET" == franka ]]; then EXTRA=franka; else EXTRA=openpi; fi
if "$DRY_RUN"; then
  printf 'uv venv --python %q %q\n' "$PYTHON_VERSION" "$VENV"
  printf 'UV_PROJECT_ENVIRONMENT=%q uv sync --project %q --frozen --extra %q\n' "$VENV" "$ROOT" "$EXTRA"
  if [[ "$TARGET" == embodied ]]; then printf '%s\n' 'Apply the pinned OpenPI transformer replacements within the selected environment.'; fi
  exit 0
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "Runtime installation supports Linux x86_64; use the documented CPU checks on other systems." >&2; exit 1; }
command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 1; }
cd "$ROOT"
uv venv --python "$PYTHON_VERSION" "$VENV"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --project "$ROOT" --frozen --extra "$EXTRA"
if [[ "$TARGET" == embodied ]]; then
  "$VENV/bin/python" - <<'PY'
from pathlib import Path
import shutil
import openpi
import transformers
source = Path(openpi.__file__).parent / 'models_pytorch' / 'transformers_replace'
target = Path(transformers.__file__).parent
if not source.is_dir():
    raise SystemExit('The pinned OpenPI transformer replacements are missing.')
for item in source.iterdir():
    if item.name.startswith('__'):
        continue
    destination = target / item.name
    if item.is_dir():
        shutil.copytree(item, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(item, destination)
print('Installed the pinned OpenPI transformer replacements.')
PY
fi
printf 'Ready: %s\n' "$VENV"
