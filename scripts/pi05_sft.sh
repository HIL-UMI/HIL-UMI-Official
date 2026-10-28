#!/bin/bash

set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash scripts/pi05_sft.sh <repo_id> <task_name> <prompt> <gpus> [training options/Hydra overrides]"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

repo_id="$1"
task_name="$2"
prompt="$3"
use_gpus="$4"
shift 4

converted_repo_id="dataset/${task_name}_franka"
converted_task_name="${task_name}_franka"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}}"
python toolkits/lerobot/convert_umi_left_to_single_franka.py \
  --src "${repo_id}" \
  --dst "${converted_repo_id}" \
  --prompt "${prompt}"

python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_singlefranka_tcp_rot6d \
  --repo-id "${converted_repo_id}" \
  --task-name "${converted_task_name}"

bash "${REPO_ROOT}/scripts/run_vla_sft.sh" umi_left_franka_sft_openpi_pytorch_tcp_rot6d \
  --task_name "${converted_task_name}" \
  --data_path "${converted_repo_id}" \
  --max_steps 20000 \
  --save_interval 10000 \
  --gpus "${use_gpus}" \
  "$@"
