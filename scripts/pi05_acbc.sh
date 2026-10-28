#!/bin/bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash scripts/pi05_acbc.sh <dataset_path> <task_name> <gpus>" \
    "[fast] [run_vla_sft options/Hydra overrides]"
  echo "Example: bash scripts/pi05_acbc.sh dataset/fold_towel_acbc fold_towel_acbc 0-7"
  echo "  fast: compute no-padding norm stats on the selected GPUs and use" \
    "<task_name>_no_padding_gpu/norm_stats.json for training"
  echo "  Aliases: --fast-norm-stats, --norm-stats-mode fast"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

dataset_path="$1"
task_name="$2"
use_gpus="$3"
shift 3

norm_stats_mode="standard"
dry_run="false"
extra_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    fast|--fast-norm-stats)
      norm_stats_mode="fast"
      shift
      ;;
    --norm-stats-mode)
      [ "$#" -ge 2 ] || {
        echo "ERROR: --norm-stats-mode requires 'standard' or 'fast'."
        exit 1
      }
      norm_stats_mode="$2"
      shift 2
      ;;
    --norm-stats-mode=*)
      norm_stats_mode="${1#*=}"
      shift
      ;;
    --dry-run)
      dry_run="true"
      extra_args+=("$1")
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

case "${norm_stats_mode}" in
  standard|fast) ;;
  *)
    echo "ERROR: --norm-stats-mode must be 'standard' or 'fast', got '${norm_stats_mode}'."
    exit 1
    ;;
esac

if [[ ! "${task_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: task_name may contain only letters, digits, dot, underscore, and hyphen."
  exit 1
fi

if ! dataset_root="$(realpath -e "${dataset_path}")"; then
  echo "ERROR: ACBC dataset does not exist: ${dataset_path}"
  exit 1
fi
case "${dataset_root}/" in
  "${REPO_ROOT}/"*) ;;
  *)
    echo "ERROR: ACBC dataset must be physically stored inside the HIL-UMI repository."
    echo "Resolved dataset: ${dataset_root}"
    echo "HIL-UMI root: ${REPO_ROOT}"
    exit 1
    ;;
esac

python toolkits/lerobot/validate_acbc_dataset.py \
  "${dataset_root}" \
  --repo-root "${REPO_ROOT}"

export UMI_LEFT_FRANKA_ASSETS_DIR="${REPO_ROOT}/assets/pi05_singlefranka_tcp_rot6d"
stats_task_name="${task_name}"
stats_gpu_ids=""
stats_gpu_count=0

if [ "${norm_stats_mode}" = "fast" ]; then
  stats_task_name="${task_name}_no_padding_gpu"
  if [[ "${use_gpus}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    gpu_start="${BASH_REMATCH[1]}"
    gpu_end="${BASH_REMATCH[2]}"
    if [ "${gpu_end}" -lt "${gpu_start}" ]; then
      echo "ERROR: invalid GPU range '${use_gpus}'."
      exit 1
    fi
    for ((gpu = gpu_start; gpu <= gpu_end; gpu++)); do
      if [ -n "${stats_gpu_ids}" ]; then
        stats_gpu_ids+=","
      fi
      stats_gpu_ids+="${gpu}"
      stats_gpu_count=$((stats_gpu_count + 1))
    done
  elif [[ "${use_gpus}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    stats_gpu_ids="${use_gpus}"
    IFS=',' read -r -a stats_gpu_array <<< "${use_gpus}"
    stats_gpu_count="${#stats_gpu_array[@]}"
  else
    echo "ERROR: fast norm stats require <gpus> like '0-7' or" \
      "'0,1,2,3'; got '${use_gpus}'."
    exit 1
  fi
fi

norm_stats_dir="${UMI_LEFT_FRANKA_ASSETS_DIR}/umi_left_franka_tcp_rot6d/${stats_task_name}"

if [ "${dry_run}" = "true" ]; then
  if [ "${norm_stats_mode}" = "fast" ]; then
    echo "Dry run: fast GPU norm stats would use CUDA_VISIBLE_DEVICES=${stats_gpu_ids}" \
      "with ${stats_gpu_count} processes."
  fi
  echo "Dry run: skipping norm-stat computation; expected path is ${norm_stats_dir}"
elif [ "${norm_stats_mode}" = "fast" ]; then
  echo "Computing no-padding norm stats on GPUs ${stats_gpu_ids}."
  CUDA_VISIBLE_DEVICES="${stats_gpu_ids}" \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${stats_gpu_count}" \
    toolkits/lerobot/calculate_norm_stats_parquet_fast.py \
    --config-name pi05_singlefranka_tcp_rot6d \
    --repo-id "${dataset_root}" \
    --task-name "${task_name}" \
    --output-task-name "${stats_task_name}" \
    --stats-device cuda

  if [ ! -f "${norm_stats_dir}/norm_stats.json" ]; then
    echo "ERROR: fast norm_stats.json not found at ${norm_stats_dir}/norm_stats.json"
    exit 1
  fi
else
  python toolkits/lerobot/calculate_norm_stats.py \
    --config-name pi05_singlefranka_tcp_rot6d \
    --repo-id "${dataset_root}" \
    --task-name "${task_name}"
fi

bash "${REPO_ROOT}/scripts/run_vla_sft.sh" umi_left_franka_sft_openpi_pytorch_tcp_rot6d \
  --task_name "${task_name}" \
  --data_path "${dataset_root}" \
  --norm_stats_path="${norm_stats_dir}" \
  --max_steps 5000 \
  --save_interval 1000 \
  --gpus "${use_gpus}" \
  "${extra_args[@]}" \
  data.prompt_from_task=true \
  data.acbc=true \
  runner.logger.experiment_name=umi_left_franka_tcp_rot6d_openpi_pytorch_acbc
