#!/bin/bash
# Modified for HIL-UMI.

set -euo pipefail

if [ "$#" -lt 10 ] || [ "$#" -gt 11 ]; then
  echo "Usage: bash scripts/pi05_post-train.sh <demo_data_path> <demo_task_name>" \
    "<prompt> <collection_task_name> <gpus> <sft_ckpt> <sft_norm_stats>" \
    "<post_train_init> <demo_ratio> <collection_ratio> [fast]"
  echo "  fast: compute no-padding norm stats on the selected GPUs and use" \
    "<demo_task_name>_no_padding_gpu/norm_stats.json for training"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

demo_data_path="$1"
task_name_demo="$2"
prompt="$3"
task_name_collection="$4"
use_gpus="$5"

sft_ckpt="$6"
sft_norm_stats="$7"
post_train_init="$8"

demo_ratio="${9}"
collection_ratio="${10}"
norm_stats_mode="${11:-}"
if [ -n "${norm_stats_mode}" ] && [ "${norm_stats_mode}" != "fast" ]; then
  echo "ERROR: optional norm-stats mode must be 'fast', got '${norm_stats_mode}'."
  exit 1
fi

repo_root="${REPO_ROOT}"
demo_task_name="${task_name_demo}"
collection_src="collection_dataset/${task_name_collection}"
post_train_data_path="${repo_root}/collection_dataset/${task_name_collection}-franka"

export HF_LEROBOT_HOME="${repo_root}"
export SFT_CKPT="${sft_ckpt}"
export SFT_NORM_STATS="${sft_norm_stats}"
export POST_TRAIN_INIT="${post_train_init}"

python toolkits/lerobot/convert_umi_left_to_single_franka.py \
  --src "${collection_src}" \
  --dst "${post_train_data_path}" \
  --prompt "${prompt}"

python -m training.utils.ckpt_convertor.openpi.convert --mode sft2new \
  --ckpt "${SFT_CKPT}" \
  --input-norm-stats "${SFT_NORM_STATS}" \
  --output-model "${POST_TRAIN_INIT}" \
  --output-norm-stats "${POST_TRAIN_INIT}/stage1/norm_stats.json"

if [ "${norm_stats_mode}" = "fast" ]; then
  stats_gpu_ids=""
  stats_gpu_count=0
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

  fast_output_task_name="${demo_task_name}_no_padding_gpu"
  echo "Computing no-padding norm stats on GPUs ${stats_gpu_ids}."
  CUDA_VISIBLE_DEVICES="${stats_gpu_ids}" \
  PYTHONPATH="${repo_root}:${PYTHONPATH:-}" \
  python -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${stats_gpu_count}" \
    toolkits/lerobot/calculate_norm_stats_parquet_fast.py \
    --config-name pi05_singlefranka_tcp_rot6d \
    --repo-id "${demo_data_path}" \
    --task-name "${demo_task_name}" \
    --post-train true \
    --post-train-data-path "${post_train_data_path}" \
    --post-train-demo-ratio "${demo_ratio}" \
    --post-train-correction-ratio "${collection_ratio}" \
    --post-train-seed 0 \
    --output-task-name "${fast_output_task_name}" \
    --stats-device cuda
else
  python toolkits/lerobot/calculate_norm_stats.py \
    --config-name pi05_singlefranka_tcp_rot6d \
    --repo-id "${demo_data_path}" \
    --task-name "${demo_task_name}" \
    --post-train true \
    --post-train-data-path "${post_train_data_path}" \
    --post-train-demo-ratio "${demo_ratio}" \
    --post-train-correction-ratio "${collection_ratio}"
fi

train_command=(
  bash "${REPO_ROOT}/scripts/run_vla_sft.sh"
  umi_left_franka_sft_openpi_pytorch_tcp_rot6d
  --task_name "${demo_task_name}"
  --data_path "${demo_data_path}"
  --post-train-model-path "${POST_TRAIN_INIT}"
  --post-train-data-path "${post_train_data_path}"
  --post-train-demo-ratio "${demo_ratio}"
  --post-train-correction-ratio "${collection_ratio}"
)
if [ "${norm_stats_mode}" = "fast" ]; then
  train_command+=(--fast-norm-stats)
fi
train_command+=(
  --max_steps 20000
  --save_interval 2000
  --gpus "${use_gpus}"
)
"${train_command[@]}"
