#!/bin/bash
# Modified for HIL-UMI.

set -euo pipefail

usage() {
  echo "Usage: bash scripts/pi05_mix_acbc.sh <group_a_path> <group_b_path>" \
    "<task_name> <gpus> <sft_ckpt> <sft_norm_stats>" \
    "<post_train_init> [options/run_vla_sft overrides]"
  echo "Options:"
  echo "  --group-a-ratio <float>       Group A relative weight (default: 0.5)."
  echo "  --group-b-ratio <float>       Group B relative weight (default: 0.5)."
  echo "  --group-b-prompt <text>       Override Group B's base ACBC prompt."
  echo "  --norm-stats-device gpu|cpu   Stats device (default: gpu)."
  echo "Example: bash scripts/pi05_mix_acbc.sh" \
    "dataset/fold_towel_adv_stage1/base" \
    "dataset/fold_towel_adv_stage1/stage1 fold_towel_mix_acbc 0-7" \
    "/path/to/global_step_5000 /path/to/norm_stats.json" \
    "/path/to/weights_only"
}

if [ "$#" -lt 7 ]; then
  usage
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

group_a_path="$1"
group_b_path="$2"
task_name="$3"
use_gpus="$4"
sft_ckpt="$5"
sft_norm_stats="$6"
post_train_init="$7"
shift 7

export SFT_CKPT="${sft_ckpt}"
export SFT_NORM_STATS="${sft_norm_stats}"
export POST_TRAIN_INIT="${post_train_init}"

group_a_ratio="0.5"
group_b_ratio="0.5"
group_b_prompt=""
norm_stats_device="gpu"
dry_run="false"
extra_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --group-a-ratio)
      [ "$#" -ge 2 ] || { echo "ERROR: --group-a-ratio requires a value."; exit 1; }
      group_a_ratio="$2"
      shift 2
      ;;
    --group-a-ratio=*)
      group_a_ratio="${1#*=}"
      shift
      ;;
    --group-b-ratio)
      [ "$#" -ge 2 ] || { echo "ERROR: --group-b-ratio requires a value."; exit 1; }
      group_b_ratio="$2"
      shift 2
      ;;
    --group-b-ratio=*)
      group_b_ratio="${1#*=}"
      shift
      ;;
    --group-b-prompt)
      [ "$#" -ge 2 ] || { echo "ERROR: --group-b-prompt requires a value."; exit 1; }
      [ -n "$2" ] || { echo "ERROR: --group-b-prompt cannot be empty."; exit 1; }
      group_b_prompt="$2"
      shift 2
      ;;
    --group-b-prompt=*)
      group_b_prompt="${1#*=}"
      [ -n "${group_b_prompt}" ] || { echo "ERROR: --group-b-prompt cannot be empty."; exit 1; }
      shift
      ;;
    --norm-stats-device)
      [ "$#" -ge 2 ] || { echo "ERROR: --norm-stats-device requires a value."; exit 1; }
      norm_stats_device="$2"
      shift 2
      ;;
    --norm-stats-device=*)
      norm_stats_device="${1#*=}"
      shift
      ;;
    --dry-run)
      dry_run="true"
      extra_args+=("--dry-run")
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! "${task_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: task_name may contain only letters, digits, dot, underscore, and hyphen."
  exit 1
fi

validate_ratio() {
  local name="$1"
  local value="$2"
  if ! awk -v value="${value}" 'BEGIN {
      if (value !~ /^[+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?$/ || value + 0 <= 0) {
        exit 1
      }
    }'; then
    echo "ERROR: ${name} must be a positive finite number, got '${value}'."
    exit 1
  fi
}

validate_ratio "--group-a-ratio" "${group_a_ratio}"
validate_ratio "--group-b-ratio" "${group_b_ratio}"

case "${norm_stats_device}" in
  gpu|cpu) ;;
  *)
    echo "ERROR: --norm-stats-device must be 'gpu' or 'cpu', got '${norm_stats_device}'."
    exit 1
    ;;
esac

resolve_dataset_path() {
  local requested="$1"
  local resolved
  if ! resolved="$(realpath -e "${requested}")"; then
    echo "ERROR: Mix ACBC dataset does not exist: ${requested}" >&2
    exit 1
  fi
  case "${resolved}/" in
    "${REPO_ROOT}/"*) ;;
    *)
      echo "ERROR: Mix ACBC datasets must be physically stored inside the HIL-UMI repository." >&2
      echo "Resolved dataset: ${resolved}" >&2
      echo "HIL-UMI root: ${REPO_ROOT}" >&2
      exit 1
      ;;
  esac
  printf "%s" "${resolved}"
}

detect_dataset_schema() {
  python - "$1" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
features = info.get("features", {})
canonical = (
    features.get("state", {}).get("shape") == [10]
    and features.get("actions", {}).get("shape") == [10]
    and features.get("image", {}).get("dtype") == "video"
    and features.get("extra_view_image", {}).get("dtype") == "video"
)
raw_umi = (
    features.get("observation.state", {}).get("shape") == [47]
    and features.get("action", {}).get("shape") == [30]
    and features.get("observation.images.cam_left", {}).get("dtype") == "video"
    and features.get("observation.images.cam_front", {}).get("dtype") == "video"
)
if canonical:
    print("canonical")
elif raw_umi:
    print("raw_umi")
else:
    raise SystemExit(
        "unsupported Mix ACBC schema: expected canonical state/actions [10] "
        "or raw UMI observation.state [47] and action [30]"
    )
PY
}

override_acbc_prompt() {
  local dataset_root="$1"
  local task_prompt="$2"
  python - "${dataset_root}" "${task_prompt}" <<'PY'
import json
import os
import pathlib
import re
import sys

dataset_root = pathlib.Path(sys.argv[1])
task_prompt = sys.argv[2].strip()
if not task_prompt or "\n" in task_prompt or "\r" in task_prompt:
    raise SystemExit("Group B prompt must be a non-empty, single-line string")
if ", Advantage: " in task_prompt:
    raise SystemExit("Group B prompt must not include an 'Advantage:' suffix")

advantage_prompt = re.compile(r"^.+, Advantage: (?P<label>[^\r\n]+)$")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl_atomic(path, rows):
    temporary_path = path.with_name(f".{path.name}.prompt.tmp")
    temporary_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def replace_prompt(value):
    if not isinstance(value, str):
        raise SystemExit(f"ACBC task prompt must be a string, got {value!r}")
    match = advantage_prompt.fullmatch(value)
    if match is None:
        raise SystemExit(
            "ACBC task prompt must have the form '<task>, Advantage: <label>', "
            f"got {value!r}"
        )
    return f"{task_prompt}, Advantage: {match.group('label')}"


meta_root = dataset_root / "meta"
tasks_path = meta_root / "tasks.jsonl"
episodes_path = meta_root / "episodes.jsonl"
task_rows = read_jsonl(tasks_path)
episode_rows = read_jsonl(episodes_path)

for row in task_rows:
    row["task"] = replace_prompt(row.get("task"))
for row in episode_rows:
    tasks = row.get("tasks", [])
    if not isinstance(tasks, list):
        raise SystemExit("ACBC episodes.jsonl tasks must be a list")
    row["tasks"] = [replace_prompt(task) for task in tasks]

write_jsonl_atomic(tasks_path, task_rows)
write_jsonl_atomic(episodes_path, episode_rows)

mapping_path = meta_root / "umi_left_franka_mapping.json"
if mapping_path.is_file():
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["group_b_prompt"] = task_prompt
    temporary_path = mapping_path.with_name(f".{mapping_path.name}.prompt.tmp")
    temporary_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, mapping_path)

print(f"Updated Group B ACBC prompt in {dataset_root}: {task_prompt}")
PY
}

prepare_group() {
  local source_root="$1"
  local group_label="$2"
  local acbc_task_prompt="${3:-}"
  local schema
  local cache_root
  schema="$(detect_dataset_schema "${source_root}")"
  if [ "${schema}" = "canonical" ]; then
    if [ -n "${acbc_task_prompt}" ]; then
      echo "ERROR: --group-b-prompt can only be applied while converting raw UMI data." >&2
      echo "Group B is already canonical: ${source_root}" >&2
      exit 1
    fi
    python toolkits/lerobot/validate_acbc_dataset.py \
      "${source_root}" \
      --repo-root "${REPO_ROOT}"
    PREPARED_GROUP_PATH="${source_root}"
    return
  fi

  cache_root="${source_root}_single_franka_acbc"
  python toolkits/lerobot/convert_umi_left_to_single_franka.py \
    --src "${source_root}" \
    --validate-acbc-source-only
  if [ "${dry_run}" = "true" ]; then
    if [ -d "${cache_root}" ]; then
      python toolkits/lerobot/convert_umi_left_to_single_franka.py \
        --src "${source_root}" \
        --dst "${cache_root}" \
        --preserve-acbc-tasks \
        --reuse-existing
      python toolkits/lerobot/validate_acbc_dataset.py \
        "${cache_root}" \
        --repo-root "${REPO_ROOT}"
    else
      echo "Dry run: ${group_label} raw UMI data would be converted to ${cache_root}"
    fi
    if [ -n "${acbc_task_prompt}" ]; then
      echo "Dry run: ${group_label} base ACBC prompt would be '${acbc_task_prompt}'"
    fi
  else
    python toolkits/lerobot/convert_umi_left_to_single_franka.py \
      --src "${source_root}" \
      --dst "${cache_root}" \
      --preserve-acbc-tasks \
      --reuse-existing
    if [ -n "${acbc_task_prompt}" ]; then
      override_acbc_prompt "${cache_root}" "${acbc_task_prompt}"
    fi
    python toolkits/lerobot/validate_acbc_dataset.py \
      "${cache_root}" \
      --repo-root "${REPO_ROOT}"
  fi
  PREPARED_GROUP_PATH="${cache_root}"
}

group_a_root="$(resolve_dataset_path "${group_a_path}")"
group_b_root="$(resolve_dataset_path "${group_b_path}")"
prepare_group "${group_a_root}" "group A" ""
prepared_group_a="${PREPARED_GROUP_PATH}"
prepare_group "${group_b_root}" "group B" "${group_b_prompt}"
prepared_group_b="${PREPARED_GROUP_PATH}"

if [ "${dry_run}" = "true" ]; then
  echo "Dry run: would convert SFT checkpoint to weights-only model:"
  echo "  ckpt:       ${SFT_CKPT}"
  echo "  norm stats: ${SFT_NORM_STATS}"
  echo "  output:     ${POST_TRAIN_INIT}"
else
  python -m training.utils.ckpt_convertor.openpi.convert --mode sft2new \
    --ckpt "${SFT_CKPT}" \
    --input-norm-stats "${SFT_NORM_STATS}" \
    --output-model "${POST_TRAIN_INIT}" \
    --output-norm-stats "${POST_TRAIN_INIT}/stage1/norm_stats.json"
fi

stats_gpu_ids=""
stats_gpu_count=0
if [ "${norm_stats_device}" = "gpu" ]; then
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
    echo "ERROR: GPU norm stats require <gpus> like '0-7' or '0,1,2,3'; got '${use_gpus}'."
    echo "Use --norm-stats-device cpu to compute stats without CUDA."
    exit 1
  fi
fi

sanitize_name() {
  printf "%s" "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

group_a_name="$(sanitize_name "$(basename "${prepared_group_a}")")"
group_b_name="$(sanitize_name "$(basename "${prepared_group_b}")")"
group_a_ratio_name="$(sanitize_name "${group_a_ratio}")"
group_b_ratio_name="$(sanitize_name "${group_b_ratio}")"
stats_name_prefix="${task_name}"
if [[ "${stats_name_prefix}" != *mix_acbc* ]]; then
  stats_name_prefix+="_mix_acbc"
fi
stats_task_name="${stats_name_prefix}_${group_a_ratio_name}_${group_b_ratio_name}_${group_a_name}_${group_b_name}_${norm_stats_device}"

export UMI_LEFT_FRANKA_ASSETS_DIR="${REPO_ROOT}/assets/pi05_singlefranka_tcp_rot6d"
norm_stats_dir="${UMI_LEFT_FRANKA_ASSETS_DIR}/umi_left_franka_tcp_rot6d/${stats_task_name}"

stats_args=(
  toolkits/lerobot/calculate_norm_stats_parquet_fast.py
  --config-name pi05_singlefranka_tcp_rot6d
  --repo-id "${prepared_group_a}"
  --task-name "${task_name}"
  --post-train true
  --post-train-data-path "${prepared_group_b}"
  --post-train-demo-ratio "${group_a_ratio}"
  --post-train-correction-ratio "${group_b_ratio}"
  --post-train-seed 0
  --output-task-name "${stats_task_name}"
)

if [ "${dry_run}" = "true" ]; then
  if [ "${norm_stats_device}" = "gpu" ]; then
    echo "Dry run: GPU norm stats would use CUDA_VISIBLE_DEVICES=${stats_gpu_ids}" \
      "with ${stats_gpu_count} processes."
  else
    echo "Dry run: norm stats would use the fast parquet CPU path."
  fi
  echo "Dry run: expected norm stats path is ${norm_stats_dir}"
else
  if [ "${norm_stats_device}" = "gpu" ]; then
    echo "Computing Mix ACBC norm stats on GPUs ${stats_gpu_ids}."
    CUDA_VISIBLE_DEVICES="${stats_gpu_ids}" \
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python -m torch.distributed.run \
      --standalone \
      --nproc-per-node="${stats_gpu_count}" \
      "${stats_args[@]}" \
      --stats-device cuda
  else
    echo "Computing Mix ACBC norm stats on CPU."
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python "${stats_args[@]}" --stats-device cpu
  fi
fi

bash "${REPO_ROOT}/scripts/run_vla_sft.sh" umi_left_franka_sft_openpi_pytorch_tcp_rot6d \
  --task_name "${task_name}" \
  --data_path "${prepared_group_a}" \
  --norm_stats_path "${norm_stats_dir}" \
  --max_steps 5000 \
  --save_interval 1000 \
  --gpus "${use_gpus}" \
  "${extra_args[@]}" \
  "actor.model.model_path=${POST_TRAIN_INIT}" \
  "runner.resume_dir=null" \
  data.prompt_from_task=true \
  data.acbc=true \
  data.mix_acbc=true \
  data.mix_acbc_data_path="${prepared_group_b}" \
  data.mix_acbc_primary_ratio="${group_a_ratio}" \
  data.mix_acbc_secondary_ratio="${group_b_ratio}" \
  runner.logger.experiment_name=umi_left_franka_tcp_rot6d_openpi_pytorch_mix_acbc
