#!/usr/bin/env bash
# Select an existing source environment; never create or relocate environments.
advantage_init_runtime() {
    local repo_root="$1"
    local project_root
    project_root="$(cd -- "${repo_root}/.." && pwd)"
    local python="${ADVANTAGE_PYTHON:-${project_root}/.venv-advantage/bin/python}"
    if [[ ! -x "${python}" ]]; then
        echo "Error: Advantage interpreter not found: ${python}. From advantage/, run 'UV_PROJECT_ENVIRONMENT=\"${project_root}/.venv-advantage\" uv sync --frozen --group ood', or set ADVANTAGE_PYTHON to an existing Advantage interpreter." >&2
        return 1
    fi
    export ADVANTAGE_ROOT="${repo_root}"
    export ADVANTAGE_DATA_ROOT="${ADVANTAGE_DATA_ROOT:-${repo_root}/data}"
    export ADVANTAGE_CHECKPOINT_ROOT="${ADVANTAGE_CHECKPOINT_ROOT:-${repo_root}/checkpoints}"
    export ADVANTAGE_PI05_BASE="${ADVANTAGE_PI05_BASE:-${ADVANTAGE_CHECKPOINT_ROOT}/pi05_base}"
    export ADVANTAGE_OUTPUT_ROOT="${ADVANTAGE_OUTPUT_ROOT:-${repo_root}/outputs}"
    export ADVANTAGE_CACHE_ROOT="${ADVANTAGE_CACHE_ROOT:-${repo_root}/.cache}"
    export ADVANTAGE_PYTHON="${python}"
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export PYTHONPATH="${repo_root}/src:${repo_root}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
    export HF_HOME="${HF_HOME:-${ADVANTAGE_CACHE_ROOT}/huggingface}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ADVANTAGE_CACHE_ROOT}/hf_datasets}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
    export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ADVANTAGE_CACHE_ROOT}/openpi}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ADVANTAGE_CACHE_ROOT}/xdg}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${ADVANTAGE_CACHE_ROOT}/matplotlib}"
    export TORCH_HOME="${TORCH_HOME:-${ADVANTAGE_CACHE_ROOT}/torch}"
    export WANDB_DIR="${WANDB_DIR:-${ADVANTAGE_OUTPUT_ROOT}/wandb}"
    export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${ADVANTAGE_CACHE_ROOT}/wandb}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
}
