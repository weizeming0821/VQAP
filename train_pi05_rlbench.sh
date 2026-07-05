#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/openpi_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/LeRobot_RLBench_Dataset}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/.cache/huggingface/datasets}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CUDA_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  DEFAULT_NPROC="${#CUDA_IDS[@]}"
else
  DEFAULT_NPROC=1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC}}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_pi05_rlbench.py "$@"
