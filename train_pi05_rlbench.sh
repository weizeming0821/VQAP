#!/usr/bin/env sh

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "${REPO_ROOT}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/openpi_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/LeRobot_RLBench_Dataset}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/.cache/huggingface/datasets}"


CUDA_VISIBLE_DEVICES="0,1,2,4" torchrun --nproc_per_node=4 scripts/train_pi05_rlbench.py "$@"
