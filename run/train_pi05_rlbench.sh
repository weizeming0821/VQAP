#!/usr/bin/env sh

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "${REPO_ROOT}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/openpi_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/LeRobot_RLBench_Dataset}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/.cache/huggingface/datasets}"



CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export CUDA_VISIBLE_DEVICES
NPROC=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

# --standalone：单机多卡，rendezvous 自动选空闲端口。
# 不加则固定占用 29500，与并行跑的另一个 torchrun 任务（如 VQAP 消融预训练）撞端口。
torchrun --standalone --nproc_per_node="${NPROC}" scripts/train_pi05_rlbench.py "$@"
