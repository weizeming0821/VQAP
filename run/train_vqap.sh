#!/usr/bin/env sh

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "${REPO_ROOT}"

# 允许外部覆盖选卡（共享服务器上需按空闲情况挑卡）；未设置时沿用默认四卡。
# 与 M0 训练并行时务必分配**不重叠**的 GPU。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES
NPROC=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

# --standalone：单机多卡，rendezvous 自动选空闲端口。
# 不加则固定占用 29500，与并行跑的另一个 torchrun 任务（如 M0 训练）撞端口。
torchrun --standalone --nproc_per_node="${NPROC}" scripts/train_vqap.py "$@"
