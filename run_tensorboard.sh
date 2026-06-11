#!/bin/bash
# TensorBoard 启动脚本
# 用法: bash run_tensorboard.sh [port]

PORT=${1:-6006}
LOGDIR="$(dirname "$0")/tensorboard"

echo "=== TensorBoard ==="
echo "Logdir : $LOGDIR"
echo "Port   : $PORT"
echo "URL    : http://$(hostname -I | awk '{print $1}'):$PORT"
echo "===================="

# tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT" --bind_all
tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT"
