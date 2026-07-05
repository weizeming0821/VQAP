#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="4,5,6,7" torchrun --nproc_per_node=4 scripts/train_pi05_rlbench.py "$@"
