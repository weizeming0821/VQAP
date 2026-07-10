#!/usr/bin/env sh

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "${REPO_ROOT}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/openpi_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/LeRobot_RLBench_Dataset}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/.cache/huggingface/datasets}"
export DISPLAY="${DISPLAY:-:99}"

if [ -n "${COPPELIASIM_ROOT:-}" ]; then
  export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

python3 scripts/eval_pi05_rlbench.py "$@"
