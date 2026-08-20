#!/usr/bin/env sh

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "${REPO_ROOT}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/openpi_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/LeRobot_RLBench_Dataset}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/.cache/huggingface/datasets}"
export DISPLAY="${DISPLAY:-:99}"
# The evaluator is PyTorch-only. Prevent Transformers from importing
# TensorFlow's LLVM runtime into the CoppeliaSim process.
export USE_TF="${USE_TF:-0}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"

if [ -n "${COPPELIASIM_ROOT:-}" ]; then
  export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  # tensorflow 的 libtensorflow_framework.so.2 导出了 471 个 OpenSSL 符号(BoringSSL 实现)，
  # 会劫持 CoppeliaSim 自带 libssl.so.1.1 的调用；Qt5Network 初始化 SSL 时 ABI 不匹配
  # → SIGSEGV(实测 2026-08-02 step30000 评测崩在 OPENSSL_init_ssl → EVP_get_digestbyname)。
  # 是否触发取决于动态库加载顺序，属潜伏雷：此前 4 次评测侥幸未中。
  # 预加载 CoppeliaSim 自己的 OpenSSL 1.1，使其在全局符号表中优先于 tensorflow。
  # 不影响 Python 的 ssl 模块——后者链接的是 soname 不同的 libssl.so.3。
  for _lib in libssl.so.1.1 libcrypto.so.1.1; do
    if [ -f "${COPPELIASIM_ROOT}/${_lib}" ]; then
      LD_PRELOAD="${COPPELIASIM_ROOT}/${_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
  done
  [ -n "${LD_PRELOAD:-}" ] && export LD_PRELOAD
fi

python3 scripts/test_pi05_rlbench.py "$@"
