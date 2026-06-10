"""训练日志与 TensorBoard 实验追踪工具模块。
    - init_logger      : 初始化 Python logging（控制台 + 文件）
    - init_tensorboard : 初始化 TensorBoard SummaryWriter（仅 rank 0）
    - finish_tensorboard : 安全关闭 TensorBoard writer
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


"""初始化训练日志，仅在 rank 0 上创建控制台与文件输出。"""
def init_logger(rank: int, exp_name: str, log_dir: str, is_resume: bool) -> logging.Logger:
	logger_name = f"vqap.{exp_name}"
	logger = logging.getLogger(logger_name)
	logger.handlers.clear()
	logger.propagate = False

	if rank != 0:
		logger.addHandler(logging.NullHandler())
		logger.setLevel(logging.CRITICAL)
		return logger

	log_root = Path(log_dir).expanduser()
	log_root.mkdir(parents=True, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	suffix = "_resume" if is_resume else ""
	log_path = log_root / f"{exp_name}{suffix}_{ts}.log"
	file_mode = "w"

	formatter = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(formatter)

	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(formatter)

	logger.setLevel(logging.DEBUG)
	logger.addHandler(file_handler)
	logger.addHandler(console_handler)
	return logger


"""初始化 TensorBoard SummaryWriter，只在 rank 0 且显式启用时创建。
    Input:
        rank         : 当前进程 rank，仅 rank 0 创建 writer
        exp_name     : 实验名称，作为 TensorBoard 日志子目录名
        cfg          : 完整训练配置字典，含 "tensorboard" 子节
        ckpt_dir     : 保留参数以兼容调用方，当前未使用（日志按 log_dir/exp_name 存放）
        is_resume    : 是否为续训（保留参数以保持接口兼容，TensorBoard 无需特殊处理）

    Output:
        SummaryWriter 或 None
"""
def init_tensorboard(
	rank: int,
	exp_name: str,
	cfg: Dict[str, Any],
	ckpt_dir: str,
	is_resume: bool,
) -> Optional[SummaryWriter]:
	tb_cfg = cfg.get("tensorboard", {})
	if rank != 0 or not bool(tb_cfg.get("enable", False)):
		return None

	logger = logging.getLogger(f"vqap.{exp_name}")
	# TensorBoard 日志直接放在项目根目录下的 log_dir/exp_name，避免与 checkpoint 目录嵌套过深。
	tb_log_dir = Path(str(tb_cfg.get("log_dir", "tensorboard"))).expanduser() / exp_name
	tb_log_dir.mkdir(parents=True, exist_ok=True)

	try:
		writer = SummaryWriter(log_dir=str(tb_log_dir))
	except Exception as error:
		if logger.handlers:
			logger.warning(
				"TensorBoard init failed; continue training without TensorBoard. reason=%s",
				error,
				exc_info=True,
			)
		return None

	# 将训练超参数以文本形式记录到 TensorBoard，便于后续查阅
	hyperparams_str = _format_hyperparams(cfg)
	writer.add_text("hyperparameters", hyperparams_str, global_step=0)

	logger.info("TensorBoard writer initialized at %s", tb_log_dir)
	return writer


"""将训练配置字典格式化为 Markdown 代码块文本，写入 TensorBoard TEXT 面板。"""
def _format_hyperparams(cfg: Dict[str, Any]) -> str:
	import yaml
	yaml_str = yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
	return f"```yaml\n{yaml_str}```"


"""安全关闭 TensorBoard writer。"""
def finish_tensorboard(rank: int, writer: Optional[SummaryWriter]) -> None:
	if rank == 0 and writer is not None:
		writer.close()
