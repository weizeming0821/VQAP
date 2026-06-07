import logging
from pathlib import Path
from typing import Any, Dict, Optional
import wandb



WANDB_ID_FILENAME = "wandb_id.txt"


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
	log_path = log_root / f"vqap_{exp_name}.log"
	file_mode = "a" if is_resume else "w"

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


def _wandb_id_path(ckpt_dir: str) -> Path:
	return Path(ckpt_dir).expanduser() / WANDB_ID_FILENAME


"""把当前 wandb run id 写入 checkpoint 目录，便于断点续训复用同一条曲线。"""
def save_wandb_run_id(ckpt_dir: str, run_id: str) -> None:
	run_id_path = _wandb_id_path(ckpt_dir)
	run_id_path.parent.mkdir(parents=True, exist_ok=True)
	run_id_path.write_text(f"{run_id}\n", encoding="utf-8")


"""从 checkpoint 目录读取已有的 wandb run id。"""
def load_wandb_run_id(ckpt_dir: str) -> Optional[str]:
	run_id_path = _wandb_id_path(ckpt_dir)
	if not run_id_path.is_file():
		return None
	run_id = run_id_path.read_text(encoding="utf-8").strip()
	return run_id or None


"""初始化 wandb，只在 rank 0 且显式启用时创建 run。"""
def init_wandb(
	rank: int,
	exp_name: str,
	cfg: Dict[str, Any],
	ckpt_dir: str,
	is_resume: bool,
) -> Any:
	wandb_cfg = cfg.get("wandb", {})
	if rank != 0 or not bool(wandb_cfg.get("enable", False)):
		return None

	run_id = load_wandb_run_id(ckpt_dir) if is_resume else None
	init_kwargs: Dict[str, Any] = {
		"project": wandb_cfg.get("project", "VQAP"),
		"name": exp_name,
		"group": wandb_cfg.get("group"),
		"tags": wandb_cfg.get("tags", []),
		"config": cfg,
	}
	if run_id is not None:
		init_kwargs["id"] = run_id
		init_kwargs["resume"] = "must"

	run = wandb.init(**init_kwargs)
	save_wandb_run_id(ckpt_dir, run.id)
	return run


"""结束 wandb run。"""
def finish_wandb(rank: int, run: Any) -> None:
	if rank == 0 and run is not None:
		run.finish()
