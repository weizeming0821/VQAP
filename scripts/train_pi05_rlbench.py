#!/usr/bin/env python3
"""PyTorch RLBench fine-tuning entrypoint for pi0.5."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any

# 在 import torch 之前设置分配器，避免 CUDA 默认缓存策略先被初始化。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "openpi-vqap"
OPENPI_SRC = OPENPI_ROOT / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

# 训练脚本默认指向项目内缓存，避免落到只读的用户目录。
os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / "openpi_cache"))
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO_ROOT / "LeRobot_RLBench_Dataset"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache" / "huggingface" / "datasets"))

import numpy as np
import torch
import torch.distributed as dist

TRAIN_NAME = "pi05_rlbench_delta_train"
LOG_ROOT = REPO_ROOT / "log"
TENSORBOARD_ROOT = REPO_ROOT / "tensorboard" / TRAIN_NAME
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / TRAIN_NAME
LATEST_CKPT_PATH = CHECKPOINT_ROOT / "latest.pth"
BEST_CKPT_PATH = CHECKPOINT_ROOT / "best.pth"
META_PATH = CHECKPOINT_ROOT / "meta.json"


"""按 --train-name 重定向所有输出目录，避免不同实验互相覆盖 checkpoint/TensorBoard。"""
def apply_train_name(train_name: str) -> None:
    global TRAIN_NAME, TENSORBOARD_ROOT, CHECKPOINT_ROOT, LATEST_CKPT_PATH, BEST_CKPT_PATH, META_PATH
    TRAIN_NAME = train_name
    TENSORBOARD_ROOT = REPO_ROOT / "tensorboard" / TRAIN_NAME
    CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / TRAIN_NAME
    LATEST_CKPT_PATH = CHECKPOINT_ROOT / "latest.pth"
    BEST_CKPT_PATH = CHECKPOINT_ROOT / "best.pth"
    META_PATH = CHECKPOINT_ROOT / "meta.json"

# 这些模块只在真正训练时导入，避免 `--help` 也要等待大段初始化。
jax = None
safetensors_torch = None
tqdm = None
openpi_pi0_config = None
openpi_pi0_pytorch = None
_normalize = None
_config = None
_data = None


"""解析训练参数，保留高频可调项，默认值仍然来自 openpi config。"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pi0.5 on RLBench LeRobot data.")
    parser.add_argument("--config-name", default="pi05_rlbench_delta_pt", help="Base openpi training config name.")
    parser.add_argument(
        "--train-name",
        default=TRAIN_NAME,
        help="Experiment name; controls checkpoints/<name> and tensorboard/<name> so runs don't clobber each other.",
    )
    parser.add_argument(
        "--freeze-vlm",
        action="store_true",
        help="Freeze the PaliGemma vision tower and language model; train only the action expert and projections.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override global batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--train-steps", type=int, default=None, help="Override total train steps.")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log interval.")
    parser.add_argument("--save-interval", type=int, default=None, help="Override checkpoint interval.")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float32"),
        default=None,
        help="Override PyTorch training precision.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--resume", action="store_true", help="Resume from a saved checkpoint index file.")
    parser.add_argument(
        "--resume-source",
        choices=("latest", "best"),
        default="latest",
        help="Choose which checkpoint index to resume from when --resume is set.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove old checkpoints and TensorBoard logs.")
    parser.add_argument("--disable-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    return parser.parse_args()


"""按需导入 openpi / jax / safetensors 等重型依赖，减少非训练场景的启动等待。"""
def load_runtime_dependencies(logger: logging.Logger | None = None) -> None:
    global jax, safetensors_torch, tqdm, openpi_pi0_config, openpi_pi0_pytorch, _normalize, _config, _data
    if _config is not None:
        return

    if logger is not None:
        logger.info("Importing openpi training stack...")

    import jax as _jax
    import safetensors.torch as _safetensors_torch
    import tqdm as _tqdm

    import openpi.models.pi0_config as _openpi_pi0_config
    import openpi.models_pytorch.pi0_pytorch as _openpi_pi0_pytorch
    import openpi.shared.normalize as _openpi_normalize
    import openpi.training.config as _openpi_config
    import openpi.training.data_loader as _openpi_data

    jax = _jax
    safetensors_torch = _safetensors_torch
    tqdm = _tqdm
    openpi_pi0_config = _openpi_pi0_config
    openpi_pi0_pytorch = _openpi_pi0_pytorch
    _normalize = _openpi_normalize
    _config = _openpi_config
    _data = _openpi_data


"""初始化 DDP 环境，并为当前进程绑定对应的 GPU 设备。"""
def setup_ddp() -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, rank, local_rank, device


"""安全销毁 DDP 进程组，避免多卡任务残留通信状态。"""
def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


"""设置随机种子；多卡时外部会传入 seed + rank。"""
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


"""初始化终端与文件日志，仅 rank0 负责真实写入。"""
def init_logging(log_path: Path, *, is_main: bool) -> logging.Logger:
    logger = logging.getLogger(TRAIN_NAME)
    logger.handlers.clear()
    logger.propagate = False

    if not is_main:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL)
        return logger

    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return logger


"""把配置对象递归转换成 JSON-safe 结构，避免 TensorBoard 记录时因特殊对象崩溃。"""
def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return make_json_safe(dataclasses.asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


"""把任意对象格式化成适合写入 TensorBoard 的文本，不让辅助记录中断训练。"""
def format_tensorboard_text(payload: Any) -> str:
    safe_payload = make_json_safe(payload)
    return f"```json\n{json.dumps(safe_payload, indent=2, ensure_ascii=True)}\n```"


"""初始化固定目录下的 TensorBoard writer，并记录本次训练配置。"""
def init_tensorboard(*, enabled: bool, is_main: bool, config: _config.TrainConfig) -> SummaryWriter | None:
    if not enabled or not is_main:
        return None

    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_ROOT))
    writer.add_text("train_config", format_tensorboard_text(dataclasses.asdict(config)), 0)
    writer.add_text(
        "runtime_paths",
        format_tensorboard_text(
            {
                "log_root": str(LOG_ROOT),
                "tensorboard_root": str(TENSORBOARD_ROOT),
                "checkpoint_root": str(CHECKPOINT_ROOT),
            }
        ),
        0,
    )
    return writer


"""读取 openpi 默认 config，并把路径与命令行覆盖项改成当前项目的版本。"""
def build_train_config(args: argparse.Namespace) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    data_config_factory = config.data
    assets_config = getattr(data_config_factory, "assets", None)
    # 把相对 assets 路径改成绝对路径，避免从不同 cwd 启动时找不到 norm stats。
    if assets_config is not None and assets_config.assets_dir and "://" not in assets_config.assets_dir:
        resolved_assets_dir = str((OPENPI_ROOT / assets_config.assets_dir).resolve())
        data_config_factory = dataclasses.replace(
            data_config_factory,
            assets=dataclasses.replace(assets_config, assets_dir=resolved_assets_dir),
        )

    replace_kwargs: dict[str, Any] = {
        "exp_name": TRAIN_NAME,
        "assets_base_dir": str(OPENPI_ROOT / "assets"),
        "checkpoint_base_dir": str(REPO_ROOT / "checkpoints"),
        "resume": args.resume,
        "overwrite": args.overwrite,
        "wandb_enabled": False,
        "data": data_config_factory,
    }
    if args.batch_size is not None:
        replace_kwargs["batch_size"] = args.batch_size
    if args.num_workers is not None:
        replace_kwargs["num_workers"] = args.num_workers
    if args.train_steps is not None:
        replace_kwargs["num_train_steps"] = args.train_steps
    if args.log_interval is not None:
        replace_kwargs["log_interval"] = args.log_interval
    if args.save_interval is not None:
        replace_kwargs["save_interval"] = args.save_interval
    if args.precision is not None:
        replace_kwargs["pytorch_training_precision"] = args.precision
    if args.seed is not None:
        replace_kwargs["seed"] = args.seed
    return dataclasses.replace(config, **replace_kwargs)


"""准备日志、TensorBoard、checkpoint 固定目录，并处理 resume / overwrite 互斥逻辑。"""
def prepare_output_dirs(
    *,
    resume: bool,
    resume_source: str,
    overwrite: bool,
    is_main: bool,
    use_ddp: bool,
) -> None:
    if is_main:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        if resume:
            resume_index_path = BEST_CKPT_PATH if resume_source == "best" else LATEST_CKPT_PATH
            if not resume_index_path.exists():
                raise FileNotFoundError(f"Resume requested but {resume_index_path} does not exist.")
            CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
            TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)
        else:
            if overwrite:
                if CHECKPOINT_ROOT.exists():
                    shutil.rmtree(CHECKPOINT_ROOT)
                if TENSORBOARD_ROOT.exists():
                    shutil.rmtree(TENSORBOARD_ROOT)
            elif CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
                raise FileExistsError(
                    f"{CHECKPOINT_ROOT} already contains checkpoints. Use --resume or --overwrite."
                )
            elif TENSORBOARD_ROOT.exists() and any(TENSORBOARD_ROOT.iterdir()):
                raise FileExistsError(
                    f"{TENSORBOARD_ROOT} already contains TensorBoard logs. Use --resume or --overwrite."
                )

            CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
            TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)

    if use_ddp:
        dist.barrier()


"""从 DDP wrapper 中取出真实模型，便于统一保存和加载权重。"""
def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


"""打印当前 GPU 显存占用，主要用于定位模型初始化或 checkpoint 加载阶段的内存问题。"""
def log_memory_usage(logger: logging.Logger, device: torch.device, step: int, *, phase: str) -> None:
    if device.type != "cuda":
        return

    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    stats = torch.cuda.memory_stats(device)
    peak_allocated = stats.get("allocated_bytes.all.peak", 0) / 1e9
    peak_reserved = stats.get("reserved_bytes.all.peak", 0) / 1e9
    logger.info(
        "step=%d phase=%s gpu_mem_allocated=%.2fGB gpu_mem_reserved=%.2fGB peak_allocated=%.2fGB peak_reserved=%.2fGB",
        step,
        phase,
        allocated,
        reserved,
        peak_allocated,
        peak_reserved,
    )


"""构建 RLBench 训练 dataloader；数据变换与归一化仍复用 openpi 官方流水线。"""
def build_dataloader(config: _config.TrainConfig) -> tuple[Any, _config.DataConfig]:
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return loader, loader.data_config()


"""按 openpi 的 pi0.5 配置构建 PyTorch 模型，并开启梯度检查点。"""
def build_model(config: _config.TrainConfig, device: torch.device) -> torch.nn.Module:
    if not isinstance(config.model, openpi_pi0_config.Pi0Config):
        model_cfg = openpi_pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi_pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


"""使用 openpi config 中的 AdamW 超参数构建优化器；只纳入可训练参数以兼容 --freeze-vlm。"""
def create_optimizer(config: _config.TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


"""冻结 PaliGemma 视觉塔与语言模型，只训练 action expert 和投影层。"""
def freeze_vlm_parameters(model: torch.nn.Module, logger: logging.Logger, *, is_main: bool) -> None:
    frozen_numel = 0
    trainable_numel = 0
    for name, param in unwrap_model(model).named_parameters():
        if name.startswith("paligemma_with_expert.paligemma."):
            param.requires_grad_(False)
            frozen_numel += param.numel()
        else:
            trainable_numel += param.numel()
    if is_main:
        logger.info(
            "Froze PaliGemma VLM: frozen=%.1fM trainable=%.1fM",
            frozen_numel / 1e6,
            trainable_numel / 1e6,
        )


"""复刻官方 PyTorch 训练脚本里的 warmup + cosine 学习率调度。"""
def lr_for_step(config: _config.TrainConfig, step: int) -> float:
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / warmup_steps

    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return end_lr + (peak_lr - end_lr) * cosine


"""新开训练时加载 pi0.5 base 权重；resume 场景下跳过。"""
def load_base_weights_if_needed(
    logger: logging.Logger,
    config: _config.TrainConfig,
    model: torch.nn.Module,
    *,
    resume: bool,
) -> None:
    if resume or config.pytorch_weight_path is None:
        return

    model_path = Path(config.pytorch_weight_path) / "model.safetensors"
    logger.info("Loading base weights from %s", model_path)
    safetensors_torch.load_model(unwrap_model(model), str(model_path))
    logger.info("Loaded base weights")


"""读取 root checkpoint 索引文件，统一处理路径不存在的报错。"""
def load_checkpoint_index(index_path: Path, map_location: torch.device | str) -> dict[str, Any]:
    if not index_path.exists():
        raise FileNotFoundError(f"Checkpoint index does not exist: {index_path}")
    return torch.load(index_path, map_location=map_location, weights_only=False)


"""从索引文件里解析出实际的 step 目录。"""
def resolve_step_dir(payload: dict[str, Any], *, index_path: Path) -> Path:
    step_dir_value = payload.get("step_dir")
    if not step_dir_value:
        raise KeyError(f"Checkpoint index {index_path} is missing `step_dir`.")
    return Path(step_dir_value).expanduser().resolve()


"""保存后只保留 latest / best 指向的完整 step 目录，避免权重文件无限堆积。"""
def prune_step_checkpoints(logger: logging.Logger) -> None:
    keep_dirs: set[Path] = set()
    for index_path in (LATEST_CKPT_PATH, BEST_CKPT_PATH):
        if not index_path.exists():
            continue
        try:
            payload = load_checkpoint_index(index_path, map_location="cpu")
            keep_dirs.add(resolve_step_dir(payload, index_path=index_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip parsing checkpoint index %s during cleanup: %s", index_path, exc)

    for step_dir in sorted(CHECKPOINT_ROOT.glob("step_*")):
        if not step_dir.is_dir():
            continue
        if step_dir.resolve() in keep_dirs:
            continue
        shutil.rmtree(step_dir)
        logger.info("Removed stale checkpoint directory: %s", step_dir)


"""从 latest/best 索引指向的 step 目录恢复模型、优化器和训练步数。"""
def load_resume_state(
    logger: logging.Logger,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    resume_source: str,
    max_train_steps: int,
) -> tuple[int, float, int]:
    index_path = BEST_CKPT_PATH if resume_source == "best" else LATEST_CKPT_PATH
    logger.info("Loading resume checkpoint from %s (%s)", resume_source, index_path)
    payload = load_checkpoint_index(index_path, map_location=device)
    step_dir = resolve_step_dir(payload, index_path=index_path)
    model_path = step_dir / "model.safetensors"
    optimizer_path = step_dir / "optimizer.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Resume checkpoint is missing model weights: {model_path}")
    if not optimizer_path.exists():
        raise FileNotFoundError(f"Resume checkpoint is missing optimizer state: {optimizer_path}")

    global_step = int(payload.get("global_step", 0))
    if global_step > max_train_steps:
        raise ValueError(
            f"Resume checkpoint step={global_step} exceeds configured num_train_steps={max_train_steps}. "
            "Increase --train-steps or choose another resume source."
        )

    safetensors_torch.load_model(unwrap_model(model), model_path, device=str(device))
    optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(optimizer_state)
    best_loss = float(payload.get("best_loss", float("inf")))
    best_step = int(payload.get("best_step", 0))
    logger.info(
        "Resumed training from %s: step=%d best_loss=%.6f best_step=%d step_dir=%s",
        resume_source,
        global_step,
        best_loss,
        best_step,
        step_dir,
    )
    return global_step, best_loss, best_step


"""通过临时文件写 JSON，避免中途中断时留下损坏的元数据文件。"""
def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


"""导出 openpi 兼容的 step 目录，供后续 policy 加载与在线测试使用。"""
def export_openpi_checkpoint(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> Path:
    final_dir = CHECKPOINT_ROOT / f"step_{global_step:07d}"
    tmp_dir = CHECKPOINT_ROOT / f".tmp_step_{global_step:07d}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safetensors_torch.save_model(unwrap_model(model), tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(
        {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        },
        tmp_dir / "metadata.pt",
    )
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    return final_dir


"""更新 latest / best 索引文件，并同步写入 step 目录与元数据摘要。"""
def save_training_state(
    logger: logging.Logger,
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    current_loss: float,
    best_loss: float,
    best_step: int,
) -> tuple[float, int]:
    step_dir = export_openpi_checkpoint(config, data_config, model, optimizer, global_step)
    is_best = current_loss <= best_loss
    if is_best:
        best_loss = current_loss
        best_step = global_step

    payload = {
        "global_step": global_step,
        "best_loss": best_loss,
        "best_step": best_step,
        "current_loss": current_loss,
        "timestamp": time.time(),
        "step_dir": str(step_dir),
        "config": dataclasses.asdict(config),
    }

    latest_tmp = LATEST_CKPT_PATH.with_suffix(".pth.tmp")
    torch.save(payload, latest_tmp)
    os.replace(latest_tmp, LATEST_CKPT_PATH)

    if is_best:
        best_tmp = BEST_CKPT_PATH.with_suffix(".pth.tmp")
        torch.save(payload, best_tmp)
        os.replace(best_tmp, BEST_CKPT_PATH)

    atomic_write_json(
        META_PATH,
        {
            "train_name": TRAIN_NAME,
            "latest_step": global_step,
            "best_step": best_step,
            "best_loss": best_loss,
            "latest_checkpoint": str(LATEST_CKPT_PATH),
            "best_checkpoint": str(BEST_CKPT_PATH) if BEST_CKPT_PATH.exists() else None,
            "latest_step_dir": str(step_dir),
        },
    )
    prune_step_checkpoints(logger)
    logger.info("Saved checkpoints at step=%d current_loss=%.6f best_loss=%.6f", global_step, current_loss, best_loss)
    return best_loss, best_step


"""训练主入口：串起目录准备、数据/模型构建、训练循环和 checkpoint 保存。"""
def train() -> int:
    args = parse_args()
    apply_train_name(args.train_name)
    use_ddp, rank, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or rank == 0

    try:
        # 先准备固定输出目录，再初始化日志，确保后续所有状态都能落盘。
        prepare_output_dirs(
            resume=args.resume,
            resume_source=args.resume_source,
            overwrite=args.overwrite,
            is_main=is_main,
            use_ddp=use_ddp,
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOG_ROOT / f"{TRAIN_NAME}_{timestamp}.log"
        logger = init_logging(log_path, is_main=is_main)

        # 训练相关的重型依赖放到这里再导入，便于 `--help` 快速返回。
        load_runtime_dependencies(logger if is_main else None)
        config = build_train_config(args)
        set_random_seed(config.seed + rank)

        writer = init_tensorboard(enabled=not args.disable_tensorboard, is_main=is_main, config=config)
        if is_main:
            logger.info("Building RLBench train dataloader...")
        loader, data_config = build_dataloader(config)
        if is_main:
            logger.info("RLBench train dataloader is ready.")

        if is_main:
            logger.info("Building pi0.5 model...")
        model = build_model(config, device)
        if is_main:
            logger.info("pi0.5 model is ready.")
        if args.freeze_vlm:
            freeze_vlm_parameters(model, logger, is_main=is_main)
        optimizer = create_optimizer(config, model)

        # 多卡训练仍走标准 DDP，保持和官方 PyTorch 入口一致。
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                static_graph=int(os.environ.get("WORLD_SIZE", "1")) >= 8,
            )

        if is_main:
            logger.info("Log file: %s", log_path)
            logger.info("TensorBoard: %s", TENSORBOARD_ROOT if writer is not None else "disabled")
            logger.info("Checkpoints: %s", CHECKPOINT_ROOT)
            logger.info(
                "Using config=%s batch_size=%d num_workers=%d train_steps=%d",
                config.name,
                config.batch_size,
                config.num_workers,
                config.num_train_steps,
            )
            logger.info(
                "Running on %s | rank=%d local_rank=%d world_size=%d",
                platform.node(),
                rank,
                local_rank,
                dist.get_world_size() if use_ddp else 1,
            )
            if args.resume:
                logger.info("Resume source: %s", args.resume_source)
            logger.info("Loaded data config for repo_id=%s", data_config.repo_id)
            log_memory_usage(logger, device, 0, phase="after_model_creation")

        global_step = 0
        best_loss = float("inf")
        best_step = 0
        if args.resume:
            global_step, best_loss, best_step = load_resume_state(
                logger,
                model,
                optimizer,
                device,
                resume_source=args.resume_source,
                max_train_steps=config.num_train_steps,
            )
        else:
            if is_main:
                logger.info("Loading base pi0.5 weights...")
            load_base_weights_if_needed(logger, config, model, resume=False)

        model.train()
        if is_main:
            logger.info("Entering train loop.")
        step_start_time = time.time()
        log_buffer: list[dict[str, float]] = []
        pbar = (
            tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
            if is_main
            else None
        )

        while global_step < config.num_train_steps:
            # 分布式 dataloader 在每轮循环前同步 epoch，保证 shuffle 行为稳定。
            if use_ddp and hasattr(loader, "set_epoch"):
                loader.set_epoch(global_step // len(loader))

            for observation, actions in loader:
                if global_step >= config.num_train_steps:
                    break

                # observation / actions 统一搬到当前 rank 对应的 device。
                observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
                actions = actions.to(torch.float32).to(device)  # noqa: PLW2901

                # 每步显式刷新学习率，和官方脚本保持同样的调度方式。
                lr = lr_for_step(config, global_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # 前向、反向、梯度裁剪与优化器更新。
                losses = model(observation, actions)
                if isinstance(losses, (list, tuple)):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)

                loss = losses.mean()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if is_main:
                    log_buffer.append(
                        {
                            "loss": float(loss.item()),
                            "lr": float(lr),
                            "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                        }
                    )

                global_step += 1

                # 按固定间隔聚合日志，并同步写入 TensorBoard。
                if is_main and global_step % config.log_interval == 0:
                    elapsed = time.time() - step_start_time
                    avg_loss = sum(item["loss"] for item in log_buffer) / len(log_buffer)
                    avg_lr = sum(item["lr"] for item in log_buffer) / len(log_buffer)
                    avg_grad = sum(item["grad_norm"] for item in log_buffer) / len(log_buffer)
                    time_per_step = elapsed / len(log_buffer)
                    logger.info(
                        "step=%d loss=%.4f lr=%.2e grad_norm=%.2f time_per_step=%.3fs",
                        global_step,
                        avg_loss,
                        avg_lr,
                        avg_grad,
                        time_per_step,
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("train/lr", avg_lr, global_step)
                        writer.add_scalar("train/grad_norm", avg_grad, global_step)
                        writer.add_scalar("train/time_per_step", time_per_step, global_step)
                    step_start_time = time.time()
                    log_buffer = []

                # latest / best 索引文件与 openpi step 导出目录一起更新。
                should_save = (
                    (global_step % config.save_interval == 0 and global_step > 0)
                    or global_step == config.num_train_steps
                )
                if is_main and should_save:
                    best_loss, best_step = save_training_state(
                        logger,
                        config,
                        data_config,
                        model,
                        optimizer,
                        global_step=global_step,
                        current_loss=float(loss.item()),
                        best_loss=best_loss,
                        best_step=best_step,
                    )

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss": f"{float(loss.item()):.4f}",
                            "lr": f"{lr:.2e}",
                            "step": global_step,
                        }
                    )

        if pbar is not None:
            pbar.close()

        if writer is not None:
            writer.close()

        if is_main:
            logger.info("Training finished at step=%d best_loss=%.6f best_step=%d", global_step, best_loss, best_step)
        return 0
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    raise SystemExit(train())
