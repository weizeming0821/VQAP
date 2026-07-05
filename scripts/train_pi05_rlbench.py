#!/usr/bin/env python3
"""PyTorch RLBench fine-tuning entrypoint for pi0.5."""

from __future__ import annotations

import argparse
import dataclasses
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

# Set allocator config before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "openpi-vqap"
OPENPI_SRC = OPENPI_ROOT / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / "openpi_cache"))
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO_ROOT / "LeRobot_RLBench_Dataset"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache" / "huggingface" / "datasets"))

import numpy as np
import torch
import torch.distributed as dist

TRAIN_NAME = "pi05_rlbench_train"
LOG_ROOT = REPO_ROOT / "log"
TENSORBOARD_ROOT = REPO_ROOT / "tensorboard" / TRAIN_NAME
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / TRAIN_NAME
LATEST_CKPT_PATH = CHECKPOINT_ROOT / "latest.pth"
BEST_CKPT_PATH = CHECKPOINT_ROOT / "best.pth"
META_PATH = CHECKPOINT_ROOT / "meta.json"

jax = None
safetensors_torch = None
tqdm = None
openpi_pi0_config = None
openpi_pi0_pytorch = None
_normalize = None
_config = None
_data = None


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune pi0.5 on RLBench LeRobot data.")
    parser.add_argument("--config-name", default="pi05_rlbench_pt", help="Base openpi training config name.")
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
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/pi05_rlbench_train/latest.pth.")
    parser.add_argument("--overwrite", action="store_true", help="Remove old checkpoints and TensorBoard logs.")
    parser.add_argument("--disable-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    return parser.parse_args()


def load_runtime_dependencies(logger: logging.Logger | None = None) -> None:
    """Import heavy training modules only when a real run starts."""
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


def setup_ddp() -> tuple[bool, int, int, torch.device]:
    """Initialize DDP when launched with torchrun."""
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


def cleanup_ddp() -> None:
    """Tear down the DDP process group."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_random_seed(seed: int) -> None:
    """Set per-process random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logging(log_path: Path, *, is_main: bool) -> logging.Logger:
    """Create console + file logging on rank 0 only."""
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


def init_tensorboard(*, enabled: bool, is_main: bool, config: _config.TrainConfig) -> SummaryWriter | None:
    """Create a SummaryWriter in a fixed TensorBoard directory."""
    if not enabled or not is_main:
        return None

    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_ROOT))
    writer.add_text("train_config", f"```json\n{json.dumps(dataclasses.asdict(config), indent=2)}\n```", 0)
    writer.add_text(
        "runtime_paths",
        f"```json\n{json.dumps({'log_root': str(LOG_ROOT), 'tensorboard_root': str(TENSORBOARD_ROOT), 'checkpoint_root': str(CHECKPOINT_ROOT)}, indent=2)}\n```",
        0,
    )
    return writer


def build_train_config(args: argparse.Namespace) -> _config.TrainConfig:
    """Load the base openpi config and apply CLI overrides."""
    config = _config.get_config(args.config_name)
    data_config_factory = config.data
    assets_config = getattr(data_config_factory, "assets", None)
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


def prepare_output_dirs(*, resume: bool, overwrite: bool, is_main: bool, use_ddp: bool) -> None:
    """Prepare fixed output directories for the single training run."""
    if is_main:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        if resume:
            if not LATEST_CKPT_PATH.exists():
                raise FileNotFoundError(f"Resume requested but {LATEST_CKPT_PATH} does not exist.")
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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when wrapped by DDP."""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def log_memory_usage(logger: logging.Logger, device: torch.device, step: int, *, phase: str) -> None:
    """Log CUDA memory statistics when available."""
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


def build_dataloader(config: _config.TrainConfig) -> tuple[Any, _config.DataConfig]:
    """Build the RLBench training dataloader."""
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return loader, loader.data_config()


def build_model(config: _config.TrainConfig, device: torch.device) -> torch.nn.Module:
    """Create the PyTorch pi0.5 model on the target device."""
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


def create_optimizer(config: _config.TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build the AdamW optimizer from the openpi config."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


def lr_for_step(config: _config.TrainConfig, step: int) -> float:
    """Match the cosine schedule used by the official PyTorch script."""
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


def load_base_weights_if_needed(
    logger: logging.Logger,
    config: _config.TrainConfig,
    model: torch.nn.Module,
    *,
    resume: bool,
) -> None:
    """Load the base PyTorch checkpoint for fresh fine-tuning runs."""
    if resume or config.pytorch_weight_path is None:
        return

    model_path = Path(config.pytorch_weight_path) / "model.safetensors"
    logger.info("Loading base weights from %s", model_path)
    safetensors_torch.load_model(unwrap_model(model), str(model_path))
    logger.info("Loaded base weights")


def load_resume_state(
    logger: logging.Logger,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, int]:
    """Restore model and optimizer state from latest.pth."""
    logger.info("Loading resume checkpoint from %s", LATEST_CKPT_PATH)
    payload = torch.load(LATEST_CKPT_PATH, map_location=device, weights_only=False)
    step_dir = Path(payload["step_dir"]).expanduser().resolve()
    safetensors_torch.load_model(unwrap_model(model), step_dir / "model.safetensors", device=str(device))
    optimizer_state = torch.load(step_dir / "optimizer.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(optimizer_state)
    global_step = int(payload.get("global_step", 0))
    best_loss = float(payload.get("best_loss", float("inf")))
    best_step = int(payload.get("best_step", 0))
    logger.info(
        "Resumed training from step=%d best_loss=%.6f best_step=%d step_dir=%s",
        global_step,
        best_loss,
        best_step,
        step_dir,
    )
    return global_step, best_loss, best_step


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def export_openpi_checkpoint(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> Path:
    """Export an openpi-style checkpoint directory for future policy loading."""
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
    """Save latest/best .pth files and an openpi export directory."""
    step_dir = export_openpi_checkpoint(config, data_config, model, optimizer, global_step)
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

    if current_loss <= best_loss:
        best_loss = current_loss
        best_step = global_step
        payload["best_loss"] = best_loss
        payload["best_step"] = best_step
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
    logger.info("Saved checkpoints at step=%d current_loss=%.6f best_loss=%.6f", global_step, current_loss, best_loss)
    return best_loss, best_step


def train() -> int:
    """Run training."""
    args = parse_args()
    use_ddp, rank, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or rank == 0

    try:
        prepare_output_dirs(resume=args.resume, overwrite=args.overwrite, is_main=is_main, use_ddp=use_ddp)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOG_ROOT / f"{TRAIN_NAME}_{timestamp}.log"
        logger = init_logging(log_path, is_main=is_main)

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
        optimizer = create_optimizer(config, model)

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
            logger.info("Using config=%s batch_size=%d num_workers=%d train_steps=%d", config.name, config.batch_size, config.num_workers, config.num_train_steps)
            logger.info("Running on %s | rank=%d local_rank=%d world_size=%d", platform.node(), rank, local_rank, dist.get_world_size() if use_ddp else 1)
            logger.info("Loaded data config for repo_id=%s", data_config.repo_id)
            log_memory_usage(logger, device, 0, phase="after_model_creation")

        global_step = 0
        best_loss = float("inf")
        best_step = 0
        if args.resume:
            global_step, best_loss, best_step = load_resume_state(logger, model, optimizer, device)
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
            if use_ddp and hasattr(loader, "set_epoch"):
                loader.set_epoch(global_step // len(loader))

            for observation, actions in loader:
                if global_step >= config.num_train_steps:
                    break

                observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
                actions = actions.to(torch.float32).to(device)  # noqa: PLW2901

                lr = lr_for_step(config, global_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

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
