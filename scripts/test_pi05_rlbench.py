#!/usr/bin/env python3
"""Standalone RLBench evaluation entrypoint for fine-tuned/base pi0.5."""

from __future__ import annotations

import argparse
import dataclasses
from collections import deque
import json
import logging
import os
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Any

import numpy as np
import yaml

# 训练/评测都默认走项目内缓存，避免把模型和数据下载到用户目录。
REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "openpi-vqap"
OPENPI_SRC = OPENPI_ROOT / "src"
RLBENCH_SRC = REPO_ROOT / "source" / "RLBench"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))
if str(RLBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(RLBENCH_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.rlbench_video_recorder import EpisodeVideoRecorder

os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / "openpi_cache"))
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO_ROOT / "LeRobot_RLBench_Dataset"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".cache" / "huggingface" / "datasets"))

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "eval_pi05_rlbench.yaml"
LOW_DIM_PICKLE = "low_dim_obs.pkl"
IMAGE_FORMAT = "%d.png"
RGB_CAMERA_SPECS = (
    ("front_rgb", "front_rgb"),
    ("wrist_rgb", "wrist_rgb"),
    ("left_shoulder_rgb", "left_shoulder_rgb"),
    ("right_shoulder_rgb", "right_shoulder_rgb"),
    ("overhead_rgb", "overhead_rgb"),
)


class EvalEnvironmentBroken(RuntimeError):
    """连续 rollout 异常超限：判定 RLBench/CoppeliaSim 环境已损坏、无法恢复。

    抛出后由主流程捕获，跳过剩余任务，但仍写出已收集结果的 summary（避免整轮白跑）。
    """

    def __init__(self, consecutive: int):
        super().__init__(f"{consecutive} consecutive rollout failures; environment considered broken.")
        self.consecutive = consecutive


"""解析评测参数；核心配置放 YAML，命令行只保留少量高频覆盖项。"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pi0.5 on RLBench.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--which",
        choices=("best", "latest", "best_eval", "official"),
        default=None,
        help="Override checkpoint.which in YAML. best_eval = 在线验证成功率最优的 ckpt 索引。",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Evaluate this openpi step directory directly, bypassing best/latest index files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write a machine-readable JSON summary (overall + per-task) to this path.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma separated task list override, e.g. open_drawer,turn_tap.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=None,
        help="Override dataset.episodes_per_task for quick smoke runs.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help="Override dataset.start_episode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override environment.max_steps.",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Override environment.replan_steps.",
    )
    parser.add_argument(
        "--arm-action-mode",
        choices=("planning", "ik", "hybrid"),
        default=None,
        help="Override environment.arm_action_mode: 'planning' (per-step path planning), "
        "'ik' (smooth IK servoing) or 'hybrid' (IK first, fall back to planning on IK failure).",
    )
    parser.add_argument(
        "--action-stride",
        type=int,
        default=None,
        help="Execute every k-th action of the predicted chunk (targets are absolute poses, "
        "so skipping is valid). Increases per-step motion to escape stall attractors.",
    )
    parser.add_argument(
        "--action-space",
        choices=("absolute", "delta"),
        default=None,
        help="Override checkpoint.action_space: how to interpret policy outputs. "
        "'delta' composes [dxyz, rot6d(dR), grip] with the current gripper pose.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override runtime.device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--record-videos",
        type=int,
        default=None,
        help="Override recording.videos_per_task (success/fail each); 0 disables recording.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, checkpoints and task plans without launching RLBench.",
    )
    parser.add_argument(
        "--replay-demo",
        action="store_true",
        help="Upper-bound test: replay demo next-frame poses through the same "
        "8D->10D->8D conversion and step pipeline, without loading any policy.",
    )
    parser.add_argument(
        "--replay-delta",
        action="store_true",
        help="Diagnostic: replay the GROUND-TRUTH delta actions (stride-3, computed "
        "with the exact training-data function state_pair_to_delta_action) through the "
        "policy's delta synthesis path (delta_action10_to_rlbench_action8). Isolates "
        "whether the delta eval path can execute tasks when fed correct actions "
        "(vs --replay-demo which validates the absolute path). No policy loaded.",
    )
    parser.add_argument(
        "--replay-keyframes",
        action="store_true",
        help="Waypoint paradigm ceiling test: replay ONLY the heuristic keyframe poses "
        "(TAVP keypoint_discovery on the loaded demo) through the planner, absolute pose "
        "per keyframe. Measures whether ground-truth waypoints + planner reach task "
        "success (upper bound for the waypoint model). No policy loaded.",
    )
    parser.add_argument(
        "--keyframe-max-seg",
        type=int,
        default=0,
        help="For --replay-keyframes: if >0, densify keyframes by subdividing any segment "
        "longer than this many frames (same data-side densification as the training "
        "manifest). 0 = standard heuristic keyframes only.",
    )
    parser.add_argument(
        "--waypoint-eval",
        action="store_true",
        help="Waypoint model eval: closed-loop rollout where the policy predicts the next "
        "absolute waypoint pose (rotvec7) each step, executed via the planner (3-view + "
        "ee_rotvec state + first-description prompt). For the pi05_rlbench_waypoint config.",
    )
    parser.add_argument(
        "--waypoint-max-waypoints",
        type=int,
        default=30,
        help="Max predicted waypoints per episode for --waypoint-eval (default 30).",
    )
    parser.add_argument(
        "--predict-diag",
        action="store_true",
        help="Diagnostic: teacher-forced prediction accuracy. Keeps the arm ON the demo "
        "trajectory (absolute replay) and at each stride-3 frame compares the policy's "
        "predicted delta to the ground-truth delta, split into xyz/rot6d/grip and tagged "
        "by gripper-transition (grasp/release) frames. Measures pure prediction accuracy "
        "with no rollout drift. Loads the policy; writes error stats to --summary-json.",
    )
    parser.add_argument(
        "--predict-diag-episodes",
        type=int,
        default=3,
        help="Episodes per task for --predict-diag (default 3).",
    )
    # ---- Exp0（原子动作动机实验，Exp_Design.md §三）----
    parser.add_argument(
        "--atom-mode",
        action="store_true",
        help="Exp0: replay the demo prefix up to each atomic segment's start frame, "
        "then hand control to the policy for that single atomic action. Records one "
        "video per (segment, prompt-condition) for human judging.",
    )
    parser.add_argument(
        "--atom-cache",
        type=Path,
        default=None,
        help="Exp0: planner-cache JSON holding the val-split subtask segmentation "
        "(produced by tools/build_subtask_cache.py against RLBench_Raw_Dataset/val).",
    )
    parser.add_argument(
        "--atom-prompt",
        choices=("atomic", "full", "both"),
        default="both",
        help="Exp0 instruction condition: 'atomic' = subtask instruction (condition b), "
        "'full' = whole-task instruction (condition a), 'both' = run each segment twice. "
        "Default both, which is what Exp_Design.md §3.4 requires.",
    )
    parser.add_argument(
        "--atom-episodes-per-task",
        type=int,
        default=3,
        help="Exp0: how many demo episodes per task to decompose (default 3).",
    )
    parser.add_argument(
        "--atom-max-steps-factor",
        type=float,
        default=2.0,
        help="Exp0: per-segment step budget = ceil(segment_raw_len / 3) * factor "
        "(3 = the action_stride used when converting training data).",
    )
    parser.add_argument(
        "--atom-min-max-steps",
        type=int,
        default=20,
        help="Exp0: floor for the per-segment step budget (default 20).",
    )
    parser.add_argument(
        "--atom-skip-actions",
        default="pose-adjust",
        help="Exp0: comma-separated action labels to skip (default 'pose-adjust', "
        "which VQAP_Design marks as a discarded/unreliable action).",
    )
    parser.add_argument(
        "--task-set",
        choices=("all", "seen12", "unseen6"),
        default=None,
        help="Override config.tasks.names with the Exp_Design.md §1.1 split. "
        "seen12 = M0/M1 training tasks (Exp1), unseen6 = held-out tasks (Exp2). "
        "The task lists are derived from openpi's RLBENCH_SEEN12_EPISODE_RANGES "
        "so they cannot drift from the training-side split.",
    )
    return parser.parse_args()


"""按 Exp_Design.md §1.1 解析 Seen12 / UnSeen6 任务集合。

Seen12 直接取自 openpi 训练侧的 RLBENCH_SEEN12_EPISODE_RANGES，保证评测划分与
训练划分是同一个事实源；UnSeen6 = 配置里的全集减去 Seen12。
"""
def resolve_task_set(task_set: str, all_task_names: list[str]) -> list[str]:
    if task_set == "all":
        return list(all_task_names)
    from openpi.training.config import RLBENCH_SEEN12_EPISODE_RANGES

    seen12 = set(RLBENCH_SEEN12_EPISODE_RANGES)
    missing = seen12 - set(all_task_names)
    if missing:
        raise ValueError(
            f"config.tasks.names is missing Seen12 task(s): {sorted(missing)}; "
            "the eval config must list all 18 RLBench tasks."
        )
    if task_set == "seen12":
        selected = [name for name in all_task_names if name in seen12]
    else:
        selected = [name for name in all_task_names if name not in seen12]
    expected = 12 if task_set == "seen12" else 6
    if len(selected) != expected:
        raise ValueError(
            f"--task-set {task_set} resolved to {len(selected)} tasks, expected {expected}. "
            f"Check config.tasks.names against Exp_Design.md §1.1."
        )
    return selected


"""读取 YAML 配置。"""
def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return payload


"""把命令行高频覆盖项写回配置字典，方便后续统一使用。"""
def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    if args.which is not None:
        config["checkpoint"]["which"] = args.which
    if args.tasks is not None:
        config["tasks"]["names"] = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if args.episodes_per_task is not None:
        config["dataset"]["episodes_per_task"] = args.episodes_per_task
    if args.start_episode is not None:
        config["dataset"]["start_episode"] = args.start_episode
    if args.max_steps is not None:
        config["environment"]["max_steps"] = args.max_steps
    if args.replan_steps is not None:
        config["environment"]["replan_steps"] = args.replan_steps
    if args.arm_action_mode is not None:
        config["environment"]["arm_action_mode"] = args.arm_action_mode
    if args.action_stride is not None:
        config["environment"]["action_stride"] = args.action_stride
    if args.action_space is not None:
        config["checkpoint"]["action_space"] = args.action_space
    if args.checkpoint_dir is not None:
        config["checkpoint"]["dir_override"] = str(args.checkpoint_dir)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.record_videos is not None:
        recording = config.setdefault("recording", {})
        if args.record_videos <= 0:
            recording["enabled"] = False
        else:
            recording["enabled"] = True
            recording["videos_per_task"] = args.record_videos
    return config


"""把 repo 内相对路径转换成绝对路径；远端 URI 保持原样。"""
def resolve_path_like(value: str | Path | None) -> str | Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.expanduser().resolve()
    if "://" in value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


"""为本次评测创建终端+文件日志。"""
def init_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("pi05_rlbench_eval")
    logger.handlers.clear()
    logger.propagate = False

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
    return logger


"""固定随机种子，保证评测进程侧的随机路径可复现。"""
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


"""延迟导入 torch，避免仅查看 --help 时就依赖完整训练环境。"""
def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("PyTorch is required to evaluate pi0.5 on RLBench.") from exc
    return torch


"""按运行环境自动决定 policy 放到 cpu 还是 cuda。"""
def resolve_runtime_device(requested: str) -> str:
    if requested != "auto":
        return requested
    torch = require_torch()
    return "cuda:0" if torch.cuda.is_available() else "cpu"


"""判断某个 Qt 变量是否被 opencv-python 的内置插件目录污染了。"""
def is_cv2_qt_path(path: str | None) -> bool:
    if not path:
        return False
    normalized = path.replace("\\", "/")
    return "/cv2/qt/" in normalized and "site-packages" in normalized


"""在启动 RLBench 前修正 Qt/X11 环境，避免 openpi 导入链把插件路径指到 cv2。"""
def sanitize_rlbench_gui_env(logger: logging.Logger) -> None:
    coppeliaroot = os.environ.get("COPPELIASIM_ROOT")
    display = os.environ.get("DISPLAY")
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    plugin_search_path = os.environ.get("QT_PLUGIN_PATH")
    font_dir = os.environ.get("QT_QPA_FONTDIR")

    logger.info(
        "GUI env before RLBench launch | DISPLAY=%s COPPELIASIM_ROOT=%s QT_QPA_PLATFORM_PLUGIN_PATH=%s QT_PLUGIN_PATH=%s QT_QPA_FONTDIR=%s",
        display,
        coppeliaroot,
        plugin_path,
        plugin_search_path,
        font_dir,
    )
    if not display:
        logger.warning(
            "DISPLAY is not set. Even with headless=True, RLBench/PyRep usually still needs an X server. "
            "Use `export DISPLAY=:99` or `DISPLAY=:99 python3 scripts/eval_pi05_rlbench.py ...`."
        )

    if is_cv2_qt_path(plugin_path):
        if coppeliaroot:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = coppeliaroot
            logger.info(
                "Replaced cv2 Qt plugin path with COPPELIASIM_ROOT: %s",
                coppeliaroot,
            )
        else:
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
            logger.warning(
                "Removed cv2 Qt plugin path, but COPPELIASIM_ROOT is unset. "
                "If RLBench still fails, export COPPELIASIM_ROOT before running eval."
            )
    elif not plugin_path and coppeliaroot:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = coppeliaroot
        logger.info("Set QT_QPA_PLATFORM_PLUGIN_PATH from COPPELIASIM_ROOT: %s", coppeliaroot)

    if is_cv2_qt_path(plugin_search_path):
        os.environ.pop("QT_PLUGIN_PATH", None)
        logger.info("Removed cv2-injected QT_PLUGIN_PATH: %s", plugin_search_path)

    if is_cv2_qt_path(font_dir):
        os.environ.pop("QT_QPA_FONTDIR", None)
        logger.info("Removed cv2-injected QT_QPA_FONTDIR: %s", font_dir)

    logger.info(
        "GUI env sanitized | DISPLAY=%s QT_QPA_PLATFORM_PLUGIN_PATH=%s QT_PLUGIN_PATH=%s QT_QPA_FONTDIR=%s",
        os.environ.get("DISPLAY"),
        os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"),
        os.environ.get("QT_PLUGIN_PATH"),
        os.environ.get("QT_QPA_FONTDIR"),
    )


"""为旧 numpy pickle 安装模块别名，兼容 raw demo 在不同 numpy 版本间反序列化。"""
def install_numpy_pickle_compat(logger: logging.Logger) -> None:
    import importlib

    alias_pairs = {
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
    }
    installed: list[str] = []

    for legacy_name, modern_name in alias_pairs.items():
        try:
            importlib.import_module(legacy_name)
            continue
        except ModuleNotFoundError:
            pass

        module = importlib.import_module(modern_name)
        sys.modules[legacy_name] = module

        parent_name, _, attr_name = legacy_name.rpartition(".")
        if parent_name:
            parent_module = importlib.import_module(parent_name)
            setattr(parent_module, attr_name, module)
        installed.append(f"{legacy_name}->{modern_name}")

    if installed:
        logger.info("Installed numpy pickle compatibility aliases: %s", ", ".join(installed))
    else:
        logger.info("Numpy pickle compatibility aliases not needed in current environment.")


"""构建 openpi 训练配置，并把相对 assets 路径修正到当前仓库。"""
def build_policy_train_config(config_name: str):
    import openpi.training.config as openpi_config

    train_config = openpi_config.get_config(config_name)
    data_config_factory = train_config.data
    assets_config = getattr(data_config_factory, "assets", None)
    if assets_config is not None and assets_config.assets_dir and "://" not in assets_config.assets_dir:
        resolved_assets_dir = str((OPENPI_ROOT / assets_config.assets_dir).resolve())
        data_config_factory = dataclasses.replace(
            data_config_factory,
            assets=dataclasses.replace(assets_config, assets_dir=resolved_assets_dir),
        )

    return dataclasses.replace(
        train_config,
        exp_name="pi05_rlbench_eval",
        assets_base_dir=str(OPENPI_ROOT / "assets"),
        checkpoint_base_dir=str(REPO_ROOT / "checkpoints"),
        wandb_enabled=False,
        data=data_config_factory,
    )


"""读取 latest/best 根索引文件。"""
def load_checkpoint_index(index_path: Path) -> dict[str, Any]:
    torch = require_torch()
    if not index_path.exists():
        raise FileNotFoundError(f"Checkpoint index does not exist: {index_path}")
    return torch.load(index_path, map_location="cpu", weights_only=False)


"""从索引文件里解析真正可用于 policy 加载的 step 目录。"""
def resolve_step_dir(index_payload: dict[str, Any], *, index_path: Path) -> Path:
    step_dir = index_payload.get("step_dir")
    if not step_dir:
        raise KeyError(f"Checkpoint index {index_path} is missing `step_dir`.")
    return Path(step_dir).expanduser().resolve()


"""列出 step 目录内自带的 norm stats 候选(assets/<asset_id> 子目录,如 train / train_delta)。"""
def step_dir_norm_candidates(step_dir: Path) -> list[Path]:
    assets_dir = step_dir / "assets"
    if not assets_dir.is_dir():
        return []
    return sorted(path for path in assets_dir.iterdir() if path.is_dir())


"""统一解析 official / best / latest / best_eval / 直接目录 五种权重来源。"""
def resolve_checkpoint_source(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_cfg = config["checkpoint"]

    dir_override = checkpoint_cfg.get("dir_override")
    if dir_override:
        step_dir = Path(dir_override).expanduser().resolve()
        if not step_dir.is_dir():
            raise FileNotFoundError(f"checkpoint dir override does not exist: {step_dir}")
        return {
            "source": f"dir:{step_dir.name}",
            "index_path": None,
            "checkpoint_dir": step_dir,
            "norm_stats_candidates": [
                *step_dir_norm_candidates(step_dir),
                resolve_path_like(checkpoint_cfg.get("rlbench_norm_stats_dir")),
            ],
            "payload": None,
        }

    source = checkpoint_cfg["which"]
    if source in ("best", "latest", "best_eval"):
        ckpt_root = resolve_path_like(checkpoint_cfg["finetuned_root_dir"])
        assert isinstance(ckpt_root, Path)
        index_path = ckpt_root / f"{source}.pth"
        payload = load_checkpoint_index(index_path)
        step_dir = resolve_step_dir(payload, index_path=index_path)
        return {
            "source": source,
            "index_path": index_path,
            "checkpoint_dir": step_dir,
            "norm_stats_candidates": [
                *step_dir_norm_candidates(step_dir),
                resolve_path_like(checkpoint_cfg.get("rlbench_norm_stats_dir")),
            ],
            "payload": payload,
        }

    if source == "official":
        checkpoint_dir = resolve_path_like(checkpoint_cfg["official_dir"])
        return {
            "source": source,
            "index_path": None,
            "checkpoint_dir": checkpoint_dir,
            "norm_stats_candidates": [resolve_path_like(checkpoint_cfg.get("rlbench_norm_stats_dir"))],
            "payload": None,
        }

    raise ValueError(f"Unsupported checkpoint.which: {source}")


"""在若干候选目录里找第一个可用的 norm stats 源。"""
def pick_norm_stats_source(candidates: list[str | Path | None]) -> str | Path:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Path):
            if (candidate / "norm_stats.json").exists():
                return candidate
            continue
        if "://" in candidate:
            return candidate
        path = Path(candidate).expanduser().resolve()
        if (path / "norm_stats.json").exists():
            return path
    pretty = [str(candidate) for candidate in candidates if candidate is not None]
    raise FileNotFoundError(f"Could not find norm_stats.json from candidates: {pretty}")


"""加载 RLBench 评测所需的 norm stats。"""
def load_norm_stats(norm_stats_dir: str | Path):
    import openpi.shared.download as openpi_download
    import openpi.shared.normalize as openpi_normalize

    resolved = openpi_download.maybe_download(str(norm_stats_dir))
    return openpi_normalize.load(resolved)


"""用 openpi 的 policy loader 创建可直接调用的 pi0.5 policy。"""
def create_policy(train_config, checkpoint_dir: str | Path, norm_stats, device: str):
    from openpi.policies import policy_config as openpi_policy_config

    return openpi_policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        pytorch_device=device,
    )


"""把 RLBench 的图像字段整理成 uint8 HWC，便于直接送入 policy。"""
def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape={image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


"""把当前 live observation 整理成 openpi RLBench policy 需要的输入格式。"""
def make_policy_observation(obs: Any, prompt: str) -> dict[str, Any]:
    state = np.concatenate(
        [
            np.asarray(obs.gripper_pose, dtype=np.float32),
            np.asarray([float(obs.gripper_open)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "observation/front": ensure_uint8_image(obs.front_rgb),
        "observation/wrist": ensure_uint8_image(obs.wrist_rgb),
        "observation/state": state,
        "prompt": prompt,
    }


"""把当前 live observation 整理成 waypoint policy(RlbenchWaypointInputs) 需要的输入。
3 视角 front/left_shoulder/right_shoulder + ee_rotvec state(7)，与训练数据构造完全一致。"""
def make_waypoint_policy_observation(obs: Any, prompt: str) -> dict[str, Any]:
    import sys as _sys

    _wp_dir = str(Path(__file__).resolve().parent.parent / "tools" / "waypoint")
    if _wp_dir not in _sys.path:
        _sys.path.insert(0, _wp_dir)
    from rlbench_pi05_waypoint.common import obs_to_state  # 与训练同一函数,保证 state 一致

    state = np.asarray(obs_to_state(obs, mode="ee_rotvec"), dtype=np.float32)  # [xyz, rotvec, grip]
    return {
        "observation/front_image": ensure_uint8_image(obs.front_rgb),
        "observation/left_shoulder_image": ensure_uint8_image(obs.left_shoulder_rgb),
        "observation/right_shoulder_image": ensure_uint8_image(obs.right_shoulder_rgb),
        "observation/state": state,
        "prompt": prompt,
    }


"""把 6D 旋转表示恢复成 3x3 旋转矩阵。"""
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected rot6d last dim 6, got shape={rot6d.shape}")
    basis = rot6d.reshape(3, 2)
    col1 = basis[:, 0]
    col2 = basis[:, 1]
    col1 = col1 / np.clip(np.linalg.norm(col1), 1e-8, None)
    col2 = col2 - np.dot(col1, col2) * col1
    col2 = col2 / np.clip(np.linalg.norm(col2), 1e-8, None)
    col3 = np.cross(col1, col2)
    return np.stack([col1, col2, col3], axis=-1).astype(np.float32)


"""把旋转矩阵转回 RLBench 需要的 xyzw 四元数。"""
def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float32)
    quat /= np.clip(np.linalg.norm(quat), 1e-8, None)
    return quat


"""把 policy 的 10D 动作恢复成 RLBench 8D 末端位姿动作。"""
def policy_action10_to_rlbench_action8(action10: np.ndarray, gripper_threshold: float) -> np.ndarray:
    action10 = np.asarray(action10, dtype=np.float32)
    if action10.shape[-1] < 10:
        raise ValueError(f"Expected policy action dim >= 10, got shape={action10.shape}")
    xyz = action10[:3]
    rot6d = action10[3:9]
    grip = float(action10[9] > gripper_threshold)
    quat_xyzw = matrix_to_quaternion_xyzw(rot6d_to_matrix(rot6d))
    return np.concatenate([xyz, quat_xyzw, np.asarray([grip], dtype=np.float32)], axis=0).astype(np.float32)


"""把 xyzw 四元数转成 3x3 旋转矩阵。"""
def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32)
    q = q / np.clip(np.linalg.norm(q), 1e-8, None)
    x, y, z, w = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


"""delta 动作空间：policy 输出 [dxyz, rot6d(dR_world), grip]，与当前实际位姿合成绝对目标。

数据转换端定义 R_next = R_delta @ R_now（世界系），这里对应 R_target = dR @ R_current。
"""
def delta_action10_to_rlbench_action8(
    action10: np.ndarray,
    current_pose7: np.ndarray,
    gripper_threshold: float,
) -> np.ndarray:
    action10 = np.asarray(action10, dtype=np.float32)
    current_pose7 = np.asarray(current_pose7, dtype=np.float32)
    if action10.shape[-1] < 10:
        raise ValueError(f"Expected policy action dim >= 10, got shape={action10.shape}")
    if current_pose7.shape != (7,):
        raise ValueError(f"Expected current pose shape (7,), got {current_pose7.shape}")

    xyz = current_pose7[:3] + action10[:3]
    delta_matrix = rot6d_to_matrix(action10[3:9])
    target_matrix = delta_matrix @ quat_xyzw_to_matrix(current_pose7[3:7])
    quat_xyzw = matrix_to_quaternion_xyzw(target_matrix)
    grip = float(action10[9] > gripper_threshold)
    return np.concatenate([xyz, quat_xyzw, np.asarray([grip], dtype=np.float32)], axis=0).astype(np.float32)


"""按 raw dataset 的 variation 目录重建全局 episode 顺序，和采集时的 round-robin 逻辑一致。"""
def build_demo_schedule(task_root: Path) -> list[tuple[int, int]]:
    variation_counts: list[tuple[int, int]] = []
    for variation_dir in sorted(task_root.glob("variation*"), key=lambda p: int(p.name.replace("variation", ""))):
        if not variation_dir.is_dir():
            continue
        episodes_dir = variation_dir / "episodes"
        if not episodes_dir.is_dir():
            continue
        count = len([p for p in episodes_dir.iterdir() if p.is_dir() and p.name.startswith("episode")])
        variation = int(variation_dir.name.replace("variation", ""))
        variation_counts.append((variation, count))

    if not variation_counts:
        raise FileNotFoundError(f"No variation episodes found under task root: {task_root}")

    local_index = {variation: 0 for variation, _ in variation_counts}
    schedule: list[tuple[int, int]] = []
    while True:
        appended = False
        for variation, count in variation_counts:
            if local_index[variation] < count:
                schedule.append((variation, local_index[variation]))
                local_index[variation] += 1
                appended = True
        if not appended:
            break
    return schedule


"""选出本次真正要评测的 episode 子集。"""
def select_demo_subset(task_root: Path, start_episode: int, episodes_per_task: int) -> list[tuple[int, int]]:
    schedule = build_demo_schedule(task_root)
    end_episode = start_episode + episodes_per_task
    if start_episode < 0:
        raise ValueError(f"start_episode must be >= 0, got {start_episode}")
    if end_episode > len(schedule):
        raise ValueError(
            f"Task {task_root.name} only has {len(schedule)} demo-replay episodes, "
            f"but start={start_episode} and episodes_per_task={episodes_per_task} require {end_episode}."
        )
    return schedule[start_episode:end_episode]


"""精简 prompt 日志，避免单行过长。"""
def short_prompt(prompt: str, limit: int = 96) -> str:
    prompt = " ".join(str(prompt).split())
    if len(prompt) <= limit:
        return prompt
    return prompt[: limit - 3] + "..."


"""避免全 NaN 时触发运行期 warning。"""
def safe_nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return float("nan")
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


"""从 RGB-only raw dataset 读取一个 demo，并补回已有 RGB 路径。"""
def load_rgb_only_demo(episode_path: Path):
    low_dim_path = episode_path / LOW_DIM_PICKLE
    if not low_dim_path.exists():
        raise FileNotFoundError(f"Demo low-dim pickle does not exist: {low_dim_path}")
    with low_dim_path.open("rb") as handle:
        demo = pickle.load(handle)

    step_count = len(demo)
    if step_count <= 0:
        raise ValueError(f"Demo at {episode_path} is empty.")
    if getattr(demo, "random_seed", None) is None:
        raise ValueError(f"Demo at {episode_path} is missing random_seed, cannot replay reset state.")

    for attribute_name, folder in RGB_CAMERA_SPECS:
        camera_dir = episode_path / folder
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"Expected RGB camera directory does not exist: {camera_dir}")
        for step_index in range(step_count):
            frame_path = camera_dir / (IMAGE_FORMAT % step_index)
            if not frame_path.is_file():
                raise FileNotFoundError(
                    f"Missing RGB frame for demo replay: {frame_path} "
                    f"(step_count={step_count}, camera={attribute_name})"
                )
            setattr(demo[step_index], attribute_name, str(frame_path))

    return demo


"""目录调度的 variation 是 ground truth；raw 数据里 demo.misc["variation_index"] 大面积被错误记录为 0，只用于告警。"""
def resolve_demo_variation(demo, fallback_variation: int) -> int:
    if len(demo) <= 0:
        raise ValueError("Demo has no observations.")
    misc = getattr(demo[0], "misc", None)
    if isinstance(misc, dict) and "variation_index" in misc:
        misc_variation = int(misc["variation_index"])
        if misc_variation != int(fallback_variation):
            logging.getLogger("pi05_rlbench_eval").warning(
                "Demo misc.variation_index=%d disagrees with directory variation=%d; trusting directory.",
                misc_variation,
                fallback_variation,
            )
    return int(fallback_variation)


class RLBenchEvalRuntime:
    """薄封装：管理 RLBench 环境、任务切换和 demo replay reset。"""

    def __init__(self, dataset_root: Path, image_size: int, headless: bool, arm_action_mode: str = "planning"):
        try:
            from pyrep.const import RenderMode
            from pyrep.errors import ConfigurationPathError, IKError
            from rlbench import CameraConfig, Environment, ObservationConfig
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.backend.exceptions import InvalidActionError
            from rlbench.backend.utils import task_file_to_task_class
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise ImportError("RLBench and PyRep runtime dependencies are required for evaluation.") from exc

        def clip_to_workspace(scene, action: np.ndarray) -> np.ndarray:
            action = np.asarray(action, dtype=np.float32).copy()
            action[:3] = np.clip(
                action[:3],
                np.asarray([scene._workspace_minx, scene._workspace_miny, scene._workspace_minz], dtype=np.float32)
                + 1e-7,
                np.asarray([scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz], dtype=np.float32)
                - 1e-7,
            )
            return action

        class ClippedEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):
            """Clamp target XYZ into the RLBench workspace before planning."""

            def action(self, scene, action, *args, **kwargs):
                super().action(scene, clip_to_workspace(scene, action))

        class ClippedEndEffectorPoseViaIK(EndEffectorPoseViaIK):
            """Clamp target XYZ into the RLBench workspace before IK servoing.

            密集下一帧位姿目标（毫米级步距）用 IK 伺服执行是连续运动，
            不像 planning 模式那样每个 waypoint 都完全停稳，因此没有卡顿。
            """

            def action(self, scene, action, *args, **kwargs):
                super().action(scene, clip_to_workspace(scene, action))

        from rlbench.action_modes.arm_action_modes import ArmActionMode

        class HybridEndEffectorPose(ArmActionMode):
            """IK 伺服优先（平滑无停顿）；IK 解不出（目标离当前位姿过远/接触滞后）时
            单步回退到路径规划追上目标，保证上限不低于 planning 模式。"""

            def __init__(self):
                self._ik = ClippedEndEffectorPoseViaIK(absolute_mode=True)
                self._planning = ClippedEndEffectorPoseViaPlanning()

            def action(self, scene, action):
                try:
                    self._ik.action(scene, action)
                except InvalidActionError:
                    self._planning.action(scene, action)

            def action_shape(self, scene):
                return (7,)

        if arm_action_mode == "planning":
            arm_mode = ClippedEndEffectorPoseViaPlanning()
        elif arm_action_mode == "ik":
            arm_mode = ClippedEndEffectorPoseViaIK(absolute_mode=True)
        elif arm_action_mode == "hybrid":
            arm_mode = HybridEndEffectorPose()
        else:
            raise ValueError(
                f"Unsupported arm_action_mode: {arm_action_mode} (expected 'planning', 'ik' or 'hybrid')"
            )

        disabled_camera = CameraConfig()
        disabled_camera.set_all(False)
        # 训练数据采集用的是 opengl3（有阴影/完整光照），评测渲染必须保持一致，
        # 否则策略输入存在整体视觉域偏移。
        active_camera = CameraConfig(
            rgb=True,
            point_cloud=False,
            mask=False,
            depth=False,
            image_size=[image_size, image_size],
            render_mode=RenderMode.OPENGL3,
        )
        obs_config = ObservationConfig(
            front_camera=active_camera,
            wrist_camera=active_camera,
            left_shoulder_camera=disabled_camera,
            right_shoulder_camera=disabled_camera,
            overhead_camera=disabled_camera,
            joint_forces=False,
            joint_positions=False,
            joint_velocities=False,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=False,
            gripper_joint_positions=False,
        )

        self._task_file_to_task_class = task_file_to_task_class
        self._step_exceptions = (IKError, ConfigurationPathError, InvalidActionError)
        self._dataset_root = dataset_root.resolve()
        self._env = Environment(
            action_mode=MoveArmThenGripper(arm_mode, Discrete()),
            obs_config=obs_config,
            dataset_root=str(dataset_root),
            headless=headless,
        )
        self._task = None
        self._active_task_name: str | None = None

    @property
    def step_exceptions(self):
        return self._step_exceptions

    @property
    def scene(self):
        return self._env._scene

    def launch(self) -> None:
        self._env.launch()

    def shutdown(self) -> None:
        self._env.shutdown()

    def set_task(self, task_name: str) -> None:
        if self._active_task_name == task_name:
            return
        self._task = self._env.get_task(self._task_file_to_task_class(task_name))
        self._active_task_name = task_name

    def reset_to_demo(
        self, task_name: str, variation: int, local_episode: int, prompt_strategy: str = "longest"
    ) -> tuple[Any, str, int, Any]:
        self.set_task(task_name)
        assert self._task is not None
        episode_path = (
            self._dataset_root
            / task_name
            / f"variation{variation}"
            / "episodes"
            / f"episode{local_episode}"
        )
        if not episode_path.is_dir():
            raise FileNotFoundError(f"Demo episode directory does not exist: {episode_path}")
        demo = load_rgb_only_demo(episode_path)
        demo_variation = resolve_demo_variation(demo, variation)
        # RLBench 的 task.reset_to_demo 内部会读取 demo.misc["variation_index"] 并覆盖
        # set_variation 的结果；raw 数据里该字段大面积被错误记录为 0，这里强制改写成
        # 目录 variation，否则场景会以错误的 variation 重建、与 demo 轨迹不匹配。
        first_misc = getattr(demo[0], "misc", None)
        if isinstance(first_misc, dict):
            first_misc["variation_index"] = demo_variation
        self._task.set_variation(demo_variation)
        descriptions, obs = self._task.reset_to_demo(demo)
        # 训练数据转换用的是 --prompt-strategy longest（取最长描述），
        # 评测 prompt 必须保持同一策略，否则语言输入分布与训练不一致。
        candidates = [text.strip() for text in (descriptions or self._task.get_task_descriptions()) if text and text.strip()]
        if not candidates:
            raise ValueError(f"No task descriptions available for {task_name} variation {demo_variation}.")
        # dense 训练用 longest；waypoint 训练的 manifest task_instruction=descs[0]（首条），
        # 故 waypoint 评测须用 "first" 以对齐语言分布。
        prompt = candidates[0] if prompt_strategy == "first" else max(candidates, key=len)
        return obs, prompt, demo_variation, demo

    def step(self, action: np.ndarray):
        assert self._task is not None
        return self._task.step(action)


"""执行一次 episode：固定 demo 初始场景，在线 rollout 并统计成功率/耗时。"""
def rollout_episode(
    policy,
    runtime: RLBenchEvalRuntime,
    *,
    task_name: str,
    variation: int,
    local_episode: int,
    max_steps: int,
    replan_steps: int,
    gripper_threshold: float,
    action_stride: int = 1,
    action_space: str = "absolute",
    recorder: EpisodeVideoRecorder | None = None,
) -> dict[str, Any]:
    obs, prompt, used_variation, _demo = runtime.reset_to_demo(task_name, variation, local_episode)
    policy_obs = make_policy_observation(obs, prompt)
    if recorder is not None:
        recorder.start_episode(
            task_name=task_name,
            prompt=prompt,
            variation=used_variation,
            local_episode=local_episode,
        )

    action_queue: deque[np.ndarray] = deque()
    infer_ms_values: list[float] = []
    error_type: str | None = None
    error_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    success = False
    steps_taken = 0
    policy_calls = 0
    episode_start = time.monotonic()

    for step_index in range(max_steps):
        if not action_queue:
            policy_output = policy.infer(policy_obs)
            action_chunk = np.asarray(policy_output["actions"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
                raise ValueError(f"Policy returned invalid action chunk shape: {action_chunk.shape}")
            # 目标是绝对位姿，隔 action_stride 取一执行是合法的，可放大单步位移。
            strided_chunk = action_chunk[::action_stride]
            for action10 in strided_chunk[: min(replan_steps, strided_chunk.shape[0])]:
                action_queue.append(np.asarray(action10, dtype=np.float32))
            infer_ms = policy_output.get("policy_timing", {}).get("infer_ms")
            if infer_ms is not None:
                infer_ms_values.append(float(infer_ms))
            policy_calls += 1

        if action_space == "delta":
            action8 = delta_action10_to_rlbench_action8(
                action_queue.popleft(),
                np.asarray(obs.gripper_pose, dtype=np.float32),
                gripper_threshold,
            )
        else:
            action8 = policy_action10_to_rlbench_action8(action_queue.popleft(), gripper_threshold)
        steps_taken = step_index + 1
        try:
            obs, reward, terminal = runtime.step(action8)
        except runtime.step_exceptions as exc:
            # 非法动作不再直接判死：丢弃剩余 chunk 强制重新推理，给策略自我恢复机会；
            # 连续失败超限才终止，避免死循环。
            error_type = type(exc).__name__
            error_count += 1
            consecutive_errors += 1
            logging.getLogger("pi05_rlbench_eval").warning(
                "Step rejected (%s) at step=%d consecutive=%d action8=%s",
                error_type,
                steps_taken,
                consecutive_errors,
                np.array2string(action8, precision=4, suppress_small=True),
            )
            action_queue.clear()
            if consecutive_errors >= max_consecutive_errors:
                break
            continue

        consecutive_errors = 0
        success = bool(reward >= 1.0)
        if success or terminal:
            break

        policy_obs = make_policy_observation(obs, prompt)

    wall_time_s = time.monotonic() - episode_start
    video_path = None
    if recorder is not None:
        video_path = recorder.end_episode(success=success, error_type=error_type)
    return {
        "task": task_name,
        "prompt": prompt,
        "variation": used_variation,
        "local_episode": local_episode,
        "success": success,
        "video_path": str(video_path) if video_path is not None else None,
        "steps": steps_taken,
        "policy_calls": policy_calls,
        "infer_ms_mean": float(np.mean(infer_ms_values)) if infer_ms_values else float("nan"),
        "infer_ms_last": float(infer_ms_values[-1]) if infer_ms_values else float("nan"),
        "wall_time_s": wall_time_s,
        "error_type": error_type,
        "error_count": error_count,
    }


"""上限测试：绕过 policy，把 demo 的下一帧位姿经过与训练/评测完全相同的
8D->10D->8D 转换链路后逐步执行，验证动作表示与执行语义的可达上限。"""
def rollout_replay_episode(
    runtime: RLBenchEvalRuntime,
    *,
    task_name: str,
    variation: int,
    local_episode: int,
    gripper_threshold: float,
    recorder: EpisodeVideoRecorder | None = None,
    use_delta: bool = False,
    action_stride: int = 3,
) -> dict[str, Any]:
    from openpi.policies.rlbench_policy import _pose8_to_pose10

    obs, prompt, used_variation, demo = runtime.reset_to_demo(task_name, variation, local_episode)
    if recorder is not None:
        recorder.start_episode(
            task_name=task_name,
            prompt=prompt,
            variation=used_variation,
            local_episode=local_episode,
        )

    def demo_pose8(step_obs) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(step_obs.gripper_pose, dtype=np.float32),
                np.asarray([float(step_obs.gripper_open)], dtype=np.float32),
            ],
            axis=0,
        )

    # action10_sequence: 每步要下发的 10D 动作。
    #  - absolute(默认): action[t] = demo t+1 帧的位姿(rot6d),经 policy_action10_to_rlbench_action8 直接下发。
    #  - delta 诊断: 用训练数据同款 state_pair_to_delta_action 在 stride-3 保留帧间算真值 delta,
    #    经 delta_action10_to_rlbench_action8(delta, 当前实际位姿) 合成——与 policy 评测路径完全一致。
    action10_sequence: list[np.ndarray] = []
    if use_delta:
        from tools.convert_rlbench_data_to_lerobot import state_pair_to_delta_action

        all_states = [demo_pose8(o) for o in list(demo)]
        num_obs = len(all_states)
        kept_indices = list(range(0, num_obs, max(1, action_stride)))
        if kept_indices[-1] != num_obs - 1:
            kept_indices.append(num_obs - 1)  # 补末帧,与转换脚本一致
        kept_states = [all_states[i] for i in kept_indices]
        for i in range(len(kept_states) - 1):
            delta8 = state_pair_to_delta_action(kept_states[i], kept_states[i + 1])
            action10_sequence.append(_pose8_to_pose10(delta8))  # [dxyz, rot6d(dquat), grip]
    else:
        for step_obs in list(demo)[1:]:
            action10_sequence.append(_pose8_to_pose10(demo_pose8(step_obs)))

    error_type: str | None = None
    error_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    success = False
    steps_taken = 0
    episode_start = time.monotonic()

    for step_index, action10 in enumerate(action10_sequence):
        if use_delta:
            action8 = delta_action10_to_rlbench_action8(
                action10, np.asarray(obs.gripper_pose, dtype=np.float32), gripper_threshold
            )
        else:
            action8 = policy_action10_to_rlbench_action8(action10, gripper_threshold)
        steps_taken = step_index + 1
        try:
            obs, reward, terminal = runtime.step(action8)
        except runtime.step_exceptions as exc:
            # 重放时跳过被拒绝的动作继续执行后续序列；连续失败超限才终止。
            error_type = type(exc).__name__
            error_count += 1
            consecutive_errors += 1
            logging.getLogger("pi05_rlbench_eval").warning(
                "Replay step rejected (%s) at step=%d consecutive=%d action8=%s",
                error_type,
                steps_taken,
                consecutive_errors,
                np.array2string(action8, precision=4, suppress_small=True),
            )
            if consecutive_errors >= max_consecutive_errors:
                break
            continue

        consecutive_errors = 0
        success = bool(reward >= 1.0)
        if success or terminal:
            break

    wall_time_s = time.monotonic() - episode_start
    video_path = None
    if recorder is not None:
        video_path = recorder.end_episode(success=success, error_type=error_type)
    return {
        "task": task_name,
        "prompt": prompt,
        "variation": used_variation,
        "local_episode": local_episode,
        "success": success,
        "video_path": str(video_path) if video_path is not None else None,
        "steps": steps_taken,
        "policy_calls": 0,
        "infer_ms_mean": float("nan"),
        "infer_ms_last": float("nan"),
        "wall_time_s": wall_time_s,
        "error_type": error_type,
        "error_count": error_count,
    }


def rollout_replay_keyframes_episode(
    runtime: RLBenchEvalRuntime,
    *,
    task_name: str,
    variation: int,
    local_episode: int,
    gripper_threshold: float,
    recorder: EpisodeVideoRecorder | None = None,
    stopping_delta: float = 0.1,
    max_seg: int = 0,
) -> dict[str, Any]:
    """Waypoint-paradigm ceiling: execute only the heuristic keyframe poses of the
    loaded demo through the planner. Keyframes come from TAVP keypoint_discovery on
    the demo (same algorithm as the training-data manifest); with max_seg>0 long
    segments are subdivided (same data-side densification as the manifest). Each
    keyframe's absolute pose is sent as one planner target — mirroring how the waypoint
    model will act, but with ground-truth waypoints instead of predictions."""
    import sys as _sys
    _wp_dir = str(Path(__file__).resolve().parent.parent / "tools" / "waypoint")
    if _wp_dir not in _sys.path:
        _sys.path.insert(0, _wp_dir)
    from keyframe import densify_waypoints, keypoint_discovery
    from openpi.policies.rlbench_policy import _pose8_to_pose10

    obs, prompt, used_variation, demo = runtime.reset_to_demo(task_name, variation, local_episode)
    if recorder is not None:
        recorder.start_episode(
            task_name=task_name,
            prompt=prompt,
            variation=used_variation,
            local_episode=local_episode,
        )

    demo_obs = list(demo)
    keyframes = keypoint_discovery(demo_obs, stopping_delta=stopping_delta)
    if max_seg and max_seg > 0:
        keyframes = densify_waypoints(keyframes, max_seg)

    def demo_pose8(step_obs) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(step_obs.gripper_pose, dtype=np.float32),
                np.asarray([float(step_obs.gripper_open)], dtype=np.float32),
            ],
            axis=0,
        )

    action10_sequence = [_pose8_to_pose10(demo_pose8(demo_obs[k])) for k in keyframes]

    error_type: str | None = None
    error_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    success = False
    steps_taken = 0
    episode_start = time.monotonic()

    for step_index, action10 in enumerate(action10_sequence):
        action8 = policy_action10_to_rlbench_action8(action10, gripper_threshold)
        steps_taken = step_index + 1
        try:
            obs, reward, terminal = runtime.step(action8)
        except runtime.step_exceptions as exc:
            error_type = type(exc).__name__
            error_count += 1
            consecutive_errors += 1
            logging.getLogger("pi05_rlbench_eval").warning(
                "Keyframe replay step rejected (%s) at keyframe=%d/%d consecutive=%d action8=%s",
                error_type,
                steps_taken,
                len(action10_sequence),
                consecutive_errors,
                np.array2string(action8, precision=4, suppress_small=True),
            )
            if consecutive_errors >= max_consecutive_errors:
                break
            continue

        consecutive_errors = 0
        success = bool(reward >= 1.0)
        if success or terminal:
            break

    wall_time_s = time.monotonic() - episode_start
    video_path = None
    if recorder is not None:
        video_path = recorder.end_episode(success=success, error_type=error_type)
    logging.getLogger("pi05_rlbench_eval").info(
        "Keyframe replay | task=%s ep=%d keyframes=%d executed=%d success=%d",
        task_name,
        local_episode,
        len(keyframes),
        steps_taken,
        int(success),
    )
    return {
        "task": task_name,
        "prompt": prompt,
        "variation": used_variation,
        "local_episode": local_episode,
        "success": success,
        "video_path": str(video_path) if video_path is not None else None,
        "steps": steps_taken,
        "policy_calls": 0,
        "infer_ms_mean": float("nan"),
        "infer_ms_last": float("nan"),
        "wall_time_s": wall_time_s,
        "error_type": error_type,
        "error_count": error_count,
        "num_keyframes": len(keyframes),
    }


def rollout_waypoint_episode(
    policy,
    runtime: RLBenchEvalRuntime,
    *,
    task_name: str,
    variation: int,
    local_episode: int,
    gripper_threshold: float,
    max_waypoints: int = 30,
    recorder: EpisodeVideoRecorder | None = None,
) -> dict[str, Any]:
    """Waypoint 模型闭环评测：每步用 policy 预测下一个绝对关键帧位姿(rotvec7)，转成
    RLBench 8D 动作经规划器执行，再重新观测预测下一个，直到成功/终止/达到 max_waypoints。
    与训练一致：3 视角 + ee_rotvec state + 首条描述 prompt。"""
    import sys as _sys

    _wp_dir = str(Path(__file__).resolve().parent.parent / "tools" / "waypoint")
    if _wp_dir not in _sys.path:
        _sys.path.insert(0, _wp_dir)
    from rlbench_pi05_waypoint.common import rotvec_to_quat

    obs, prompt, used_variation, _demo = runtime.reset_to_demo(
        task_name, variation, local_episode, prompt_strategy="first"
    )
    if recorder is not None:
        recorder.start_episode(
            task_name=task_name, prompt=prompt, variation=used_variation, local_episode=local_episode
        )

    infer_ms_values: list[float] = []
    error_type: str | None = None
    error_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    success = False
    steps_taken = 0
    policy_calls = 0
    episode_start = time.monotonic()

    for wp_index in range(max_waypoints):
        policy_obs = make_waypoint_policy_observation(obs, prompt)
        policy_output = policy.infer(policy_obs)
        act = np.asarray(policy_output["actions"], dtype=np.float32)
        if act.ndim == 2:  # [horizon=1, 7]
            act = act[0]
        act = act[:7]
        xyz = act[:3]
        quat_xyzw = np.asarray(rotvec_to_quat(act[3:6]), dtype=np.float32)
        grip = float(act[6] > gripper_threshold)
        action8 = np.concatenate([xyz, quat_xyzw, np.asarray([grip], dtype=np.float32)], axis=0).astype(np.float32)
        policy_calls += 1
        steps_taken = wp_index + 1
        infer_ms = policy_output.get("policy_timing", {}).get("infer_ms")
        if infer_ms is not None:
            infer_ms_values.append(float(infer_ms))

        try:
            obs, reward, terminal = runtime.step(action8)
        except runtime.step_exceptions as exc:
            error_type = type(exc).__name__
            error_count += 1
            consecutive_errors += 1
            logging.getLogger("pi05_rlbench_eval").warning(
                "Waypoint step rejected (%s) at waypoint=%d consecutive=%d action8=%s",
                error_type,
                steps_taken,
                consecutive_errors,
                np.array2string(action8, precision=4, suppress_small=True),
            )
            if consecutive_errors >= max_consecutive_errors:
                break
            continue

        consecutive_errors = 0
        success = bool(reward >= 1.0)
        if success or terminal:
            break

    wall_time_s = time.monotonic() - episode_start
    video_path = None
    if recorder is not None:
        video_path = recorder.end_episode(success=success, error_type=error_type)
    return {
        "task": task_name,
        "prompt": prompt,
        "variation": used_variation,
        "local_episode": local_episode,
        "success": success,
        "video_path": str(video_path) if video_path is not None else None,
        "steps": steps_taken,
        "policy_calls": policy_calls,
        "infer_ms_mean": float(np.mean(infer_ms_values)) if infer_ms_values else float("nan"),
        "infer_ms_last": float(infer_ms_values[-1]) if infer_ms_values else float("nan"),
        "wall_time_s": wall_time_s,
        "error_type": error_type,
        "error_count": error_count,
    }


"""Exp0：从 planner-cache 读取 val split 的原子分段。

只依赖 cache 的 JSON 字段，不 import data/planner_cache.py —— 那个模块的校验器要求
cache_key == output_episode_index（train 侧 LeRobot 索引），而 val split 没有 LeRobot
转换，其 cache_key 会退化成 raw_episode_key。这里两种键都接受。
"""
def load_atom_segments(
    cache_path: Path,
    *,
    task_names: list[str],
    episodes_per_task: int,
    skip_actions: set[str],
    logger: logging.Logger,
) -> dict[str, list[dict[str, Any]]]:
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Atom-mode planner cache does not exist: {cache_path}. "
            "Generate it first with tools/build_subtask_cache.py against the val split."
        )
    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, dict) or not episodes_payload:
        raise ValueError(f"{cache_path} has no 'episodes' object.")

    wanted = set(task_names)
    # task -> variation -> episode -> record，用于按任务挑前 N 条 episode。
    by_task: dict[str, list[dict[str, Any]]] = {name: [] for name in task_names}
    skipped_review = 0
    for cache_key, record in episodes_payload.items():
        if not isinstance(record, dict):
            continue
        task_name = str(record.get("task_name", "")).strip()
        if task_name not in wanted:
            continue
        if bool(record.get("needs_review", False)):
            skipped_review += 1
            continue
        variation_raw = str(record.get("variation", "")).strip()
        episode_raw = str(record.get("episode", "")).strip()
        try:
            variation = int(variation_raw.replace("variation", ""))
            local_episode = int(episode_raw.replace("episode", ""))
        except ValueError as exc:
            raise ValueError(
                f"{cache_path}:{cache_key} has unparsable variation/episode "
                f"({variation_raw!r}/{episode_raw!r})."
            ) from exc
        by_task[task_name].append(
            {
                "cache_key": cache_key,
                "task_name": task_name,
                "variation": variation,
                "local_episode": local_episode,
                "task_instruction": str(record.get("task_instruction", "")).strip(),
                "all_frames": int(record.get("all_frames", 0)),
                "segments": record.get("segments") or [],
            }
        )

    if skipped_review:
        logger.warning("Atom cache: skipped %d episode(s) flagged needs_review.", skipped_review)

    plans: dict[str, list[dict[str, Any]]] = {}
    for task_name in task_names:
        records = by_task[task_name]
        if not records:
            raise ValueError(
                f"Atom cache {cache_path} contains no usable episode for task={task_name}."
            )
        # 固定顺序（variation, episode）后取前 N 条，保证可复现。
        records.sort(key=lambda item: (item["variation"], item["local_episode"]))
        selected = records[:episodes_per_task]
        if len(selected) < episodes_per_task:
            logger.warning(
                "Atom cache: task=%s only has %d episode(s), requested %d.",
                task_name,
                len(selected),
                episodes_per_task,
            )
        entries: list[dict[str, Any]] = []
        for record in selected:
            task_instruction = record["task_instruction"]
            if not task_instruction:
                raise ValueError(f"{cache_path}:{record['cache_key']} is missing task_instruction.")
            for segment in record["segments"]:
                action = str(segment.get("action", "")).strip()
                if action in skip_actions:
                    continue
                instruction = str(segment.get("instruction", "")).strip()
                if not instruction:
                    raise ValueError(
                        f"{cache_path}:{record['cache_key']} segment "
                        f"{segment.get('segment_index')} is missing instruction."
                    )
                start_frame = int(segment["raw_start_frame"])
                end_frame = int(segment["raw_end_frame"])
                if not 0 <= start_frame <= end_frame:
                    raise ValueError(
                        f"{cache_path}:{record['cache_key']} segment "
                        f"{segment.get('segment_index')} has invalid raw frame range "
                        f"[{start_frame}, {end_frame}]."
                    )
                entries.append(
                    {
                        "task_name": task_name,
                        "variation": record["variation"],
                        "local_episode": record["local_episode"],
                        "segment_index": int(segment.get("segment_index", len(entries))),
                        "action": action,
                        "atomic_instruction": instruction,
                        "task_instruction": task_instruction,
                        "raw_start_frame": start_frame,
                        "raw_end_frame": end_frame,
                    }
                )
        if not entries:
            raise ValueError(
                f"Atom cache {cache_path}: task={task_name} produced no segment after "
                f"skipping actions {sorted(skip_actions)}."
            )
        plans[task_name] = entries
    return plans


"""构造一条"评测出错"的占位结果，使单局异常不拖垮整轮评测、且 summary 字段完整。

用于 RLBench 偶发的场景放置失败(TaskEnvironmentError)等可恢复异常：把该局记为
success=False + errored=True，评测继续下一局，成功率分母仍计入(视为失败局)。
"""
def make_errored_result(
    task_name: str, variation: int, local_episode: int, error_type: str
) -> dict[str, Any]:
    return {
        "task": task_name,
        "prompt": "",
        "variation": int(variation),
        "local_episode": int(local_episode),
        "success": False,
        "errored": True,
        "video_path": None,
        "steps": 0,
        "policy_calls": 0,
        "infer_ms_mean": float("nan"),
        "infer_ms_last": float("nan"),
        "wall_time_s": 0.0,
        "error_type": error_type,
        "error_count": 1,
    }


"""Exp0：把 demo 前缀重放到原子段起点，再把控制权交给 policy 执行该原子动作。

与 rollout_replay_episode 共用同一条 8D->10D->8D 动作管线，保证前缀重放的落点
与 replay 上限实验一致；交接后的推理路径与 rollout_episode 完全一致。
"""
def rollout_atom_episode(
    policy,
    runtime: RLBenchEvalRuntime,
    *,
    entry: dict[str, Any],
    prompt_condition: str,
    max_steps: int,
    replan_steps: int,
    gripper_threshold: float,
    action_space: str,
    recorder: EpisodeVideoRecorder | None = None,
    logger: logging.Logger,
) -> dict[str, Any]:
    from openpi.policies.rlbench_policy import _pose8_to_pose10

    task_name = entry["task_name"]
    start_frame = entry["raw_start_frame"]
    obs, full_prompt, used_variation, demo = runtime.reset_to_demo(
        task_name, entry["variation"], entry["local_episode"]
    )
    demo_steps = list(demo)
    if start_frame >= len(demo_steps):
        raise ValueError(
            f"Segment start_frame={start_frame} exceeds demo length {len(demo_steps)} "
            f"for {task_name}/var{entry['variation']}/ep{entry['local_episode']}."
        )

    # ---- 阶段 1：重放前缀（不录像，录像从交接点开始，让人工只判 policy 的行为）----
    prefix_error: str | None = None
    prefix_steps = 0
    for step_obs in demo_steps[1 : start_frame + 1]:
        pose8 = np.concatenate(
            [
                np.asarray(step_obs.gripper_pose, dtype=np.float32),
                np.asarray([float(step_obs.gripper_open)], dtype=np.float32),
            ],
            axis=0,
        )
        action8 = policy_action10_to_rlbench_action8(_pose8_to_pose10(pose8), gripper_threshold)
        prefix_steps += 1
        try:
            obs, _reward, _terminal = runtime.step(action8)
        except runtime.step_exceptions as exc:
            # 前缀重放失败意味着起始状态不可信，该段作废（不计入成功率分母）。
            prefix_error = type(exc).__name__
            logger.warning(
                "Atom prefix replay rejected (%s) at prefix_step=%d task=%s var=%d ep=%d seg=%d",
                prefix_error,
                prefix_steps,
                task_name,
                entry["variation"],
                entry["local_episode"],
                entry["segment_index"],
            )
            break

    prompt = entry["atomic_instruction"] if prompt_condition == "atomic" else full_prompt

    if prefix_error is not None:
        return {
            "task": task_name,
            "prompt": prompt,
            "variation": used_variation,
            "local_episode": entry["local_episode"],
            "segment_index": entry["segment_index"],
            "action": entry["action"],
            "prompt_condition": prompt_condition,
            "prefix_frames": start_frame,
            "prefix_error": prefix_error,
            "valid": False,
            "success": False,
            "video_path": None,
            "steps": 0,
            "policy_calls": 0,
            "infer_ms_mean": float("nan"),
            "infer_ms_last": float("nan"),
            "wall_time_s": 0.0,
            "error_type": prefix_error,
            "error_count": 1,
        }

    # ---- 阶段 2：policy 接管 ----
    # 每个 (segment, condition) 用独立的录像分桶，既保证全部录下来，也避免文件名撞车。
    video_bucket = (
        f"{task_name}__seg{entry['segment_index']:02d}_{entry['action']}__{prompt_condition}"
    )
    if recorder is not None:
        recorder.start_episode(
            task_name=video_bucket,
            prompt=prompt,
            variation=used_variation,
            local_episode=entry["local_episode"],
        )

    policy_obs = make_policy_observation(obs, prompt)
    action_queue: deque[np.ndarray] = deque()
    infer_ms_values: list[float] = []
    error_type: str | None = None
    error_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    task_success = False
    steps_taken = 0
    policy_calls = 0
    episode_start = time.monotonic()

    for step_index in range(max_steps):
        if not action_queue:
            policy_output = policy.infer(policy_obs)
            action_chunk = np.asarray(policy_output["actions"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
                raise ValueError(f"Policy returned invalid action chunk shape: {action_chunk.shape}")
            for action10 in action_chunk[: min(replan_steps, action_chunk.shape[0])]:
                action_queue.append(np.asarray(action10, dtype=np.float32))
            infer_ms = policy_output.get("policy_timing", {}).get("infer_ms")
            if infer_ms is not None:
                infer_ms_values.append(float(infer_ms))
            policy_calls += 1

        if action_space == "delta":
            action8 = delta_action10_to_rlbench_action8(
                action_queue.popleft(),
                np.asarray(obs.gripper_pose, dtype=np.float32),
                gripper_threshold,
            )
        else:
            action8 = policy_action10_to_rlbench_action8(action_queue.popleft(), gripper_threshold)
        steps_taken = step_index + 1
        try:
            obs, reward, terminal = runtime.step(action8)
        except runtime.step_exceptions as exc:
            error_type = type(exc).__name__
            error_count += 1
            consecutive_errors += 1
            action_queue.clear()
            if consecutive_errors >= max_consecutive_errors:
                break
            continue

        consecutive_errors = 0
        # 注意：这是**任务级**成功判据。原子动作是否成功由人工看视频判定
        # （RLBench 没有原子级 success detector，见 Exp_Design.md §3.2）。
        task_success = bool(reward >= 1.0)
        if task_success or terminal:
            break

        policy_obs = make_policy_observation(obs, prompt)

    wall_time_s = time.monotonic() - episode_start
    video_path = None
    if recorder is not None:
        # 录像分桶按 success/fail 各留配额；原子段一律按 fail 归档以走同一目录，
        # 真正的成败由人工判定表决定。
        video_path = recorder.end_episode(success=False, error_type=error_type)
    return {
        "task": task_name,
        "prompt": prompt,
        "variation": used_variation,
        "local_episode": entry["local_episode"],
        "segment_index": entry["segment_index"],
        "action": entry["action"],
        "prompt_condition": prompt_condition,
        "prefix_frames": start_frame,
        "prefix_error": None,
        "valid": True,
        "success": task_success,
        "video_path": str(video_path) if video_path is not None else None,
        "steps": steps_taken,
        "policy_calls": policy_calls,
        "infer_ms_mean": float(np.mean(infer_ms_values)) if infer_ms_values else float("nan"),
        "infer_ms_last": float(infer_ms_values[-1]) if infer_ms_values else float("nan"),
        "wall_time_s": wall_time_s,
        "error_type": error_type,
        "error_count": error_count,
    }


"""教师强制预测精度诊断：用 demo 绝对位姿让机械臂始终在轨(teacher forcing),
在每个 stride-3 帧比较 policy 预测的 delta 与真值 delta。因手臂不偏离 demo 轨迹,
测的是**纯预测精度**(无 rollout 累积漂移),用于定位"哪一阶段预测不准"。"""
def rollout_predict_diag_episode(
    policy,
    runtime: RLBenchEvalRuntime,
    *,
    task_name: str,
    variation: int,
    local_episode: int,
    gripper_threshold: float,
    action_stride: int = 3,
) -> dict[str, Any]:
    from openpi.policies.rlbench_policy import _pose8_to_pose10
    from tools.convert_rlbench_data_to_lerobot import state_pair_to_delta_action

    obs, prompt, used_variation, demo = runtime.reset_to_demo(task_name, variation, local_episode)
    all_states = [
        np.concatenate(
            [np.asarray(o.gripper_pose, dtype=np.float32), np.asarray([float(o.gripper_open)], dtype=np.float32)]
        )
        for o in list(demo)
    ]
    num_obs = len(all_states)
    kept = list(range(0, num_obs, max(1, action_stride)))
    if kept[-1] != num_obs - 1:
        kept.append(num_obs - 1)

    records: list[dict[str, Any]] = []
    for ki in range(len(kept) - 1):
        t, t_next = kept[ki], kept[ki + 1]
        policy_obs = make_policy_observation(obs, prompt)
        pred10 = np.asarray(policy.infer(policy_obs)["actions"], dtype=np.float32)[0]
        gt10 = _pose8_to_pose10(state_pair_to_delta_action(all_states[t], all_states[t_next]))
        grip_now = all_states[t][7] > 0.5
        grip_next = all_states[t_next][7] > 0.5
        records.append(
            {
                "frame": int(t),
                "xyz_err": float(np.linalg.norm(pred10[:3] - gt10[:3])),
                "rot6d_err": float(np.linalg.norm(pred10[3:9] - gt10[3:9])),
                "grip_err": float(abs(float(pred10[9]) - float(gt10[9]))),
                "is_grip_transition": bool(grip_now != grip_next),
                "gt_xyz_mag": float(np.linalg.norm(gt10[:3])),
            }
        )
        # 教师强制: 用 demo 绝对位姿逐帧推进到 t_next,让手臂回到 demo 轨迹。
        broke = False
        for tt in range(t + 1, t_next + 1):
            action8 = policy_action10_to_rlbench_action8(_pose8_to_pose10(all_states[tt]), gripper_threshold)
            try:
                obs, _reward, terminal = runtime.step(action8)
            except runtime.step_exceptions:
                broke = True
                break
            if terminal:
                break
        if broke:
            break
    return {
        "task": task_name,
        "variation": used_variation,
        "local_episode": local_episode,
        "records": records,
    }


"""主流程：解析配置、加载 policy、逐任务评测并写单日志汇总。"""
def main() -> int:
    args = parse_args()
    # --replay-delta 复用 --replay-demo 的全部分支(不加载 policy、checkpoint_source=replay 等),
    # 仅在 rollout 时切到 delta 路径。二者不可同时指定。
    if args.replay_delta and args.replay_demo:
        raise ValueError("--replay-demo and --replay-delta are mutually exclusive.")
    if args.replay_keyframes and (args.replay_demo or args.replay_delta):
        raise ValueError("--replay-keyframes is mutually exclusive with --replay-demo/--replay-delta.")
    if args.waypoint_eval and (args.replay_demo or args.replay_delta or args.replay_keyframes or args.predict_diag or args.atom_mode):
        raise ValueError("--waypoint-eval is mutually exclusive with replay/predict-diag/atom modes.")
    replay_use_delta = bool(args.replay_delta)
    if args.replay_delta:
        args.replay_demo = True
    # --replay-keyframes 复用 --replay-demo 的无 policy 分支(checkpoint_source=replay 等),
    # 仅在 rollout 时切到关键帧执行路径。
    if args.replay_keyframes:
        args.replay_demo = True
    config = apply_cli_overrides(load_yaml_config(args.config), args)
    if args.replay_demo:
        checkpoint_source = "replay"
    elif config["checkpoint"].get("dir_override"):
        checkpoint_source = Path(config["checkpoint"]["dir_override"]).name
    else:
        checkpoint_source = config["checkpoint"]["which"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = resolve_path_like(config["logging"]["log_root"])
    assert isinstance(log_root, Path)
    run_name = f"{config['logging']['log_name_prefix']}_{checkpoint_source}_{timestamp}"
    log_path = log_root / f"{run_name}.log"
    logger = init_logging(log_path)

    set_random_seed(int(config["runtime"]["seed"]))
    if args.replay_demo:
        device = None
        train_config = None
        checkpoint_info = None
        norm_stats_source = None
    else:
        device = resolve_runtime_device(str(config["runtime"]["device"]))
        train_config = build_policy_train_config(config["checkpoint"]["train_config_name"])
        checkpoint_info = resolve_checkpoint_source(config)
        norm_stats_source = pick_norm_stats_source(checkpoint_info["norm_stats_candidates"])

    dataset_root = resolve_path_like(config["dataset"]["raw_root"])
    assert isinstance(dataset_root, Path)
    split = str(config["dataset"]["split"])
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Dataset split root does not exist: {split_root}")

    task_names = config["tasks"]["names"]
    if not task_names:
        raise ValueError("No tasks specified in config.tasks.names.")
    # --tasks 已在 apply_cli_overrides 中生效；--task-set 只在未显式给 --tasks 时应用，
    # 两者同时给出必然有一个被忽略，直接报错而不是静默选一个。
    if args.task_set:
        if args.tasks:
            raise ValueError("--tasks and --task-set are mutually exclusive; pass only one.")
        task_names = resolve_task_set(args.task_set, task_names)
        config["tasks"]["names"] = task_names

    start_episode = int(config["dataset"]["start_episode"])
    episodes_per_task = int(config["dataset"]["episodes_per_task"])
    max_steps = int(config["environment"]["max_steps"])
    replan_steps = int(config["environment"]["replan_steps"])
    gripper_threshold = float(config["environment"]["gripper_threshold"])
    arm_action_mode = str(config["environment"].get("arm_action_mode", "planning"))
    action_stride = int(config["environment"].get("action_stride", 1))
    if action_stride < 1:
        raise ValueError(f"environment.action_stride must be >= 1, got {action_stride}")
    action_space = str(config["checkpoint"].get("action_space", "absolute"))
    if action_space not in ("absolute", "delta"):
        raise ValueError(f"checkpoint.action_space must be absolute|delta, got {action_space}")
    if action_space == "delta" and action_stride != 1:
        # delta 是相邻保留帧之间的增量，跳步执行会丢运动量；stride 应在数据转换端设置。
        raise ValueError("action_stride > 1 is only valid for absolute action space.")
    variation_mode = str(config["dataset"]["variation_mode"])
    if replan_steps < 1:
        raise ValueError(f"environment.replan_steps must be >= 1, got {replan_steps}")
    if episodes_per_task < 1:
        raise ValueError(f"dataset.episodes_per_task must be >= 1, got {episodes_per_task}")
    if variation_mode != "demo_replay":
        raise ValueError(f"Only variation_mode=demo_replay is supported right now, got {variation_mode}")

    recording_cfg = dict(config.get("recording") or {})
    recording_enabled = bool(recording_cfg.get("enabled", False))
    video_root: Path | None = None
    if recording_enabled:
        video_save_dir = resolve_path_like(recording_cfg.get("save_dir", "result"))
        assert isinstance(video_save_dir, Path)
        video_root = video_save_dir / run_name

    if args.atom_mode:
        if args.replay_demo:
            raise ValueError("--atom-mode and --replay-demo are mutually exclusive.")
        if args.atom_cache is None:
            raise ValueError("--atom-mode requires --atom-cache pointing at the val planner cache.")
        if args.atom_max_steps_factor <= 0:
            raise ValueError("--atom-max-steps-factor must be > 0.")
        if args.atom_episodes_per_task < 1:
            raise ValueError("--atom-episodes-per-task must be >= 1.")

    task_plans: dict[str, list[tuple[int, int]]] = {}
    atom_plans: dict[str, list[dict[str, Any]]] = {}
    if args.atom_mode:
        skip_actions = {item.strip() for item in str(args.atom_skip_actions).split(",") if item.strip()}
        atom_plans = load_atom_segments(
            args.atom_cache,
            task_names=task_names,
            episodes_per_task=int(args.atom_episodes_per_task),
            skip_actions=skip_actions,
            logger=logger,
        )
    else:
        plan_eps = int(args.predict_diag_episodes) if args.predict_diag else episodes_per_task
        for task_name in task_names:
            task_root = split_root / task_name
            if not task_root.is_dir():
                raise FileNotFoundError(f"Task raw data directory does not exist: {task_root}")
            task_plans[task_name] = select_demo_subset(task_root, start_episode, plan_eps)

    logger.info("bash> %s", " ".join(sys.argv))
    logger.info("Log file: %s", log_path)
    logger.info("Checkpoint source: %s", checkpoint_source)
    if args.replay_keyframes:
        logger.info("Replay mode: executing heuristic KEYFRAME poses through planner, no policy is loaded.")
    elif args.replay_demo:
        logger.info("Replay mode: executing demo next-frame poses, no policy is loaded.")
    else:
        logger.info("Resolved checkpoint dir: %s", checkpoint_info["checkpoint_dir"])
        if checkpoint_info["index_path"] is not None:
            logger.info("Resolved checkpoint index: %s", checkpoint_info["index_path"])
        logger.info("Norm stats source: %s", norm_stats_source)
        logger.info("Runtime device: %s", device)
    logger.info(
        "Eval config | split=%s variation_mode=%s tasks=%d episodes_per_task=%d start_episode=%d max_steps=%d replan_steps=%d arm_action_mode=%s action_stride=%d action_space=%s",
        split,
        variation_mode,
        len(task_names),
        episodes_per_task,
        start_episode,
        max_steps,
        replan_steps,
        arm_action_mode,
        action_stride,
        action_space,
    )
    if recording_enabled:
        logger.info(
            "Recording config | videos_per_task=%s (success/fail each) video_root=%s",
            recording_cfg.get("videos_per_task", 1),
            video_root,
        )
    else:
        logger.info("Recording disabled.")
    if args.atom_mode:
        conditions = ("atomic", "full") if args.atom_prompt == "both" else (args.atom_prompt,)
        total_segments = sum(len(entries) for entries in atom_plans.values())
        logger.info(
            "Atom mode (Exp0) | cache=%s tasks=%d segments=%d conditions=%s rollouts=%d "
            "episodes_per_task=%d skip_actions=%s max_steps_factor=%.2f min_max_steps=%d",
            args.atom_cache,
            len(atom_plans),
            total_segments,
            ",".join(conditions),
            total_segments * len(conditions),
            args.atom_episodes_per_task,
            args.atom_skip_actions,
            args.atom_max_steps_factor,
            args.atom_min_max_steps,
        )
        for task_name in task_names:
            actions = [entry["action"] for entry in atom_plans[task_name]]
            logger.info(
                "Atom plan | task=%s segments=%d actions=%s",
                task_name,
                len(atom_plans[task_name]),
                ",".join(actions[:12]) + ("..." if len(actions) > 12 else ""),
            )
    else:
        for task_name in task_names:
            logger.info("Task plan | task=%s episodes=%d first_entries=%s", task_name, len(task_plans[task_name]), task_plans[task_name][:5])

    if args.dry_run:
        logger.info("Dry run finished. Config and checkpoint resolution succeeded.")
        return 0

    policy = None
    if not args.replay_demo:
        norm_stats = load_norm_stats(norm_stats_source)
        policy = create_policy(train_config, checkpoint_info["checkpoint_dir"], norm_stats, device)
    # cv2 首次 import 时会把 Qt 插件路径指到自带目录；必须在 sanitize 之前触发,
    # 否则 replay 模式下录像器延迟导入 cv2 会在 sanitize 之后重新污染环境导致 CoppeliaSim 崩溃。
    try:
        import cv2  # noqa: F401
    except ImportError:
        pass
    sanitize_rlbench_gui_env(logger)
    install_numpy_pickle_compat(logger)

    runtime = RLBenchEvalRuntime(
        dataset_root=split_root,
        image_size=int(config["environment"]["image_size"]),
        headless=bool(config["environment"]["headless"]),
        arm_action_mode=arm_action_mode,
    )

    recorder: EpisodeVideoRecorder | None = None
    if recording_enabled:
        assert video_root is not None
        recorder = EpisodeVideoRecorder(
            video_root,
            videos_per_task=int(recording_cfg.get("videos_per_task", 1)),
            resolution=tuple(recording_cfg.get("resolution", [640, 480])),
            fps=int(recording_cfg.get("fps", 30)),
            rotate_speed=float(recording_cfg.get("rotate_speed", 0.005)),
            logger=logger,
        )

    if args.atom_mode:
        conditions = ("atomic", "full") if args.atom_prompt == "both" else (args.atom_prompt,)
        total_episodes = sum(len(plan) for plan in atom_plans.values()) * len(conditions)
    else:
        total_episodes = sum(len(plan) for plan in task_plans.values())
    completed_episodes = 0
    overall_results: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    eval_start = time.monotonic()
    consecutive_rollout_errors = 0
    max_consecutive_rollout_errors = 10

    if args.atom_mode:
        try:
            runtime.launch()
            if recorder is not None:
                recorder.attach(runtime.scene)
            for task_index, task_name in enumerate(task_names, start=1):
                logger.info(
                    "========== Atom task %d/%d | %s ==========", task_index, len(task_names), task_name
                )
                for entry in atom_plans[task_name]:
                    segment_raw_len = entry["raw_end_frame"] - entry["raw_start_frame"] + 1
                    # policy 一步 ≈ demo 三帧（训练数据 action_stride=3），故按 len/3 折算再留余量。
                    atom_max_steps = max(
                        int(args.atom_min_max_steps),
                        int(np.ceil(segment_raw_len / 3.0 * float(args.atom_max_steps_factor))),
                    )
                    for condition in conditions:
                        completed_episodes += 1
                        logger.info(
                            "[%d/%d] task=%s var=%d ep=%d seg=%d action=%s condition=%s "
                            "raw_frames=[%d,%d] atom_max_steps=%d",
                            completed_episodes,
                            total_episodes,
                            task_name,
                            entry["variation"],
                            entry["local_episode"],
                            entry["segment_index"],
                            entry["action"],
                            condition,
                            entry["raw_start_frame"],
                            entry["raw_end_frame"],
                            atom_max_steps,
                        )
                        result = rollout_atom_episode(
                            policy,
                            runtime,
                            entry=entry,
                            prompt_condition=condition,
                            max_steps=atom_max_steps,
                            replan_steps=replan_steps,
                            gripper_threshold=gripper_threshold,
                            action_space=action_space,
                            recorder=recorder,
                            logger=logger,
                        )
                        result["atom_max_steps"] = atom_max_steps
                        result["segment_raw_len"] = segment_raw_len
                        overall_results.append(result)
                        logger.info(
                            "Atom result | task=%s seg=%d action=%s condition=%s valid=%d "
                            "task_success=%d steps=%d/%d prefix_frames=%d error=%s video=%s prompt=%s",
                            task_name,
                            result["segment_index"],
                            result["action"],
                            result["prompt_condition"],
                            int(result["valid"]),
                            int(result["success"]),
                            result["steps"],
                            atom_max_steps,
                            result["prefix_frames"],
                            result["error_type"],
                            result["video_path"],
                            short_prompt(result["prompt"]),
                        )
        finally:
            if recorder is not None:
                recorder.finalize()
            runtime.shutdown()

        # 人工判定清单：每行一个待判视频，判定结果由人工回填 human_success 列。
        manifest_path = log_path.with_name(log_path.stem + "_atom_manifest.json")
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "note": (
                        "Exp0 atom rollouts. RLBench has no atomic-level success detector, so "
                        "'success' below is TASK-level only. Fill 'human_success' (0/1) by "
                        "watching each video; see Exp_Design.md §3.2 step 5."
                    ),
                    "atom_cache": str(args.atom_cache),
                    "checkpoint": str(checkpoint_info["checkpoint_dir"]),
                    "rollouts": [
                        {**item, "human_success": None} for item in overall_results
                    ],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        valid_count = sum(1 for item in overall_results if item["valid"])
        logger.info(
            "Atom mode finished | rollouts=%d valid=%d invalid_prefix=%d manifest=%s",
            len(overall_results),
            valid_count,
            len(overall_results) - valid_count,
            manifest_path,
        )
        logger.info("Next: watch the videos and fill human_success in %s", manifest_path)
        return 0

    if args.predict_diag:
        import numpy as _np

        all_records: list[dict[str, Any]] = []
        try:
            runtime.launch()
            for task_index, task_name in enumerate(task_names, start=1):
                logger.info(
                    "===== Predict-diag task %d/%d | %s =====", task_index, len(task_names), task_name
                )
                for (variation, local_episode) in task_plans[task_name]:
                    ep = rollout_predict_diag_episode(
                        policy,
                        runtime,
                        task_name=task_name,
                        variation=variation,
                        local_episode=local_episode,
                        gripper_threshold=gripper_threshold,
                        action_stride=3,
                    )
                    for r in ep["records"]:
                        r["task"] = task_name
                        all_records.append(r)
        finally:
            runtime.shutdown()

        # 聚合: 逐任务 + 抓取/释放帧 vs 普通帧 的预测误差。
        def agg(recs: list[dict[str, Any]]) -> dict[str, float]:
            if not recs:
                return {"n": 0}
            return {
                "n": len(recs),
                "xyz_err": float(_np.mean([r["xyz_err"] for r in recs])),
                "rot6d_err": float(_np.mean([r["rot6d_err"] for r in recs])),
                "grip_err": float(_np.mean([r["grip_err"] for r in recs])),
                "xyz_err_rel": float(
                    _np.mean([r["xyz_err"] for r in recs]) / (_np.mean([r["gt_xyz_mag"] for r in recs]) + 1e-6)
                ),
            }

        per_task = {}
        for task_name in task_names:
            recs = [r for r in all_records if r["task"] == task_name]
            per_task[task_name] = {
                "all": agg(recs),
                "grip_transition": agg([r for r in recs if r["is_grip_transition"]]),
                "non_transition": agg([r for r in recs if not r["is_grip_transition"]]),
            }
        summary = {
            "checkpoint_dir": str(checkpoint_info["checkpoint_dir"]),
            "episodes_per_task": int(args.predict_diag_episodes),
            "note": (
                "Teacher-forced 1-step prediction error (arm kept on demo trajectory). "
                "xyz_err in meters, rot6d_err L2 on 6D rotation, grip_err on gripper scalar. "
                "grip_transition = frames where gripper opens/closes (grasp/release moments)."
            ),
            "per_task": per_task,
        }
        summary_path = args.summary_json or (log_path.with_name(log_path.stem + "_predict_diag.json"))
        with Path(summary_path).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=True)
        logger.info("Predict-diag summary written: %s", summary_path)
        for task_name in task_names:
            a = per_task[task_name]["all"]
            g = per_task[task_name]["grip_transition"]
            logger.info(
                "predict-diag | %-28s all: xyz=%.4f rot6d=%.4f grip=%.3f (n=%d) | grasp-frames: xyz=%.4f rot6d=%.4f grip=%.3f (n=%d)",
                task_name,
                a.get("xyz_err", float("nan")),
                a.get("rot6d_err", float("nan")),
                a.get("grip_err", float("nan")),
                a.get("n", 0),
                g.get("xyz_err", float("nan")),
                g.get("rot6d_err", float("nan")),
                g.get("grip_err", float("nan")),
                g.get("n", 0),
            )
        return 0

    eval_env_broken: EvalEnvironmentBroken | None = None
    try:
        runtime.launch()
        if recorder is not None:
            recorder.attach(runtime.scene)
        for task_index, task_name in enumerate(task_names, start=1):
            task_results: list[dict[str, Any]] = []
            logger.info("========== Task %d/%d | %s ==========", task_index, len(task_names), task_name)
            for episode_index, (variation, local_episode) in enumerate(task_plans[task_name], start=1):
                completed_episodes += 1
                logger.info(
                    "[%d/%d] task=%s episode=%d/%d variation=%d demo_episode=%d",
                    completed_episodes,
                    total_episodes,
                    task_name,
                    episode_index,
                    len(task_plans[task_name]),
                    variation,
                    local_episode,
                )
                # 单局 rollout 包裹异常处理：RLBench 偶发的场景放置失败等不应拖垮整轮评测、
                # 丢失已完成任务的结果。可恢复异常记为 errored 局并继续；连续失败超限视为
                # 环境已损坏(如 CoppeliaSim 崩溃),提前结束但保证 summary 落盘。
                try:
                    if args.replay_keyframes:
                        result = rollout_replay_keyframes_episode(
                            runtime,
                            task_name=task_name,
                            variation=variation,
                            local_episode=local_episode,
                            gripper_threshold=gripper_threshold,
                            recorder=recorder,
                            max_seg=int(args.keyframe_max_seg),
                        )
                    elif args.replay_demo:
                        result = rollout_replay_episode(
                            runtime,
                            task_name=task_name,
                            variation=variation,
                            local_episode=local_episode,
                            gripper_threshold=gripper_threshold,
                            recorder=recorder,
                            use_delta=replay_use_delta,
                            action_stride=3,  # 训练数据 convert --action-stride 3,delta 真值须同 stride
                        )
                    elif args.waypoint_eval:
                        result = rollout_waypoint_episode(
                            policy,
                            runtime,
                            task_name=task_name,
                            variation=variation,
                            local_episode=local_episode,
                            gripper_threshold=gripper_threshold,
                            max_waypoints=int(args.waypoint_max_waypoints),
                            recorder=recorder,
                        )
                    else:
                        result = rollout_episode(
                            policy,
                            runtime,
                            task_name=task_name,
                            variation=variation,
                            local_episode=local_episode,
                            max_steps=max_steps,
                            replan_steps=replan_steps,
                            gripper_threshold=gripper_threshold,
                            action_stride=action_stride,
                            action_space=action_space,
                            recorder=recorder,
                        )
                    consecutive_rollout_errors = 0
                except Exception as exc:  # noqa: BLE001
                    error_type = type(exc).__name__
                    consecutive_rollout_errors += 1
                    logger.warning(
                        "Rollout error (%s) task=%s var=%d ep=%d — 记为 errored 局并继续 "
                        "(consecutive=%d): %s",
                        error_type,
                        task_name,
                        variation,
                        local_episode,
                        consecutive_rollout_errors,
                        exc,
                    )
                    result = make_errored_result(task_name, variation, local_episode, error_type)
                    if consecutive_rollout_errors >= max_consecutive_rollout_errors:
                        task_results.append(result)
                        overall_results.append(result)
                        logger.error(
                            "连续 %d 局 rollout 异常，判定环境已损坏，提前结束评测(summary 仍会写出)。",
                            consecutive_rollout_errors,
                        )
                        raise EvalEnvironmentBroken(consecutive_rollout_errors) from exc
                task_results.append(result)
                overall_results.append(result)
                logger.info(
                    "Episode result | task=%s success=%d steps=%d policy_calls=%d infer_ms_mean=%.2f infer_ms_last=%.2f wall_time_s=%.2f variation=%d error=%s error_count=%d video=%s prompt=%s",
                    task_name,
                    int(result["success"]),
                    result["steps"],
                    result["policy_calls"],
                    result["infer_ms_mean"],
                    result["infer_ms_last"],
                    result["wall_time_s"],
                    result["variation"],
                    result["error_type"],
                    result["error_count"],
                    result["video_path"],
                    short_prompt(result["prompt"]),
                )

            task_successes = sum(int(item["success"]) for item in task_results)
            task_success_rate = task_successes / max(1, len(task_results))
            avg_steps = float(np.mean([item["steps"] for item in task_results]))
            avg_infer_ms = safe_nanmean([item["infer_ms_mean"] for item in task_results])
            task_summary = {
                "task": task_name,
                "success_rate": task_success_rate,
                "successes": task_successes,
                "episodes": len(task_results),
                "avg_steps": avg_steps,
                "avg_infer_ms": avg_infer_ms,
            }
            task_summaries.append(task_summary)
            logger.info(
                "Task summary | task=%s success_rate=%.4f successes=%d/%d avg_steps=%.2f avg_infer_ms=%.2f",
                task_summary["task"],
                task_summary["success_rate"],
                task_summary["successes"],
                task_summary["episodes"],
                task_summary["avg_steps"],
                task_summary["avg_infer_ms"],
            )
    except EvalEnvironmentBroken as exc:
        # 环境损坏提前结束：吞掉异常，落到下方 summary 写出逻辑，保住已完成的结果。
        eval_env_broken = exc
        logger.error("评测因环境损坏提前结束（已完成 %d 局，summary 仍会写出）。", len(overall_results))
    finally:
        if recorder is not None:
            recorder.finalize()
        runtime.shutdown()

    if not overall_results:
        # 一局都没跑完就损坏：无可用结果，直接抛出。
        if eval_env_broken is not None:
            raise eval_env_broken
        raise RuntimeError("评测未产生任何结果，无法生成 summary。")

    # macro 只对**实际有评测结果**的任务求均值，避免提前结束时空任务产生 nan。
    evaluated_tasks = [t for t in task_names if any(item["task"] == t for item in overall_results)]
    overall_successes = sum(int(item["success"]) for item in overall_results)
    overall_success_rate = overall_successes / max(1, len(overall_results))
    macro_task_success_rate = float(
        np.mean(
            [
                np.mean([int(item["success"]) for item in overall_results if item["task"] == task_name])
                for task_name in evaluated_tasks
            ]
        )
    )
    total_wall_time_s = time.monotonic() - eval_start
    logger.info("========== Final Summary ==========")
    for task_summary in task_summaries:
        logger.info(
            "Final task summary | task=%s success_rate=%.4f successes=%d/%d avg_steps=%.2f avg_infer_ms=%.2f",
            task_summary["task"],
            task_summary["success_rate"],
            task_summary["successes"],
            task_summary["episodes"],
            task_summary["avg_steps"],
            task_summary["avg_infer_ms"],
        )
    logger.info("Overall success rate (micro): %.4f", overall_success_rate)
    logger.info("Overall success rate (macro): %.4f", macro_task_success_rate)
    logger.info("Overall successes: %d/%d", overall_successes, len(overall_results))
    logger.info("Total wall time (s): %.2f", total_wall_time_s)
    errored_episodes = sum(1 for item in overall_results if item.get("errored"))
    if errored_episodes:
        logger.warning("Errored episodes (计入失败分母): %d", errored_episodes)
    if args.summary_json is not None:
        summary_payload = {
            "checkpoint_source": checkpoint_source,
            "checkpoint_dir": str(checkpoint_info["checkpoint_dir"]) if checkpoint_info else None,
            "success_rate_micro": overall_success_rate,
            "success_rate_macro": macro_task_success_rate,
            "successes": overall_successes,
            "episodes": len(overall_results),
            "errored_episodes": errored_episodes,
            "partial_env_broken": eval_env_broken is not None,
            "evaluated_tasks": evaluated_tasks,
            "total_wall_time_s": total_wall_time_s,
            "log_path": str(log_path),
            "task_summaries": task_summaries,
            "episode_results": [
                {key: value for key, value in item.items() if key != "video_path"} for item in overall_results
            ],
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, ensure_ascii=True)
        logger.info("Summary JSON written: %s", args.summary_json)
    if recorder is not None:
        summary_path = recorder.write_summary(
            header=f"Eval run: {run_name} | checkpoint={checkpoint_source} | log={log_path}"
        )
        logger.info("Recording summary written: %s", summary_path)
    logger.info("Evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
