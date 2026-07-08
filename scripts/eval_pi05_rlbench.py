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


"""解析评测参数；核心配置放 YAML，命令行只保留少量高频覆盖项。"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pi0.5 on RLBench.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--which",
        choices=("best", "latest", "official"),
        default=None,
        help="Override checkpoint.which in YAML.",
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
    return parser.parse_args()


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


"""统一解析 official / best / latest 三种权重来源。"""
def resolve_checkpoint_source(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_cfg = config["checkpoint"]
    source = checkpoint_cfg["which"]
    if source in ("best", "latest"):
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
                step_dir / "assets" / "train",
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


"""优先从 demo.misc 里读 variation_index；缺失时再退回调度器给的 variation。"""
def resolve_demo_variation(demo, fallback_variation: int) -> int:
    if len(demo) <= 0:
        raise ValueError("Demo has no observations.")
    misc = getattr(demo[0], "misc", None)
    if isinstance(misc, dict) and "variation_index" in misc:
        return int(misc["variation_index"])
    return int(fallback_variation)


class RLBenchEvalRuntime:
    """薄封装：管理 RLBench 环境、任务切换和 demo replay reset。"""

    def __init__(self, dataset_root: Path, image_size: int, headless: bool):
        try:
            from pyrep.const import RenderMode
            from pyrep.errors import ConfigurationPathError, IKError
            from rlbench import CameraConfig, Environment, ObservationConfig
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.backend.exceptions import InvalidActionError
            from rlbench.backend.utils import task_file_to_task_class
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise ImportError("RLBench and PyRep runtime dependencies are required for evaluation.") from exc

        class ClippedEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):
            """Clamp target XYZ into the RLBench workspace before planning."""

            def action(self, scene, action, *args, **kwargs):
                action = np.asarray(action, dtype=np.float32).copy()
                action[:3] = np.clip(
                    action[:3],
                    np.asarray([scene._workspace_minx, scene._workspace_miny, scene._workspace_minz], dtype=np.float32)
                    + 1e-7,
                    np.asarray([scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz], dtype=np.float32)
                    - 1e-7,
                )
                super().action(scene, action)

        disabled_camera = CameraConfig()
        disabled_camera.set_all(False)
        active_camera = CameraConfig(
            rgb=True,
            point_cloud=False,
            mask=False,
            depth=False,
            image_size=[image_size, image_size],
            render_mode=RenderMode.OPENGL,
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
            action_mode=MoveArmThenGripper(ClippedEndEffectorPoseViaPlanning(), Discrete()),
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

    def reset_to_demo(self, task_name: str, variation: int, local_episode: int) -> tuple[Any, str, int]:
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
        self._task.set_variation(demo_variation)
        descriptions, obs = self._task.reset_to_demo(demo)
        prompt = descriptions[0] if descriptions else self._task.get_task_descriptions()[0]
        return obs, prompt, demo_variation

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
    recorder: EpisodeVideoRecorder | None = None,
) -> dict[str, Any]:
    obs, prompt, used_variation = runtime.reset_to_demo(task_name, variation, local_episode)
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
            for action10 in action_chunk[: min(replan_steps, action_chunk.shape[0])]:
                action_queue.append(np.asarray(action10, dtype=np.float32))
            infer_ms = policy_output.get("policy_timing", {}).get("infer_ms")
            if infer_ms is not None:
                infer_ms_values.append(float(infer_ms))
            policy_calls += 1

        action8 = policy_action10_to_rlbench_action8(action_queue.popleft(), gripper_threshold)
        steps_taken = step_index + 1
        try:
            obs, reward, terminal = runtime.step(action8)
        except runtime.step_exceptions as exc:
            reward = 0.0
            terminal = True
            error_type = type(exc).__name__
        else:
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
    }


"""主流程：解析配置、加载 policy、逐任务评测并写单日志汇总。"""
def main() -> int:
    args = parse_args()
    config = apply_cli_overrides(load_yaml_config(args.config), args)
    checkpoint_source = config["checkpoint"]["which"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_root = resolve_path_like(config["logging"]["log_root"])
    assert isinstance(log_root, Path)
    run_name = f"{config['logging']['log_name_prefix']}_{checkpoint_source}_{timestamp}"
    log_path = log_root / f"{run_name}.log"
    logger = init_logging(log_path)

    set_random_seed(int(config["runtime"]["seed"]))
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

    start_episode = int(config["dataset"]["start_episode"])
    episodes_per_task = int(config["dataset"]["episodes_per_task"])
    max_steps = int(config["environment"]["max_steps"])
    replan_steps = int(config["environment"]["replan_steps"])
    gripper_threshold = float(config["environment"]["gripper_threshold"])
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

    task_plans: dict[str, list[tuple[int, int]]] = {}
    for task_name in task_names:
        task_root = split_root / task_name
        if not task_root.is_dir():
            raise FileNotFoundError(f"Task raw data directory does not exist: {task_root}")
        task_plans[task_name] = select_demo_subset(task_root, start_episode, episodes_per_task)

    logger.info("bash> %s", " ".join(sys.argv))
    logger.info("Log file: %s", log_path)
    logger.info("Checkpoint source: %s", checkpoint_source)
    logger.info("Resolved checkpoint dir: %s", checkpoint_info["checkpoint_dir"])
    if checkpoint_info["index_path"] is not None:
        logger.info("Resolved checkpoint index: %s", checkpoint_info["index_path"])
    logger.info("Norm stats source: %s", norm_stats_source)
    logger.info("Runtime device: %s", device)
    logger.info(
        "Eval config | split=%s variation_mode=%s tasks=%d episodes_per_task=%d start_episode=%d max_steps=%d replan_steps=%d",
        split,
        variation_mode,
        len(task_names),
        episodes_per_task,
        start_episode,
        max_steps,
        replan_steps,
    )
    if recording_enabled:
        logger.info(
            "Recording config | videos_per_task=%s (success/fail each) video_root=%s",
            recording_cfg.get("videos_per_task", 1),
            video_root,
        )
    else:
        logger.info("Recording disabled.")
    for task_name in task_names:
        logger.info("Task plan | task=%s episodes=%d first_entries=%s", task_name, len(task_plans[task_name]), task_plans[task_name][:5])

    if args.dry_run:
        logger.info("Dry run finished. Config and checkpoint resolution succeeded.")
        return 0

    norm_stats = load_norm_stats(norm_stats_source)
    policy = create_policy(train_config, checkpoint_info["checkpoint_dir"], norm_stats, device)
    sanitize_rlbench_gui_env(logger)
    install_numpy_pickle_compat(logger)

    runtime = RLBenchEvalRuntime(
        dataset_root=split_root,
        image_size=int(config["environment"]["image_size"]),
        headless=bool(config["environment"]["headless"]),
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

    total_episodes = sum(len(plan) for plan in task_plans.values())
    completed_episodes = 0
    overall_results: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    eval_start = time.monotonic()

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
                result = rollout_episode(
                    policy,
                    runtime,
                    task_name=task_name,
                    variation=variation,
                    local_episode=local_episode,
                    max_steps=max_steps,
                    replan_steps=replan_steps,
                    gripper_threshold=gripper_threshold,
                    recorder=recorder,
                )
                task_results.append(result)
                overall_results.append(result)
                logger.info(
                    "Episode result | task=%s success=%d steps=%d policy_calls=%d infer_ms_mean=%.2f infer_ms_last=%.2f wall_time_s=%.2f variation=%d error=%s video=%s prompt=%s",
                    task_name,
                    int(result["success"]),
                    result["steps"],
                    result["policy_calls"],
                    result["infer_ms_mean"],
                    result["infer_ms_last"],
                    result["wall_time_s"],
                    result["variation"],
                    result["error_type"],
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
    finally:
        if recorder is not None:
            recorder.finalize()
        runtime.shutdown()

    overall_successes = sum(int(item["success"]) for item in overall_results)
    overall_success_rate = overall_successes / max(1, len(overall_results))
    macro_task_success_rate = float(
        np.mean(
            [
                np.mean([int(item["success"]) for item in overall_results if item["task"] == task_name])
                for task_name in task_names
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
    if recorder is not None:
        summary_path = recorder.write_summary(
            header=f"Eval run: {run_name} | checkpoint={checkpoint_source} | log={log_path}"
        )
        logger.info("Recording summary written: %s", summary_path)
    logger.info("Evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
