#!/usr/bin/env python3
"""Build offline QwenVL subtask segmentation cache for VQAP + pi0.5 M1 training."""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
import types
from typing import Any
import warnings


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RAW_ROOT = REPO_ROOT / "RLBench_Raw_Dataset" / "train"
DEFAULT_LEROBOT_ROOT = REPO_ROOT / "LeRobot_RLBench_Dataset" / "train_delta"
DEFAULT_PHASE_LABEL = REPO_ROOT / "Phase_Action_Label.csv"
DEFAULT_OUTPUT = REPO_ROOT / "planner_cache" / "train_delta" / "qwenvl_segments.json"
DEFAULT_RLBENCH_ROOT = REPO_ROOT / "source" / "RLBench"
TGS_ROOT = REPO_ROOT / "traj_generator_segmentation"
LOW_DIM_PICKLE = "low_dim_obs.pkl"
VARIATION_DESCRIPTIONS = "variation_descriptions.pkl"
CONVERSION_PROGRESS = ".conversion_progress.jsonl"
EPISODES_DIR = "episodes"

DEFAULT_CAMERAS = (
    "front_rgb",
    "wrist_rgb",
    "left_shoulder_rgb",
    "right_shoulder_rgb",
    "overhead_rgb",
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ACTION_WHITELIST = {
    "approach",
    "grasp",
    "lift",
    "transfer",
    "place",
    "push",
    "pull",
    "press",
    "rotate",
    "slide",
    "insert",
    "hang",
    "wipe",
    "flip-open",
    "flip-close",
    "revolve-in",
    "revolve-out",
    "pose-adjust",
}


@dataclass(frozen=True)
class EpisodeSpec:
    """描述一个待分割的 raw RLBench episode。"""

    task_name: str
    variation: str
    episode: str
    raw_episode_key: str
    episode_dir: Path
    task_instruction: str
    phase_prior: list[str]
    output_episode_index: int | None
    lerobot_frames: int | None


@dataclass(frozen=True)
class ImageInput:
    """记录传给 QwenVL 的单张图片及其语义标签。"""

    frame: int
    camera: str
    path: Path


class CompatibleUnpickler(pickle.Unpickler):
    """兼容旧 pickle 里 numpy 模块路径变化。"""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core":
            module = "numpy.core"
        elif module.startswith("numpy._core."):
            module = module.replace("numpy._core.", "numpy.core.", 1)
        return super().find_class(module, name)


# 解析命令行参数；默认路径全部指向当前 M1 delta 主线。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline QwenVL planner-cache for VQAP + pi0.5 M1 training."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--phase-label", type=Path, default=DEFAULT_PHASE_LABEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--rlbench-root", type=Path, default=DEFAULT_RLBENCH_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--variations", nargs="*", default=None)
    parser.add_argument("--episode-keys", nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL"))
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--max-sampled-frames", type=int, default=12)
    parser.add_argument("--keyframe-window", type=int, default=2)
    parser.add_argument(
        "--action-stride",
        type=int,
        default=None,
        help=(
            "Raw-to-LeRobot temporal stride. If omitted, infer it per episode from "
            "the conversion progress frame count."
        ),
    )
    parser.add_argument("--image-max-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--min-phase-len", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-review",
        action="store_true",
        help="When --resume is used, reprocess existing records marked needs_review or empty.",
    )
    parser.add_argument("--no-keyframes", action="store_true")
    args = parser.parse_args()

    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be >= 1.")
    if args.max_episodes_per_task is not None and args.max_episodes_per_task < 1:
        raise ValueError("--max-episodes-per-task must be >= 1.")
    if args.max_sampled_frames < 2:
        raise ValueError("--max-sampled-frames must be >= 2.")
    if args.action_stride is not None and args.action_stride < 1:
        raise ValueError("--action-stride must be >= 1.")
    if args.keyframe_window < 0:
        raise ValueError("--keyframe-window must be >= 0.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0.")
    if args.retry_review and not args.resume:
        raise ValueError("--retry-review requires --resume.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    return args


# 返回稳定 UTC 时间，写入 cache 元信息。
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 将 variation 参数统一成 variation0 这种目录名。
def normalize_variation(value: str) -> str:
    value = value.strip()
    if value.startswith("variation"):
        return value
    if value.isdigit():
        return f"variation{int(value)}"
    return value


# 解析 Phase_Action_Label.csv；空 task 单元沿用上一行 task。
def load_phase_labels(path: Path) -> dict[tuple[str, str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Phase label CSV not found: {path}")

    labels: dict[tuple[str, str], list[str]] = {}
    current_task = ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        phase_columns = sorted(
            [name for name in reader.fieldnames if name.startswith("Phase")],
            key=lambda name: int(name.removeprefix("Phase")),
        )
        for row in reader:
            task = (row.get("Task") or "").strip()
            if task:
                current_task = task
            if not current_task:
                continue
            variation_raw = (row.get("Variation") or "").strip()
            if not variation_raw:
                continue
            variation = normalize_variation(variation_raw)
            phases = [
                (row.get(column) or "").strip()
                for column in phase_columns
                if (row.get(column) or "").strip()
            ]
            bad_actions = [action for action in phases if action not in ACTION_WHITELIST]
            if bad_actions:
                raise ValueError(
                    f"{path} contains unsupported actions for {current_task}/{variation}: "
                    f"{bad_actions}"
                )
            labels[(current_task, variation)] = phases
    return labels


# 从 CSV 中取当前 task/variation 的阶段先验；若具体 variation 缺失，回退同 task 第一条。
def resolve_phase_prior(
    labels: dict[tuple[str, str], list[str]],
    task_name: str,
    variation: str,
) -> list[str]:
    direct = labels.get((task_name, variation))
    if direct:
        return list(direct)
    for (task, _variation), phases in labels.items():
        if task == task_name:
            return list(phases)
    raise KeyError(f"No phase prior found for task={task_name}, variation={variation}.")


# 读取 conversion progress，用 raw episode key 对齐 LeRobot episode_index。
def load_conversion_records(lerobot_root: Path) -> dict[str, dict[str, Any]]:
    progress_path = lerobot_root / CONVERSION_PROGRESS
    if not progress_path.is_file():
        raise FileNotFoundError(f"Conversion progress log not found: {progress_path}")
    records: dict[str, dict[str, Any]] = {}
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_key = record.get("raw_episode_key")
            if raw_key:
                records[str(raw_key)] = record
    return records


# 读取 variation_descriptions.pkl，并复用 M0 的 longest prompt 策略。
def pick_longest_instruction(variation_dir: Path) -> str:
    desc_path = variation_dir / VARIATION_DESCRIPTIONS
    if not desc_path.is_file():
        raise FileNotFoundError(f"variation descriptions not found: {desc_path}")
    with desc_path.open("rb") as handle:
        descriptions = pickle.load(handle)
    if not isinstance(descriptions, list):
        raise TypeError(f"{desc_path} should contain a list of strings.")
    cleaned = [text.strip() for text in descriptions if isinstance(text, str) and text.strip()]
    if not cleaned:
        raise ValueError(f"{desc_path} contains no valid instruction.")
    return max(cleaned, key=len)


# 根据 raw 数据、CSV 阶段先验、LeRobot 映射收集待处理 episode。
def collect_episode_specs(
    *,
    raw_root: Path,
    phase_labels: dict[tuple[str, str], list[str]],
    conversion_records: dict[str, dict[str, Any]],
    tasks: list[str] | None,
    variations: list[str] | None,
    episode_keys: list[str] | None,
    max_episodes: int | None,
    max_episodes_per_task: int | None,
) -> list[EpisodeSpec]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    requested_tasks = set(tasks or [])
    requested_variations = {normalize_variation(item) for item in (variations or [])}
    requested_keys = set(episode_keys or [])

    task_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    specs: list[EpisodeSpec] = []
    per_task_count: dict[str, int] = {}

    for task_dir in task_dirs:
        task_name = task_dir.name
        if requested_tasks and task_name not in requested_tasks:
            continue
        variation_dirs = [
            path
            for path in task_dir.iterdir()
            if path.is_dir() and re.fullmatch(r"variation\d+", path.name)
        ]
        for variation_dir in sorted(variation_dirs, key=path_numeric_suffix):
            variation = variation_dir.name
            if requested_variations and variation not in requested_variations:
                continue
            episodes_root = variation_dir / EPISODES_DIR
            if not episodes_root.is_dir():
                continue
            task_instruction = pick_longest_instruction(variation_dir)
            phase_prior = resolve_phase_prior(phase_labels, task_name, variation)
            episode_dirs = [
                path
                for path in episodes_root.iterdir()
                if path.is_dir() and re.fullmatch(r"episode\d+", path.name)
            ]
            for episode_dir in sorted(episode_dirs, key=path_numeric_suffix):
                raw_key = f"{task_name}/{variation}/{episode_dir.name}"
                if requested_keys and raw_key not in requested_keys:
                    continue
                if max_episodes_per_task is not None:
                    if per_task_count.get(task_name, 0) >= max_episodes_per_task:
                        continue
                record = conversion_records.get(raw_key)
                output_index = None if record is None else int(record["output_episode_index"])
                lerobot_frames = None if record is None else int(record["frames"])
                specs.append(
                    EpisodeSpec(
                        task_name=task_name,
                        variation=variation,
                        episode=episode_dir.name,
                        raw_episode_key=raw_key,
                        episode_dir=episode_dir,
                        task_instruction=task_instruction,
                        phase_prior=phase_prior,
                        output_episode_index=output_index,
                        lerobot_frames=lerobot_frames,
                    )
                )
                per_task_count[task_name] = per_task_count.get(task_name, 0) + 1
                if max_episodes is not None and len(specs) >= max_episodes:
                    return specs
    if requested_keys:
        found = {spec.raw_episode_key for spec in specs}
        missing = sorted(requested_keys - found)
        if missing:
            raise FileNotFoundError(f"Requested episode keys not found: {missing}")
    return specs


# 用目录名中的数字后缀排序，例如 variation2 在 variation10 前面。
def path_numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 0


# 安装最小 rlbench 模块，保证 low_dim_obs.pkl 能反序列化 Observation。
def install_rlbench_pickle_support(rlbench_root: Path) -> None:
    package_root = rlbench_root / "rlbench"
    backend_root = package_root / "backend"
    observation_py = backend_root / "observation.py"
    if not observation_py.is_file():
        raise FileNotFoundError(f"Observation definition not found: {observation_py}")

    rlbench_module = sys.modules.get("rlbench")
    if rlbench_module is None:
        rlbench_module = types.ModuleType("rlbench")
        rlbench_module.__path__ = [str(package_root)]
        sys.modules["rlbench"] = rlbench_module

    backend_module = sys.modules.get("rlbench.backend")
    if backend_module is None:
        backend_module = types.ModuleType("rlbench.backend")
        backend_module.__path__ = [str(backend_root)]
        sys.modules["rlbench.backend"] = backend_module

    if "rlbench.backend.observation" not in sys.modules:
        spec = importlib.util.spec_from_file_location("rlbench.backend.observation", observation_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {observation_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["rlbench.backend.observation"] = module
        spec.loader.exec_module(module)


# 加载一个 raw episode 的低维轨迹。
def load_demo(episode_dir: Path, rlbench_root: Path) -> list[Any]:
    install_rlbench_pickle_support(rlbench_root)
    low_dim_path = episode_dir / LOW_DIM_PICKLE
    if not low_dim_path.is_file():
        raise FileNotFoundError(f"low_dim_obs.pkl not found: {low_dim_path}")
    with low_dim_path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            demo = CompatibleUnpickler(handle).load()
    if isinstance(demo, list):
        return demo
    if hasattr(demo, "__len__") and hasattr(demo, "__getitem__"):
        return [demo[index] for index in range(len(demo))]
    raise TypeError(f"{low_dim_path} should contain a trajectory sequence, got {type(demo)!r}.")


# 用旧 traj_generator_segmentation 的低维信号生成候选关键帧。
def extract_keyframe_candidates(
    demo: list[Any],
    *,
    enabled: bool,
    min_phase_len: int | None,
) -> tuple[list[int], dict[str, Any], list[str]]:
    if not enabled:
        return [], {"disabled": True}, []
    try:
        config_module, keyframe_module = load_traj_segmentation_modules()

        default_phase_len = int(getattr(config_module, "RUN_MIN_PHASE_LEN"))
        phase_len = default_phase_len if min_phase_len is None else min_phase_len
        keyframes, debug_info, _num_phases = keyframe_module.extract_keyframes(
            demo, min_phase_len=phase_len
        )
        keyframes = sorted({int(frame) for frame in keyframes if 0 <= int(frame) < len(demo)})
        return keyframes, debug_info, []
    except Exception as exc:  # noqa: BLE001 - 关键帧只是候选，失败时可降级。
        return [], {"error": repr(exc)}, [f"keyframe_extraction_failed: {exc}"]


# 只加载关键帧分割所需子模块，避开 traj_generator_segmentation.__init__ 的采集依赖。
def load_traj_segmentation_modules() -> tuple[Any, Any]:
    package_name = "traj_generator_segmentation"
    package = sys.modules.get(package_name)
    if package is None or not hasattr(package, "__path__"):
        package = types.ModuleType(package_name)
        package.__path__ = [str(TGS_ROOT)]
        sys.modules[package_name] = package

    config_module = load_module_from_path(
        f"{package_name}.config",
        TGS_ROOT / "config.py",
    )
    load_module_from_path(f"{package_name}.thresholds", TGS_ROOT / "thresholds.py")
    load_module_from_path(f"{package_name}.signals", TGS_ROOT / "signals.py")
    load_module_from_path(f"{package_name}.interaction", TGS_ROOT / "interaction.py")
    keyframe_module = load_module_from_path(
        f"{package_name}.keyframe",
        TGS_ROOT / "keyframe.py",
    )
    return config_module, keyframe_module


# 通过文件路径加载模块，并注册到 sys.modules 以支持相对 import。
def load_module_from_path(module_name: str, path: Path) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 按转换脚本的 action_stride 规则重建 raw frame 到 LeRobot frame 的映射。
def build_kept_indices(num_raw_frames: int, action_stride: int) -> list[int]:
    if num_raw_frames <= 0:
        return []
    kept = list(range(0, num_raw_frames, action_stride))
    if kept[-1] != num_raw_frames - 1:
        kept.append(num_raw_frames - 1)
    return kept


# 根据 conversion progress 中的 LeRobot 帧数反推 action_stride。
def infer_action_stride(num_raw_frames: int, lerobot_frames: int | None) -> tuple[int, list[str]]:
    warnings_out: list[str] = []
    if lerobot_frames is None or lerobot_frames <= 0:
        warnings_out.append("action_stride_infer_failed:missing_lerobot_frames;fallback=1")
        return 1, warnings_out
    matches = [
        stride
        for stride in range(1, max(2, num_raw_frames + 1))
        if len(build_kept_indices(num_raw_frames, stride)) - 1 == lerobot_frames
    ]
    if not matches:
        warnings_out.append(
            f"action_stride_infer_failed:raw_frames={num_raw_frames},"
            f"lerobot_frames={lerobot_frames};fallback=1"
        )
        return 1, warnings_out
    if len(matches) > 1:
        warnings_out.append(f"action_stride_infer_ambiguous:matches={matches};use={matches[0]}")
    return int(matches[0]), warnings_out


# 将 raw 边界映射到 LeRobot frame index；end 取不超过 raw_end 的最后一个 kept frame。
def raw_to_lerobot_frame(raw_frame: int, kept_indices: list[int], *, is_end: bool) -> int:
    if len(kept_indices) <= 1:
        return 0
    usable = kept_indices[:-1]
    raw_frame = int(max(0, min(raw_frame, kept_indices[-1])))
    if is_end:
        idx = 0
        for i, kept in enumerate(usable):
            if kept <= raw_frame:
                idx = i
            else:
                break
        return idx
    for i, kept in enumerate(usable):
        if kept >= raw_frame:
            return i
    return len(usable) - 1


# 在关键帧候选、首尾帧和均匀采样之间做平衡，控制 API 图片数量。
def select_sample_frames(
    *,
    num_raw_frames: int,
    keyframes: list[int],
    max_sampled_frames: int,
    keyframe_window: int,
) -> list[int]:
    if num_raw_frames <= 0:
        return []
    endpoints = {0, num_raw_frames - 1}
    exact = sorted(endpoints | {clip_frame(kf, num_raw_frames) for kf in keyframes})
    if len(exact) >= max_sampled_frames:
        return balanced_reduce(exact, max_sampled_frames)

    selected = set(exact)
    for keyframe in keyframes:
        for offset in range(-keyframe_window, keyframe_window + 1):
            selected.add(clip_frame(keyframe + offset, num_raw_frames))
            if len(selected) >= max_sampled_frames:
                break
        if len(selected) >= max_sampled_frames:
            break

    if len(selected) < max_sampled_frames:
        for frame in uniform_frames(num_raw_frames, max_sampled_frames):
            selected.add(frame)
            if len(selected) >= max_sampled_frames:
                break
    return balanced_reduce(sorted(selected), max_sampled_frames)


# 将帧号限制在 episode 合法范围。
def clip_frame(frame: int, num_raw_frames: int) -> int:
    return int(max(0, min(frame, num_raw_frames - 1)))


# 均匀采样若干帧，补足关键帧候选漏检时的视觉覆盖。
def uniform_frames(num_raw_frames: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    if num_raw_frames <= 1:
        return [0]
    return sorted({int(round(i * (num_raw_frames - 1) / (count - 1))) for i in range(count)})


# 当候选过多时，保留覆盖全时段的代表帧。
def balanced_reduce(frames: list[int], limit: int) -> list[int]:
    unique = sorted(set(frames))
    if len(unique) <= limit:
        return unique
    if limit <= 2:
        return [unique[0], unique[-1]][:limit]
    positions = [round(i * (len(unique) - 1) / (limit - 1)) for i in range(limit)]
    return sorted({unique[int(pos)] for pos in positions})


# 收集每个采样帧的多视角图片路径，缺失视角只记录 warning。
def collect_image_inputs(
    *,
    episode_dir: Path,
    frames: list[int],
    cameras: list[str],
) -> tuple[list[ImageInput], list[str]]:
    image_inputs: list[ImageInput] = []
    warnings: list[str] = []
    for frame in frames:
        for camera in cameras:
            image_path = episode_dir / camera / f"{frame}.png"
            if image_path.is_file():
                image_inputs.append(ImageInput(frame=frame, camera=camera, path=image_path))
            else:
                warnings.append(f"missing_image:{camera}/{frame}.png")
    return image_inputs, warnings


# 构造系统提示词：CSV 阶段是主约束，视觉模型只估边界和子指令。
def build_system_prompt() -> str:
    allowed = ", ".join(sorted(ACTION_WHITELIST))
    return (
        "You are a robot trajectory segmenter for RLBench demonstrations.\n"
        "Your job is to split one complete robot trajectory into ordered semantic subtasks.\n"
        "The provided phase_prior from CSV is the main source of action stages. "
        "Use visual evidence and keyframe candidates mainly to estimate frame boundaries and "
        "write phase-specific instructions.\n"
        f"Allowed action labels: {allowed}.\n"
        "Do not invent action labels outside the allowed list. Do not remove phase_prior actions "
        "unless the input explicitly says they are optional.\n"
        "pose-adjust is allowed only as an extra planning/segmentation label; if used, set "
        "code_policy to map_to_neighbor or always_off.\n"
        "The output must be strict JSON only. Do not wrap it in markdown. Do not provide hidden "
        "chain-of-thought; use at most short boundary_reason strings."
    )


# 构造用户提示词：包含 longest 任务指令、CSV 阶段、关键帧候选和输出 schema。
def build_user_prompt(
    *,
    spec: EpisodeSpec,
    num_raw_frames: int,
    sampled_frames: list[int],
    keyframes: list[int],
    cameras: list[str],
) -> str:
    phase_json = json.dumps(spec.phase_prior, ensure_ascii=False)
    sampled_json = json.dumps(sampled_frames, ensure_ascii=False)
    keyframe_json = json.dumps(keyframes, ensure_ascii=False)
    cameras_json = json.dumps(cameras, ensure_ascii=False)
    expected_steps = len(spec.phase_prior)
    return f"""
Segment this RLBench robot demonstration.

Episode metadata:
- task_name: {spec.task_name}
- variation: {spec.variation}
- episode: {spec.episode}
- raw_episode_key: {spec.raw_episode_key}
- output_episode_index: {spec.output_episode_index}
- longest_task_instruction: {spec.task_instruction}
- total_raw_frames: {num_raw_frames}
- total_lerobot_frames: {spec.lerobot_frames}
- phase_prior_from_csv: {phase_json}
- expected_number_of_csv_phases: {expected_steps}
- keyframe_candidates_from_low_dim: {keyframe_json}
- sampled_frames_with_images: {sampled_json}
- cameras_per_sampled_frame: {cameras_json}

Important rules:
1. The phase_prior_from_csv is the primary action sequence. Output one segment per CSV phase by default.
2. Use the images and keyframe candidates to estimate raw_start_frame/raw_end_frame boundaries.
3. Segment boundaries must be contiguous, non-overlapping, and cover frame 0 through frame {num_raw_frames - 1}.
4. raw_start_frame and raw_end_frame are inclusive raw RLBench frame indices.
5. The first segment must start at 0. The last segment must end at {num_raw_frames - 1}.
6. action must match the CSV phase at the same order, unless you insert an extra pose-adjust segment.
7. If pose-adjust is inserted, code_policy must be map_to_neighbor or always_off.
8. Each instruction must be an English subtask instruction derived from the longest_task_instruction.
9. Keep instruction short but specific, for example "approach the drawer handle".
10. confidence must be a number between 0 and 1.
11. If keyframe_candidates look inaccurate, rely on phase_prior plus visual temporal order and set needs_review=true only when boundaries remain uncertain.

Return exactly this JSON object:
{{
  "needs_review": false,
  "review_reasons": [],
  "segments": [
    {{
      "action": "approach",
      "instruction": "approach the target object",
      "raw_start_frame": 0,
      "raw_end_frame": 10,
      "code_policy": "normal",
      "confidence": 0.8,
      "boundary_reason": "short visible evidence only"
    }}
  ]
}}
""".strip()


# dry-run 使用：把 prompt、图片路径和关键帧候选落盘，便于人工检查。
def write_prompt_debug(
    *,
    debug_dir: Path,
    spec: EpisodeSpec,
    system_prompt: str,
    user_prompt: str,
    image_inputs: list[ImageInput],
    sampled_frames: list[int],
    keyframes: list[int],
    keyframe_debug: dict[str, Any],
    warnings: list[str],
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = debug_stem(spec)
    path = debug_dir / f"{stem}_prompt.json"
    payload = {
        "raw_episode_key": spec.raw_episode_key,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "sampled_frames": sampled_frames,
        "keyframe_candidates": keyframes,
        "image_inputs": [
            {"frame": item.frame, "camera": item.camera, "path": str(item.path)}
            for item in image_inputs
        ],
        "keyframe_debug": make_json_safe(keyframe_debug),
        "warnings": warnings,
    }
    atomic_write_json(path, payload)
    return path


# 把图片压缩成 data URL，避免 API 调用依赖外部文件服务。
def image_to_data_url(path: Path, *, max_size: int, jpeg_quality: int) -> str:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))
        with tempfile.SpooledTemporaryFile() as buffer:
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# 按 OpenAI-compatible multimodal message 格式构造请求体。
def build_api_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    image_inputs: list[ImageInput],
    image_max_size: int,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    current_frame: int | None = None
    for image_input in image_inputs:
        if image_input.frame != current_frame:
            current_frame = image_input.frame
            content.append({"type": "text", "text": f"Images for raw frame {current_frame}:"})
        content.append(
            {
                "type": "text",
                "text": f"camera={image_input.camera}, frame={image_input.frame}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        image_input.path,
                        max_size=image_max_size,
                        jpeg_quality=jpeg_quality,
                    )
                },
            }
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


# 调用百炼 OpenAI 兼容接口；openai 包只在真实调用时导入。
def call_qwenvl_api(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
    return content, {"usage": usage_dict}


# 从模型回复中提取 JSON；即使模型误加 markdown fence，也尽量恢复。
def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise TypeError("API response JSON must be an object.")
    return data


# 校验 API 输出，并补齐 LeRobot frame 边界。
def normalize_api_result(
    *,
    data: dict[str, Any],
    spec: EpisodeSpec,
    num_raw_frames: int,
    kept_indices: list[int],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    hard_errors: list[str] = []
    segments_in = data.get("segments")
    if not isinstance(segments_in, list) or not segments_in:
        raise ValueError("response.segments must be a non-empty list.")

    normalized: list[dict[str, Any]] = []
    previous_end = -1
    for index, segment in enumerate(segments_in):
        if not isinstance(segment, dict):
            raise TypeError(f"segments[{index}] must be an object.")
        action = str(segment.get("action", "")).strip()
        if action not in ACTION_WHITELIST:
            raise ValueError(f"segments[{index}].action is unsupported: {action!r}")
        if action != "pose-adjust" and index < len(spec.phase_prior):
            expected = spec.phase_prior[index]
            if action != expected:
                reasons.append(f"phase_mismatch:index={index},expected={expected},got={action}")
        instruction = str(segment.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"segments[{index}].instruction is empty.")
        start = int(segment.get("raw_start_frame"))
        end = int(segment.get("raw_end_frame"))
        if start < 0 or end < start or end >= num_raw_frames:
            raise ValueError(
                f"segments[{index}] has invalid raw frame range: {start}-{end}."
            )
        if index == 0 and start != 0:
            reasons.append(f"first_segment_start_adjusted:{start}->0")
            start = 0
        if previous_end >= 0 and start != previous_end + 1:
            hard_errors.append(f"gap_or_overlap:index={index},prev_end={previous_end},start={start}")
        previous_end = end
        code_policy = str(segment.get("code_policy", "normal")).strip() or "normal"
        if action == "pose-adjust" and code_policy == "normal":
            reasons.append("pose_adjust_code_policy_adjusted:normal->map_to_neighbor")
            code_policy = "map_to_neighbor"
        if action != "pose-adjust" and code_policy not in {"normal", "map_to_neighbor", "always_off"}:
            reasons.append(f"unknown_code_policy:{code_policy}")
        confidence = segment.get("confidence", None)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
            reasons.append(f"missing_or_invalid_confidence:index={index}")
        confidence_value = max(0.0, min(1.0, confidence_value))
        normalized.append(
            {
                "segment_index": index,
                "action": action,
                "instruction": instruction,
                "raw_start_frame": start,
                "raw_end_frame": end,
                "lerobot_start_frame": raw_to_lerobot_frame(start, kept_indices, is_end=False),
                "lerobot_end_frame": raw_to_lerobot_frame(end, kept_indices, is_end=True),
                "code_policy": code_policy,
                "confidence": confidence_value,
                "boundary_reason": str(segment.get("boundary_reason", "")).strip(),
            }
        )

    if normalized[-1]["raw_end_frame"] != num_raw_frames - 1:
        hard_errors.append(
            f"last_segment_end_mismatch:{normalized[-1]['raw_end_frame']}!={num_raw_frames - 1}"
        )
    phase_actions = [seg["action"] for seg in normalized if seg["action"] != "pose-adjust"]
    if phase_actions != spec.phase_prior:
        hard_errors.append(f"phase_sequence_mismatch:expected={spec.phase_prior},got={phase_actions}")

    if hard_errors:
        raise ValueError("; ".join(hard_errors))

    needs_review = bool(data.get("needs_review", False)) or bool(reasons)
    review_reasons = data.get("review_reasons", [])
    if not isinstance(review_reasons, list):
        review_reasons = [str(review_reasons)]
    review_reasons = [str(item) for item in review_reasons] + reasons

    return {
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "segments": normalized,
    }, reasons


# 构造文本纠错提示，要求模型只修正 JSON，不重新解释。
def build_correction_prompt(error: str, num_raw_frames: int, phase_prior: list[str]) -> str:
    return (
        "The previous JSON was invalid for this trajectory segmentation task.\n"
        f"Validation error: {error}\n"
        f"Frame range must cover 0 through {num_raw_frames - 1} with contiguous segments.\n"
        f"The non-pose-adjust action sequence should match this phase_prior: {phase_prior}.\n"
        "Return corrected strict JSON only, with the same schema."
    )


# 单个 episode 的 API 调用、解析、校验与 debug 落盘。
def run_api_for_episode(
    *,
    spec: EpisodeSpec,
    system_prompt: str,
    user_prompt: str,
    image_inputs: list[ImageInput],
    num_raw_frames: int,
    kept_indices: list[int],
    args: argparse.Namespace,
    debug_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key env var: {args.api_key_env}")
    if not args.base_url:
        raise EnvironmentError("Missing --base-url or DASHSCOPE_BASE_URL.")

    messages = build_api_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_inputs=image_inputs,
        image_max_size=args.image_max_size,
        jpeg_quality=args.jpeg_quality,
    )
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = debug_stem(spec)
    last_text = ""
    api_meta: dict[str, Any] = {}
    errors: list[str] = []
    for attempt in range(args.max_retries + 1):
        if attempt > 0:
            messages.append({"role": "assistant", "content": last_text})
            messages.append(
                {
                    "role": "user",
                    "content": build_correction_prompt(errors[-1], num_raw_frames, spec.phase_prior),
                }
            )
        text, call_meta = call_qwenvl_api(
            messages=messages,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        last_text = text
        api_meta = call_meta
        response_path = debug_dir / f"{stem}_response_attempt{attempt}.txt"
        response_path.write_text(text, encoding="utf-8")
        try:
            parsed = parse_json_response(text)
            normalized, validation_reasons = normalize_api_result(
                data=parsed,
                spec=spec,
                num_raw_frames=num_raw_frames,
                kept_indices=kept_indices,
            )
            api_meta["response_path"] = str(response_path)
            api_meta["validation_reasons"] = validation_reasons
            return normalized, api_meta
        except Exception as exc:  # noqa: BLE001 - 这里需要把 API 原始错误写入 cache。
            errors.append(str(exc))

    return {
        "needs_review": True,
        "review_reasons": [f"api_response_invalid_after_retries:{error}" for error in errors],
        "segments": [],
    }, {
        "response_path": str(debug_dir / f"{stem}_response_attempt{args.max_retries}.txt"),
        "errors": errors,
        **api_meta,
    }


# 生成写 debug 文件用的稳定前缀。
def debug_stem(spec: EpisodeSpec) -> str:
    output_index = "unknown" if spec.output_episode_index is None else f"{spec.output_episode_index:06d}"
    return f"{output_index}_{spec.task_name}_{spec.variation}_{spec.episode}"


# 将 numpy 数值、Path 等对象转换成 JSON 可写形式。
def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    return str(value)


# 原子写 JSON，避免中途失败留下半截 cache。
def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    Path(tmp_name).replace(path)


# 初始化或恢复总 cache；重复字段统一放在 meta。
def init_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and args.overwrite:
        return build_empty_cache(args)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists: {args.output}. Use --resume or --overwrite."
            )
        with args.output.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
        if "episodes" not in cache or not isinstance(cache["episodes"], dict):
            raise ValueError(f"Existing cache has invalid format: {args.output}")
        return cache
    return build_empty_cache(args)


# 构造空 cache 的统一 meta。
def build_empty_cache(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "meta": {
            "schema_version": "vqap_subtask_cache_v1",
            "source": "qwen_vl_api",
            "model": args.model,
            "action_space": "delta",
            "phase_label_file": str(args.phase_label),
            "prompt_policy": "longest",
            "raw_root": str(args.raw_root),
            "lerobot_root": str(args.lerobot_root),
            "cameras": parse_cameras(args.cameras),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
        "episodes": {},
    }


# 解析相机列表，默认 5 视角。
def parse_cameras(cameras: str) -> list[str]:
    parsed = [item.strip() for item in cameras.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--cameras must contain at least one camera.")
    return parsed


# 构建单个 episode 的最终 cache record。
def build_episode_record(
    *,
    spec: EpisodeSpec,
    num_raw_frames: int,
    kept_indices: list[int],
    sampled_frames: list[int],
    keyframes: list[int],
    keyframe_warnings: list[str],
    image_warnings: list[str],
    prompt_debug_path: Path,
    api_result: dict[str, Any] | None,
    api_meta: dict[str, Any] | None,
    action_stride: int,
    dry_run: bool,
) -> dict[str, Any]:
    expected_lerobot_frames = max(0, len(kept_indices) - 1)
    warnings = list(keyframe_warnings) + list(image_warnings)
    if spec.lerobot_frames is not None and spec.lerobot_frames != expected_lerobot_frames:
        warnings.append(
            f"lerobot_frame_count_mismatch:progress={spec.lerobot_frames},"
            f"reconstructed={expected_lerobot_frames}"
        )
    record: dict[str, Any] = {
        "task_name": spec.task_name,
        "variation": spec.variation,
        "episode": spec.episode,
        "raw_episode_key": spec.raw_episode_key,
        "output_episode_index": spec.output_episode_index,
        "task_instruction": spec.task_instruction,
        "all_frames": num_raw_frames,
        "lerobot_frames": spec.lerobot_frames,
        "phase_prior": spec.phase_prior,
        "keyframe_candidates": keyframes,
        "sampled_frames": sampled_frames,
        "frame_mapping": {
            "method": "reconstructed_kept_indices_from_action_stride",
            "action_stride": action_stride,
            "kept_indices_count": len(kept_indices),
        },
        "dry_run": dry_run,
        "needs_review": True if dry_run else bool(api_result and api_result.get("needs_review")),
        "review_reasons": ["dry_run_no_api_call"] if dry_run else list(api_result.get("review_reasons", [])),
        "warnings": warnings,
        "debug_files": {"prompt": str(prompt_debug_path)},
        "segments": [] if api_result is None else api_result.get("segments", []),
    }
    if api_meta:
        record["api"] = api_meta
        if response_path := api_meta.get("response_path"):
            record["debug_files"]["response"] = response_path
    return record


# 主流程：收集 episode、生成 prompt、dry-run 或调用 API、增量写 cache。
def main() -> int:
    args = parse_args()
    cameras = parse_cameras(args.cameras)
    debug_dir = args.debug_dir or (args.output.parent / "debug")

    phase_labels = load_phase_labels(args.phase_label)
    conversion_records = load_conversion_records(args.lerobot_root)
    specs = collect_episode_specs(
        raw_root=args.raw_root,
        phase_labels=phase_labels,
        conversion_records=conversion_records,
        tasks=args.tasks,
        variations=args.variations,
        episode_keys=args.episode_keys,
        max_episodes=args.max_episodes,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    if not specs:
        print("[build_subtask_cache] no episode selected.")
        return 0

    cache = build_empty_cache(args) if args.dry_run else init_cache(args)
    processed = 0
    skipped = 0
    total = len(specs)
    print(
        f"[build_subtask_cache] selected={total} output={args.output} "
        f"dry_run={args.dry_run} resume={args.resume} overwrite={args.overwrite} "
        f"retry_review={args.retry_review}",
        flush=True,
    )

    for position, spec in enumerate(specs, start=1):
        progress = f"[{position}/{total}]"
        cache_key = str(spec.output_episode_index) if spec.output_episode_index is not None else spec.raw_episode_key
        if not args.dry_run and args.resume and cache_key in cache["episodes"] and not args.overwrite:
            existing_record = cache["episodes"][cache_key]
            should_retry_review = (
                args.retry_review
                and (
                    bool(existing_record.get("needs_review"))
                    or not existing_record.get("segments")
                )
            )
            if not should_retry_review:
                skipped += 1
                print(f"{progress} [skip] {cache_key} {spec.raw_episode_key}", flush=True)
                continue
            print(f"{progress} [retry-review] {cache_key} {spec.raw_episode_key}", flush=True)
        else:
            print(f"{progress} [start] {cache_key} {spec.raw_episode_key}", flush=True)

        demo = load_demo(spec.episode_dir, args.rlbench_root)
        num_raw_frames = len(demo)
        effective_action_stride = args.action_stride
        stride_warnings: list[str] = []
        if effective_action_stride is None:
            effective_action_stride, stride_warnings = infer_action_stride(
                num_raw_frames,
                spec.lerobot_frames,
            )
        kept_indices = build_kept_indices(num_raw_frames, effective_action_stride)
        keyframes, keyframe_debug, keyframe_warnings = extract_keyframe_candidates(
            demo,
            enabled=not args.no_keyframes,
            min_phase_len=args.min_phase_len,
        )
        sampled_frames = select_sample_frames(
            num_raw_frames=num_raw_frames,
            keyframes=keyframes,
            max_sampled_frames=args.max_sampled_frames,
            keyframe_window=args.keyframe_window,
        )
        image_inputs, image_warnings = collect_image_inputs(
            episode_dir=spec.episode_dir,
            frames=sampled_frames,
            cameras=cameras,
        )
        print(
            f"{progress} [prepared] frames={num_raw_frames} lerobot_frames={spec.lerobot_frames} "
            f"stride={effective_action_stride} phases={len(spec.phase_prior)} "
            f"keyframes={len(keyframes)} sampled={len(sampled_frames)} images={len(image_inputs)}",
            flush=True,
        )
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(
            spec=spec,
            num_raw_frames=num_raw_frames,
            sampled_frames=sampled_frames,
            keyframes=keyframes,
            cameras=cameras,
        )
        prompt_debug_path = write_prompt_debug(
            debug_dir=debug_dir,
            spec=spec,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_inputs=image_inputs,
            sampled_frames=sampled_frames,
            keyframes=keyframes,
            keyframe_debug=keyframe_debug,
            warnings=keyframe_warnings + image_warnings,
        )

        api_result: dict[str, Any] | None = None
        api_meta: dict[str, Any] | None = None
        if not args.dry_run:
            print(
                f"{progress} [api] calling model={args.model} images={len(image_inputs)}",
                flush=True,
            )
            try:
                api_result, api_meta = run_api_for_episode(
                    spec=spec,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_inputs=image_inputs,
                    num_raw_frames=num_raw_frames,
                    kept_indices=kept_indices,
                    args=args,
                    debug_dir=debug_dir,
                )
            except Exception as exc:
                print(
                    f"{progress} [api-error] {cache_key} {spec.raw_episode_key}: {exc}\n"
                    "Completed previous episodes have already been written. "
                    "After fixing quota/network/key issues, rerun with --resume.",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        record = build_episode_record(
            spec=spec,
            num_raw_frames=num_raw_frames,
            kept_indices=kept_indices,
            sampled_frames=sampled_frames,
            keyframes=keyframes,
            keyframe_warnings=keyframe_warnings,
            image_warnings=stride_warnings + image_warnings,
            prompt_debug_path=prompt_debug_path,
            api_result=api_result,
            api_meta=api_meta,
            action_stride=effective_action_stride,
            dry_run=args.dry_run,
        )
        cache["episodes"][cache_key] = record
        cache["meta"]["updated_at"] = utc_now_iso()
        processed += 1

        status = "dry-run" if args.dry_run else "done"
        print(
            f"{progress} [{status}] {cache_key} {spec.raw_episode_key} "
            f"frames={num_raw_frames} phases={len(spec.phase_prior)} "
            f"keyframes={len(keyframes)} sampled={len(sampled_frames)} "
            f"needs_review={record['needs_review']}"
            + (f" warnings={len(record['warnings'])}" if record["warnings"] else ""),
            flush=True,
        )
        if not args.dry_run:
            atomic_write_json(args.output, cache)

        if args.sleep_seconds > 0 and not args.dry_run:
            import time

            time.sleep(args.sleep_seconds)

    if args.dry_run:
        print(
            f"[build_subtask_cache] dry-run complete: processed={processed}, skipped={skipped}. "
            f"Debug prompts are under {debug_dir}",
            flush=True,
        )
    else:
        atomic_write_json(args.output, cache)
        print(
            f"[build_subtask_cache] wrote {args.output}: processed={processed}, skipped={skipped}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
