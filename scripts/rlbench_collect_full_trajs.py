#!/usr/bin/env python3
"""Collect raw RLBench full-trajectory demos for the first-stage dataset."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RLBENCH_ROOT = REPO_ROOT / "source" / "RLBench"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RLBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RLBENCH_ROOT))


TASKS = [
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "light_bulb_in",
    "slide_block_to_target",
    "place_shape_in_shape_sorter",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan",
    "turn_tap",
    "close_jar",
    "reach_and_drag",
    "insert_onto_square_peg",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "stack_wine",
    "push_buttons",
]

EPISODES_FOLDER = "episodes"
EPISODE_FOLDER = "episode%d"
VARIATIONS_FOLDER = "variation%d"
LOW_DIM_PICKLE = "low_dim_obs.pkl"
VARIATION_DESCRIPTIONS = "variation_descriptions.pkl"
IMAGE_FORMAT = "%d.png"
FRONT_RGB_FOLDER = "front_rgb"
WRIST_RGB_FOLDER = "wrist_rgb"
LEFT_SHOULDER_RGB_FOLDER = "left_shoulder_rgb"
RIGHT_SHOULDER_RGB_FOLDER = "right_shoulder_rgb"
OVERHEAD_RGB_FOLDER = "overhead_rgb"

CAMERA_SPECS = [
    ("front_rgb", FRONT_RGB_FOLDER),
    ("wrist_rgb", WRIST_RGB_FOLDER),
    ("left_shoulder_rgb", LEFT_SHOULDER_RGB_FOLDER),
    ("right_shoulder_rgb", RIGHT_SHOULDER_RGB_FOLDER),
    ("overhead_rgb", OVERHEAD_RGB_FOLDER),
]
CAMERA_PREFIXES = ("front", "wrist", "left_shoulder", "right_shoulder", "overhead")

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "RLBench_Raw_Dataset"
DEFAULT_EPISODES_PER_TASK = {"train": 100, "val": 25}
DEFAULT_SEED_BASE = {"train": 0, "val": 100000}
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MAX_VARIATIONS_PER_TASK = 5
DEFAULT_MAX_RETRIES = 10
DEFAULT_NUM_WORKERS = 16
MANIFEST_NAME = "manifest.json"
TASK_TO_INDEX = {task_name: index for index, task_name in enumerate(TASKS)}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


"""按需导入 RLBench / PyRep，保证 --help 这类轻量调用不依赖完整环境。"""
def import_rlbench_dependencies() -> dict[str, Any]:
    try:
        from pyrep.const import RenderMode
        from rlbench import ObservationConfig
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.backend.utils import task_file_to_task_class
        from rlbench.environment import Environment
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        missing_name = exc.name or "unknown dependency"
        raise RuntimeError(
            "Missing RLBench dependency "
            f"'{missing_name}'. Activate the RLBench/PyRep environment before "
            "running collection."
        ) from exc

    return {
        "RenderMode": RenderMode,
        "ObservationConfig": ObservationConfig,
        "MoveArmThenGripper": MoveArmThenGripper,
        "EndEffectorPoseViaPlanning": EndEffectorPoseViaPlanning,
        "Discrete": Discrete,
        "task_file_to_task_class": task_file_to_task_class,
        "Environment": Environment,
    }


"""解析脚本参数，并补上 split 对应的默认值。"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect raw RLBench full-trajectory demos."
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        required=True,
        help="Dataset split to collect.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task ids. Defaults to the fixed 18-task list.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for the raw dataset.",
    )
    parser.add_argument(
        "--max-variations-per-task",
        type=int,
        default=DEFAULT_MAX_VARIATIONS_PER_TASK,
        help="Maximum number of variations to use per task.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=None,
        help="Total number of episodes to collect per task.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        metavar=("HEIGHT", "WIDTH"),
        help="RGB image size to save.",
    )
    parser.add_argument(
        "--renderer",
        choices=("opengl", "opengl3"),
        default="opengl3",
        help="Renderer backend for the RLBench cameras.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Base random seed for this split.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retry count when a live demo collection fails.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()

    if args.episodes_per_task is None:
        args.episodes_per_task = DEFAULT_EPISODES_PER_TASK[args.split]
    if args.seed_base is None:
        args.seed_base = DEFAULT_SEED_BASE[args.split]
    if args.max_variations_per_task < 1:
        raise ValueError("--max-variations-per-task must be >= 1.")
    if args.episodes_per_task < 1:
        raise ValueError("--episodes-per-task must be >= 1.")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1.")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1.")

    return args


"""构造仅包含 5 路 RGB + 低维状态的 ObservationConfig。"""
def build_obs_config(
    image_size: tuple[int, int],
    renderer: str,
    rlbench_deps: dict[str, Any],
) -> Any:
    observation_config_cls = rlbench_deps["ObservationConfig"]
    render_mode_enum = rlbench_deps["RenderMode"]

    obs_config = observation_config_cls()
    obs_config.set_all(False)
    obs_config.set_all_low_dim(True)

    render_mode = (
        render_mode_enum.OPENGL
        if renderer == "opengl"
        else render_mode_enum.OPENGL3
    )
    # 第一阶段只保存 5 路 RGB，depth / point cloud / mask 全部关闭。
    for camera_config in (
        obs_config.front_camera,
        obs_config.wrist_camera,
        obs_config.left_shoulder_camera,
        obs_config.right_shoulder_camera,
        obs_config.overhead_camera,
    ):
        camera_config.rgb = True
        camera_config.depth = False
        camera_config.point_cloud = False
        camera_config.mask = False
        camera_config.image_size = list(image_size)
        camera_config.render_mode = render_mode
        camera_config.depth_in_meters = False
        camera_config.masks_as_one_channel = False

    return obs_config


"""构造 RLBench 环境；这里固定使用绝对位姿规划动作模式。"""
def build_env(obs_config: Any, rlbench_deps: dict[str, Any]) -> Any:
    move_arm_then_gripper = rlbench_deps["MoveArmThenGripper"]
    end_effector_pose_via_planning = rlbench_deps["EndEffectorPoseViaPlanning"]
    discrete = rlbench_deps["Discrete"]
    environment_cls = rlbench_deps["Environment"]

    action_mode = move_arm_then_gripper(
        arm_action_mode=end_effector_pose_via_planning(
            absolute_mode=True,
            collision_checking=True,
        ),
        gripper_action_mode=discrete(),
    )
    return environment_cls(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


"""递归转成 JSON 友好的原生类型，避免 numpy 标量写 manifest 时报错。"""
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


"""原子写 JSON，避免采集中断时留下半写入 manifest。"""
def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                to_jsonable(payload),
                handle,
                indent=2,
                ensure_ascii=True,
                sort_keys=False,
            )
            handle.write("\n")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


"""读取已有 manifest；首次运行时创建最小骨架。"""
def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    now = utc_now_iso()
    return {
        "dataset_root": str(DEFAULT_OUTPUT_ROOT),
        "rlbench_root": str(RLBENCH_ROOT),
        "script": str(SCRIPT_DIR / "rlbench_collect_full_trajs.py"),
        "task_list": TASKS,
        "camera_folders": [folder for _, folder in CAMERA_SPECS],
        "created_at": now,
        "updated_at": now,
        "splits": {},
        "failures": [],
    }


"""写入当前 split 的采集配置，保证 manifest 和本次运行参数一致。"""
def update_manifest_context(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    selected_tasks: list[str],
) -> None:
    now = utc_now_iso()
    manifest["dataset_root"] = str(args.output_root.resolve())
    manifest["updated_at"] = now
    split_manifest = manifest.setdefault("splits", {}).setdefault(args.split, {})
    split_manifest["episodes_per_task"] = args.episodes_per_task
    split_manifest["max_variations_per_task"] = args.max_variations_per_task
    split_manifest["seed_base"] = args.seed_base
    split_manifest["image_size"] = list(args.image_size)
    split_manifest["renderer"] = args.renderer
    split_manifest["num_workers_requested"] = args.num_workers
    split_manifest["tasks_requested"] = selected_tasks
    split_manifest.setdefault("tasks", {})
    split_manifest.setdefault("totals", {"target_episodes": 0, "collected_episodes": 0})


"""按 round-robin 生成 task 内的 variation 采集计划。"""
def build_task_plan(
    variation_numbers: list[int], episodes_per_task: int
) -> tuple[list[tuple[int, int]], dict[int, int]]:
    if not variation_numbers:
        raise ValueError("variation_numbers must not be empty.")

    # 这里同时维护每个 variation 的局部 episode 编号，方便落盘为 episode0/1/2...
    per_variation_counts = {variation: 0 for variation in variation_numbers}
    plan: list[tuple[int, int]] = []
    for global_index in range(episodes_per_task):
        variation = variation_numbers[global_index % len(variation_numbers)]
        local_episode_index = per_variation_counts[variation]
        per_variation_counts[variation] += 1
        plan.append((variation, local_episode_index))
    return plan, per_variation_counts


"""生成稳定的 episode 级随机种子，保证同一任务在重跑时可复现。"""
def episode_seed(
    seed_base: int,
    task_position: int,
    variation: int,
    local_episode_index: int,
) -> int:
    return seed_base + task_position * 100000 + variation * 1000 + local_episode_index


"""保存 low_dim 之前清掉高维图像字段，避免把图像重复 pickle 进去。"""
def clear_high_dim_fields(observation: Any) -> None:
    for prefix in CAMERA_PREFIXES:
        for suffix in ("rgb", "depth", "point_cloud", "mask"):
            setattr(observation, f"{prefix}_{suffix}", None)


"""把一个完整 demo 保存成 5 路 RGB + low_dim_obs.pkl。"""
def save_demo(demo: Any, episode_path: Path) -> None:
    ensure_dir(episode_path)
    camera_dirs = {folder: episode_path / folder for _, folder in CAMERA_SPECS}
    for camera_dir in camera_dirs.values():
        ensure_dir(camera_dir)

    # 先逐帧落 RGB，再把 observation 里的高维字段清空后统一 pickle 低维观测。
    for step_index, observation in enumerate(demo):
        for attribute_name, folder in CAMERA_SPECS:
            image_array = getattr(observation, attribute_name)
            if image_array is None:
                raise RuntimeError(
                    f"Missing {attribute_name} at step {step_index} for {episode_path}."
                )
            image = Image.fromarray(np.asarray(image_array, dtype=np.uint8))
            image.save(camera_dirs[folder] / (IMAGE_FORMAT % step_index))
        clear_high_dim_fields(observation)

    with (episode_path / LOW_DIM_PICKLE).open("wb") as handle:
        pickle.dump(demo, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""检查一个 episode 目录是否完整，可用于默认断点续跑。"""
def episode_status(episode_path: Path) -> tuple[bool, str]:
    if not episode_path.exists():
        return False, "missing episode directory"

    low_dim_path = episode_path / LOW_DIM_PICKLE
    if not low_dim_path.exists():
        return False, "missing low_dim_obs.pkl"

    try:
        with low_dim_path.open("rb") as handle:
            demo = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        return False, f"failed to read low_dim_obs.pkl: {exc}"

    step_count = len(demo)
    if step_count <= 0:
        return False, "empty demo"

    for _, folder in CAMERA_SPECS:
        camera_dir = episode_path / folder
        if not camera_dir.is_dir():
            return False, f"missing {folder}"
        png_count = len(list(camera_dir.glob("*.png")))
        if png_count != step_count:
            return False, f"{folder} has {png_count} frames, expected {step_count}"

    return True, "complete"


"""为每个 variation 保存语言描述；已有文件时直接复用。"""
def ensure_variation_descriptions(task_env: Any, variation: int, variation_path: Path) -> None:
    description_path = variation_path / VARIATION_DESCRIPTIONS
    if description_path.exists():
        return

    ensure_dir(variation_path)
    task_env.set_variation(variation)
    descriptions, _ = task_env.reset()
    with description_path.open("wb") as handle:
        pickle.dump(descriptions, handle, protocol=pickle.HIGHEST_PROTOCOL)


def task_variation_path(output_root: Path, split: str, task_name: str, variation: int) -> Path:
    return output_root / split / task_name / (VARIATIONS_FOLDER % variation)


def episode_path_for_job(
    output_root: Path,
    split: str,
    task_name: str,
    variation: int,
    local_episode_index: int,
) -> Path:
    return (
        task_variation_path(output_root, split, task_name, variation)
        / EPISODES_FOLDER
        / (EPISODE_FOLDER % local_episode_index)
    )


def episode_lock_path(episode_path: Path) -> Path:
    return episode_path.parent / f"{episode_path.name}.lock"


def split_lock_path(output_root: Path, split: str) -> Path:
    return output_root / f".{split}.collect.lock"


"""给单个 episode 加文件锁，避免并发写同一目录。"""
@contextmanager
def episode_file_lock(lock_path: Path):
    ensure_dir(lock_path.parent)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


"""给整个 split 加运行锁，避免两个顶层脚本同时改同一份 manifest。"""
@contextmanager
def split_run_lock(lock_path: Path):
    ensure_dir(lock_path.parent)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another collector is already running for this split: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


"""拿到 manifest 中当前 task 的统计入口，不存在就创建。"""
def task_entry_for_manifest(
    split_manifest: dict[str, Any],
    task_name: str,
) -> dict[str, Any]:
    return split_manifest.setdefault("tasks", {}).setdefault(task_name, {})


"""把单个 task 的采集进度回写到 manifest。"""
def sync_task_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    task_name: str,
    available_variations: int,
    used_variations: list[int],
    variation_targets: dict[int, int],
    complete_indices: dict[int, set[int]],
) -> None:
    split_manifest = manifest["splits"][args.split]
    task_entry = task_entry_for_manifest(split_manifest, task_name)
    task_entry["available_variations"] = available_variations
    task_entry["used_variations"] = used_variations
    task_entry["target_episodes"] = args.episodes_per_task
    task_entry["collected_episodes"] = sum(len(indices) for indices in complete_indices.values())
    task_entry["variations"] = {}
    for variation in used_variations:
        task_entry["variations"][f"variation{variation}"] = {
            "target_episodes": variation_targets[variation],
            "collected_episodes": len(complete_indices[variation]),
        }

    split_manifest["totals"]["target_episodes"] = (
        sum(task["target_episodes"] for task in split_manifest["tasks"].values())
    )
    split_manifest["totals"]["collected_episodes"] = sum(
        task["collected_episodes"] for task in split_manifest["tasks"].values()
    )
    manifest["updated_at"] = utc_now_iso()


"""记录采集失败，便于后续定位具体 task / variation / episode。"""
def record_failure(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    task_name: str | None,
    variation: int | None,
    local_episode_index: int | None,
    seed: int | None,
    stage: str,
    error: str,
) -> None:
    manifest.setdefault("failures", []).append(
        {
            "timestamp": utc_now_iso(),
            "split": args.split,
            "task": task_name,
            "variation": variation,
            "episode": local_episode_index,
            "seed": seed,
            "stage": stage,
            "error": error,
        }
    )
    manifest["updated_at"] = utc_now_iso()


"""把多个 task 的 pending jobs 交织起来，避免前几个 task 长时间独占 worker。"""
def interleave_job_groups(job_groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    ordered_jobs: list[dict[str, Any]] = []
    non_empty_groups = [group for group in job_groups if group]
    max_group_size = max((len(group) for group in non_empty_groups), default=0)
    for group_index in range(max_group_size):
        for group in non_empty_groups:
            if group_index < len(group):
                ordered_jobs.append(group[group_index])
    return ordered_jobs


"""构造单个 episode 的采集 job。"""
def build_episode_job(
    args: argparse.Namespace,
    task_name: str,
    task_position: int,
    variation: int,
    local_episode_index: int,
) -> dict[str, Any]:
    episode_path = episode_path_for_job(
        args.output_root,
        args.split,
        task_name,
        variation,
        local_episode_index,
    )
    return {
        "task_name": task_name,
        "variation": variation,
        "local_episode_index": local_episode_index,
        "seed": episode_seed(args.seed_base, task_position, variation, local_episode_index),
        "episode_path": str(episode_path),
        "lock_path": str(episode_lock_path(episode_path)),
    }


"""根据机器资源和待采集数量，决定本次真正启用的 worker 数。"""
def resolve_num_workers(requested_workers: int, pending_job_count: int) -> int:
    if pending_job_count <= 0:
        return 0
    cpu_count = os.cpu_count() or 1
    return max(1, min(requested_workers, cpu_count, pending_job_count))


"""扫描 task，建立已有进度和待采集 jobs。"""
def prepare_task_specs(
    env: Any,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    selected_tasks: list[str],
    rlbench_deps: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    task_specs: dict[str, dict[str, Any]] = {}
    pending_by_task: list[list[dict[str, Any]]] = []

    for task_name in selected_tasks:
        task_class = rlbench_deps["task_file_to_task_class"](task_name)
        task_env = env.get_task(task_class)
        available_variations = task_env.variation_count()
        used_variations = list(range(min(available_variations, args.max_variations_per_task)))
        if not used_variations:
            raise RuntimeError(f"Task {task_name} has no available variation.")

        plan, variation_targets = build_task_plan(used_variations, args.episodes_per_task)
        complete_indices = {variation: set() for variation in used_variations}
        blocked_variations: set[int] = set()

        print(
            f"[{args.split}] task={task_name} available_variations={available_variations} "
            f"used_variations={used_variations} target_episodes={args.episodes_per_task}"
        )

        # 先把 variation 描述补齐，失败的 variation 本轮直接不再派发新 job。
        for variation in used_variations:
            variation_path = task_variation_path(args.output_root, args.split, task_name, variation)
            try:
                ensure_variation_descriptions(task_env, variation, variation_path)
            except Exception as exc:  # pragma: no cover - runtime-only path
                blocked_variations.add(variation)
                record_failure(
                    manifest,
                    args,
                    task_name,
                    variation,
                    None,
                    None,
                    "variation_setup",
                    str(exc),
                )
                print(
                    f"[{args.split}] task={task_name} variation={variation} "
                    f"failed to save descriptions: {exc}"
                )

        pending_jobs: list[dict[str, Any]] = []
        task_position = TASK_TO_INDEX[task_name]
        for variation, local_episode_index in plan:
            episode_path = episode_path_for_job(
                args.output_root,
                args.split,
                task_name,
                variation,
                local_episode_index,
            )
            is_complete, _ = episode_status(episode_path)
            if is_complete:
                complete_indices[variation].add(local_episode_index)
                continue
            if variation in blocked_variations:
                continue
            pending_jobs.append(
                build_episode_job(
                    args,
                    task_name,
                    task_position,
                    variation,
                    local_episode_index,
                )
            )

        task_specs[task_name] = {
            "available_variations": available_variations,
            "used_variations": used_variations,
            "variation_targets": variation_targets,
            "complete_indices": complete_indices,
        }
        pending_by_task.append(pending_jobs)

        sync_task_manifest(
            manifest,
            args,
            task_name,
            available_variations,
            used_variations,
            variation_targets,
            complete_indices,
        )
        atomic_write_json(manifest_path, manifest)

        print(
            f"[{args.split}] task={task_name} ready collected="
            f"{sum(len(indices) for indices in complete_indices.values())} "
            f"pending={len(pending_jobs)}"
        )

    return task_specs, interleave_job_groups(pending_by_task)


"""把 worker 回传的结果并入 manifest 统计。"""
def apply_worker_result(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    manifest_path: Path,
    task_specs: dict[str, dict[str, Any]],
    result: dict[str, Any],
) -> None:
    task_name = result["task_name"]
    variation = result["variation"]
    local_episode_index = result["local_episode_index"]
    task_spec = task_specs[task_name]

    if result["status"] in {"success", "skipped"}:
        task_spec["complete_indices"][variation].add(local_episode_index)
    else:
        record_failure(
            manifest,
            args,
            task_name,
            variation,
            local_episode_index,
            result["seed"],
            result["stage"],
            result["error"],
        )

    sync_task_manifest(
        manifest,
        args,
        task_name,
        task_spec["available_variations"],
        task_spec["used_variations"],
        task_spec["variation_targets"],
        task_spec["complete_indices"],
    )
    atomic_write_json(manifest_path, manifest)


"""采集单个 episode；worker 里每拿到一个 job 就执行一次。"""
def collect_episode_job(
    env: Any,
    args: argparse.Namespace,
    worker_index: int,
    task_env_cache: dict[str, Any],
    job: dict[str, Any],
    rlbench_deps: dict[str, Any],
) -> dict[str, Any]:
    task_name = job["task_name"]
    variation = job["variation"]
    local_episode_index = job["local_episode_index"]
    seed = job["seed"]
    episode_path = Path(job["episode_path"])
    lock_path = Path(job["lock_path"])

    task_env = task_env_cache.get(task_name)
    if task_env is None:
        task_class = rlbench_deps["task_file_to_task_class"](task_name)
        task_env = env.get_task(task_class)
        task_env_cache[task_name] = task_env

    with episode_file_lock(lock_path):
        # 进 worker 后再做一次检查，避免恢复运行时和其他实例撞到同一 episode。
        is_complete, reason = episode_status(episode_path)
        if is_complete:
            print(
                f"[{args.split}] worker={worker_index} skip task={task_name} "
                f"variation={variation} episode={local_episode_index}: already complete"
            )
            return {
                "type": "episode",
                "status": "skipped",
                "task_name": task_name,
                "variation": variation,
                "local_episode_index": local_episode_index,
                "seed": seed,
                "reason": reason,
            }

        if episode_path.exists():
            # 残缺目录先删掉，避免旧文件污染本次重采。
            shutil.rmtree(episode_path)

        ensure_dir(episode_path.parent)
        print(
            f"[{args.split}] worker={worker_index} collect task={task_name} "
            f"variation={variation} episode={local_episode_index} seed={seed}"
        )

        try:
            task_env.set_variation(variation)
            np.random.seed(seed)
            # live_demos=True 直接从 RLBench 环境生成专家完整轨迹。
            demo, = task_env.get_demos(
                amount=1,
                live_demos=True,
                max_attempts=args.max_retries,
            )
            save_demo(demo, episode_path)
            is_complete, reason = episode_status(episode_path)
            if not is_complete:
                raise RuntimeError(f"post-save validation failed: {reason}")
            return {
                "type": "episode",
                "status": "success",
                "task_name": task_name,
                "variation": variation,
                "local_episode_index": local_episode_index,
                "seed": seed,
            }
        except Exception as exc:  # pragma: no cover - runtime-only path
            if episode_path.exists():
                shutil.rmtree(episode_path)
            print(
                f"[{args.split}] worker={worker_index} failed task={task_name} "
                f"variation={variation} episode={local_episode_index}: {exc}"
            )
            return {
                "type": "episode",
                "status": "failed",
                "task_name": task_name,
                "variation": variation,
                "local_episode_index": local_episode_index,
                "seed": seed,
                "stage": "collect_episode",
                "error": str(exc),
            }


"""单个 worker 的主循环：初始化 RLBench 环境，然后持续消费 jobs。"""
def worker_main(
    worker_index: int,
    args: argparse.Namespace,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    env = None
    launched = False
    try:
        rlbench_deps = import_rlbench_dependencies()
        obs_config = build_obs_config(tuple(args.image_size), args.renderer, rlbench_deps)
        env = build_env(obs_config, rlbench_deps)
        env.launch()
        launched = True

        task_env_cache: dict[str, Any] = {}
        while True:
            job = job_queue.get()
            if job is None:
                break
            result_queue.put(
                collect_episode_job(
                    env=env,
                    args=args,
                    worker_index=worker_index,
                    task_env_cache=task_env_cache,
                    job=job,
                    rlbench_deps=rlbench_deps,
                )
            )
    except Exception:  # pragma: no cover - runtime-only path
        result_queue.put(
            {
                "type": "worker_error",
                "worker_index": worker_index,
                "error": traceback.format_exc(),
            }
        )
    finally:
        if env is not None and launched:
            env.shutdown()
        result_queue.put(
            {
                "type": "worker_done",
                "worker_index": worker_index,
                "pid": os.getpid(),
            }
        )


"""启动并行采集；manifest 始终只由父进程串行写入。"""
def run_parallel_collection(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    task_specs: dict[str, dict[str, Any]],
    pending_jobs: list[dict[str, Any]],
) -> None:
    effective_workers = resolve_num_workers(args.num_workers, len(pending_jobs))
    manifest["splits"][args.split]["num_workers_effective"] = effective_workers
    manifest["updated_at"] = utc_now_iso()
    atomic_write_json(manifest_path, manifest)

    if effective_workers == 0:
        print(f"[{args.split}] no pending episodes. manifest={manifest_path}")
        return

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for job in pending_jobs:
        job_queue.put(job)
    for _ in range(effective_workers):
        job_queue.put(None)

    workers = [
        ctx.Process(
            target=worker_main,
            args=(worker_index, args, job_queue, result_queue),
            name=f"rlbench-collector-{worker_index}",
        )
        for worker_index in range(effective_workers)
    ]
    for worker in workers:
        worker.start()

    print(
        f"[{args.split}] start parallel collection: "
        f"workers={effective_workers} pending_episodes={len(pending_jobs)}"
    )

    processed_episodes = 0
    finished_workers = 0
    pending_error: Exception | None = None

    try:
        while finished_workers < effective_workers:
            try:
                result = result_queue.get(timeout=5.0)
            except queue.Empty:
                failed_workers = [
                    (worker.name, worker.exitcode)
                    for worker in workers
                    if worker.exitcode not in (None, 0)
                ]
                if failed_workers:
                    raise RuntimeError(f"Worker exited unexpectedly: {failed_workers}")
                continue

            if result["type"] == "episode":
                processed_episodes += 1
                apply_worker_result(
                    manifest=manifest,
                    args=args,
                    manifest_path=manifest_path,
                    task_specs=task_specs,
                    result=result,
                )
                print(
                    f"[{args.split}] progress {processed_episodes}/{len(pending_jobs)} "
                    f"last={result['status']} task={result['task_name']} "
                    f"variation={result['variation']} "
                    f"episode={result['local_episode_index']}"
                )
            elif result["type"] == "worker_error":
                record_failure(
                    manifest,
                    args,
                    None,
                    None,
                    None,
                    None,
                    "worker_process",
                    f"worker={result['worker_index']}\n{result['error']}",
                )
                atomic_write_json(manifest_path, manifest)
                raise RuntimeError(f"Worker {result['worker_index']} crashed.")
            elif result["type"] == "worker_done":
                finished_workers += 1

        if processed_episodes != len(pending_jobs):
            raise RuntimeError(
                "Processed episode count does not match the pending job count: "
                f"{processed_episodes} vs {len(pending_jobs)}."
            )
    except Exception as exc:
        pending_error = exc
    finally:
        if pending_error is not None:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
        for worker in workers:
            worker.join()

    bad_exit_codes = [
        (worker.name, worker.exitcode)
        for worker in workers
        if worker.exitcode not in (0, None)
    ]
    if pending_error is not None:
        raise pending_error
    if bad_exit_codes:
        raise RuntimeError(f"Worker exited with non-zero code: {bad_exit_codes}")


"""脚本主入口：先扫描恢复状态，再由多个 worker 并行采集。"""
def main() -> int:
    args = parse_args()
    selected_tasks = list(args.tasks) if args.tasks else list(TASKS)
    unknown_tasks = sorted(set(selected_tasks) - set(TASKS))
    if unknown_tasks:
        raise ValueError(f"Unknown task ids: {unknown_tasks}")

    # 保留用户传入顺序，同时去掉重复 task。
    selected_tasks = list(dict.fromkeys(selected_tasks))
    run_lock_path = split_lock_path(args.output_root, args.split)

    with split_run_lock(run_lock_path):
        manifest_path = args.output_root / MANIFEST_NAME
        manifest = load_manifest(manifest_path)
        update_manifest_context(manifest, args, selected_tasks)
        atomic_write_json(manifest_path, manifest)

        rlbench_deps = import_rlbench_dependencies()
        obs_config = build_obs_config(tuple(args.image_size), args.renderer, rlbench_deps)
        env = build_env(obs_config, rlbench_deps)
        launched = False

        try:
            env.launch()
            launched = True
            task_specs, pending_jobs = prepare_task_specs(
                env=env,
                args=args,
                manifest=manifest,
                manifest_path=manifest_path,
                selected_tasks=selected_tasks,
                rlbench_deps=rlbench_deps,
            )
        finally:
            if launched:
                env.shutdown()

        run_parallel_collection(
            args=args,
            manifest=manifest,
            manifest_path=manifest_path,
            task_specs=task_specs,
            pending_jobs=pending_jobs,
        )

        print(f"[{args.split}] collection finished. manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
