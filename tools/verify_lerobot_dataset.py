#!/usr/bin/env python3
"""Verify a converted LeRobot dataset against the raw RLBench dataset.

对每个 episode 做端到端一致性校验（默认全量,图像抽样）：
1. 账本一致：info.json / episodes.jsonl / tasks.jsonl / .conversion_progress.jsonl
   数量互相吻合,输出索引连续。
2. 来源身份：progress log 指向的原始 episode 目录存在,其 meta.json 的任务身份正确。
3. 指令对齐：episode 的 task 字符串必须能由该 episode 所在 variation 的
   variation_descriptions.pkl 按指定 prompt 策略确定性推出。
4. 状态/动作对齐：从原始 low_dim_obs.pkl 按 stride 独立重算 state 与 action。
   delta 旋转用旋转矩阵路径独立实现（不共用转换脚本的四元数代码）,交叉验证数学。
5. 图像对齐：抽样 episode,解码 parquet 内嵌 PNG 与原始帧逐像素比较。

用法:
    python tools/verify_lerobot_dataset.py \
        --dataset-root LeRobot_RLBench_Dataset/train_delta \
        --raw-root RLBench_Raw_Dataset --split train \
        --action-stride 3 --action-repr delta --prompt-strategy random
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import pickle
import random
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RLBENCH_ROOT = REPO_ROOT / "source" / "RLBench"

LOW_DIM_PICKLE = "low_dim_obs.pkl"
EPISODE_META = "meta.json"
VARIATION_DESCRIPTIONS = "variation_descriptions.pkl"
PROGRESS_LOG_NAME = ".conversion_progress.jsonl"
CAMERA_COLUMNS = {
    "observation.images.front": "front_rgb",
    "observation.images.wrist": "wrist_rgb",
    "observation.images.left_shoulder": "left_shoulder_rgb",
    "observation.images.right_shoulder": "right_shoulder_rgb",
    "observation.images.overhead": "overhead_rgb",
}


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core":
            module = "numpy.core"
        elif module.startswith("numpy._core."):
            module = module.replace("numpy._core.", "numpy.core.", 1)
        return super().find_class(module, name)


def install_rlbench_pickle_support(rlbench_root: Path) -> None:
    package_root = rlbench_root / "rlbench"
    backend_root = package_root / "backend"
    observation_py = backend_root / "observation.py"
    if not observation_py.is_file():
        raise FileNotFoundError(f"Observation definition not found: {observation_py}")
    if "rlbench" not in sys.modules:
        module = types.ModuleType("rlbench")
        module.__path__ = [str(package_root)]
        sys.modules["rlbench"] = module
    if "rlbench.backend" not in sys.modules:
        module = types.ModuleType("rlbench.backend")
        module.__path__ = [str(backend_root)]
        sys.modules["rlbench.backend"] = module
    if "rlbench.backend.observation" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "rlbench.backend.observation", observation_py
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["rlbench.backend.observation"] = module
        spec.loader.exec_module(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "RLBench_Raw_Dataset")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--rlbench-root", type=Path, default=DEFAULT_RLBENCH_ROOT)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--action-repr", choices=("absolute", "delta"), default="absolute")
    parser.add_argument(
        "--prompt-strategy", choices=("longest", "first", "shortest", "random"),
        default="longest",
    )
    parser.add_argument(
        "--image-sample-episodes", type=int, default=60,
        help="Number of episodes for pixel-level image comparison (0 = skip).",
    )
    parser.add_argument(
        "--image-frames-per-episode", type=int, default=2,
        help="Frames per sampled episode for image comparison.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    return parser.parse_args()


# ---------- 独立实现的四元数/旋转数学（勿与转换脚本共用代码） ----------

def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def sign_normalize(q: np.ndarray) -> np.ndarray:
    return -q if q[3] < 0 else q


def expected_instruction(descriptions: list[str], strategy: str, episode_key: str) -> str:
    cleaned = [text.strip() for text in descriptions if text and text.strip()]
    if strategy == "first":
        return cleaned[0]
    if strategy == "shortest":
        return min(cleaned, key=len)
    if strategy == "random":
        seed = int.from_bytes(hashlib.sha256(episode_key.encode("utf-8")).digest()[:8], "big")
        return random.Random(seed).choice(cleaned)
    return max(cleaned, key=len)


def kept_indices_for(num_observations: int, stride: int) -> list[int]:
    kept = list(range(0, num_observations, stride))
    if kept[-1] != num_observations - 1:
        kept.append(num_observations - 1)
    return kept


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_WORKER_ARGS: dict[str, Any] = {}


def init_worker(rlbench_root: str, config: dict[str, Any]) -> None:
    install_rlbench_pickle_support(Path(rlbench_root))
    _WORKER_ARGS.update(config)


def verify_episode(job: dict[str, Any]) -> list[str]:
    """Verify one episode; returns a list of problems (empty = OK)."""
    problems: list[str] = []
    raw_dir = Path(job["raw_episode_dir"])
    parquet_path = Path(job["parquet_path"])
    stride = _WORKER_ARGS["action_stride"]
    action_repr = _WORKER_ARGS["action_repr"]
    key = job["raw_episode_key"]

    import pyarrow.parquet as pq

    # 1. 原始 episode meta 身份
    meta_path = raw_dir / EPISODE_META
    if not meta_path.is_file():
        return [f"{key}: raw episode missing {EPISODE_META}"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("task_name") != job["task"] or meta.get("scene_task_name") != job["task"]:
        problems.append(
            f"{key}: raw meta identity mismatch "
            f"(task_name={meta.get('task_name')!r}, scene={meta.get('scene_task_name')!r})"
        )

    # 2. 从原始 low_dim 独立重算 state / action
    with (raw_dir / LOW_DIM_PICKLE).open("rb") as handle:
        demo = CompatibleUnpickler(handle).load()
    kept = kept_indices_for(len(demo), stride)
    expected_frames = len(kept) - 1

    poses = np.stack(
        [np.asarray(demo[i].gripper_pose, dtype=np.float64) for i in kept], axis=0
    )
    grips = np.asarray([float(demo[i].gripper_open) for i in kept], dtype=np.float64)

    table = pq.read_table(
        parquet_path,
        columns=["observation.state", "action", "frame_index", "task_index"],
    )
    if table.num_rows != expected_frames:
        problems.append(
            f"{key}: parquet has {table.num_rows} frames, expected {expected_frames}"
        )
        return problems

    states = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
    actions = np.stack(table.column("action").to_numpy(zero_copy_only=False))

    for t in range(expected_frames):
        expected_state = np.concatenate(
            [poses[t, :3], sign_normalize(poses[t, 3:7]), grips[t : t + 1]]
        )
        if not np.allclose(states[t], expected_state, atol=1e-5):
            problems.append(
                f"{key}: state mismatch at frame {t}: {states[t]} vs {expected_state}"
            )
            break

    for t in range(expected_frames):
        if action_repr == "delta":
            dxyz = poses[t + 1, :3] - poses[t, :3]
            rotation_delta = quat_xyzw_to_matrix(poses[t + 1, 3:7]) @ quat_xyzw_to_matrix(
                poses[t, 3:7]
            ).T
            dquat = sign_normalize(matrix_to_quat_xyzw(rotation_delta))
            expected_action = np.concatenate([dxyz, dquat, grips[t + 1 : t + 2]])
        else:
            expected_action = np.concatenate(
                [poses[t + 1, :3], sign_normalize(poses[t + 1, 3:7]), grips[t + 1 : t + 2]]
            )
        if not np.allclose(actions[t], expected_action, atol=1e-4):
            problems.append(
                f"{key}: action mismatch at frame {t}: {actions[t]} vs {expected_action}"
            )
            break

    # 3. 指令对齐
    task_index_values = set(table.column("task_index").to_pylist())
    if len(task_index_values) != 1:
        problems.append(f"{key}: multiple task_index values in one episode: {task_index_values}")
    else:
        task_index = task_index_values.pop()
        instruction = _WORKER_ARGS["task_index_to_text"][str(task_index)]
        descriptions = _WORKER_ARGS["descriptions_by_variation"][
            f"{job['task']}/{job['variation']}"
        ]
        expected = expected_instruction(
            descriptions, _WORKER_ARGS["prompt_strategy"], key
        )
        if instruction != expected:
            problems.append(
                f"{key}: instruction mismatch: dataset={instruction!r}, expected={expected!r}"
            )
        if instruction.strip() not in {d.strip() for d in descriptions}:
            problems.append(
                f"{key}: instruction {instruction!r} not in variation descriptions"
            )

    # 4. 图像逐像素抽查
    if job["image_frame_samples"]:
        image_table = pq.read_table(parquet_path, columns=list(CAMERA_COLUMNS))
        for t in job["image_frame_samples"]:
            raw_frame_index = kept[t]
            for column, raw_folder in CAMERA_COLUMNS.items():
                blob = image_table.column(column)[t].as_py()["bytes"]
                with Image.open(io.BytesIO(blob)) as image:
                    converted = np.asarray(image.convert("RGB"))
                raw_path = raw_dir / raw_folder / f"{raw_frame_index}.png"
                with Image.open(raw_path) as image:
                    original = np.asarray(image.convert("RGB"))
                if not np.array_equal(converted, original):
                    problems.append(
                        f"{key}: image mismatch frame {t} ({column} vs {raw_path.name})"
                    )
    return problems


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    split_root = args.raw_root.resolve() / args.split

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes_meta = load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks_meta = load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    progress = load_jsonl(dataset_root / PROGRESS_LOG_NAME)

    problems: list[str] = []
    if info["total_episodes"] != len(episodes_meta):
        problems.append(
            f"info total_episodes={info['total_episodes']} != episodes.jsonl {len(episodes_meta)}"
        )
    if info["total_episodes"] != len(progress):
        problems.append(
            f"info total_episodes={info['total_episodes']} != progress log {len(progress)}"
        )
    indices = sorted(int(r["output_episode_index"]) for r in progress)
    if indices != list(range(len(progress))):
        problems.append("progress log output_episode_index not contiguous from 0")

    task_index_to_text = {str(r["task_index"]): r["task"] for r in tasks_meta}
    episode_text = {
        int(r["episode_index"]): r["tasks"][0] for r in episodes_meta
    }
    episode_length = {int(r["episode_index"]): int(r["length"]) for r in episodes_meta}

    # 预载各 variation 的语言描述
    descriptions_by_variation: dict[str, list[str]] = {}
    for record in progress:
        vkey = f"{record['task']}/{record['variation']}"
        if vkey not in descriptions_by_variation:
            desc_path = split_root / record["task"] / record["variation"] / VARIATION_DESCRIPTIONS
            with desc_path.open("rb") as handle:
                descriptions_by_variation[vkey] = pickle.load(handle)

    chunks_size = int(info.get("chunks_size", 1000))
    rng = random.Random(args.seed)
    image_episode_set = set(
        rng.sample(range(len(progress)), min(args.image_sample_episodes, len(progress)))
    )

    jobs = []
    for record in progress:
        output_index = int(record["output_episode_index"])
        parquet_path = (
            dataset_root
            / "data"
            / f"chunk-{output_index // chunks_size:03d}"
            / f"episode_{output_index:06d}.parquet"
        )
        if not parquet_path.is_file():
            problems.append(f"{record['raw_episode_key']}: missing parquet {parquet_path}")
            continue
        frames = episode_length.get(output_index)
        if frames is None:
            problems.append(f"{record['raw_episode_key']}: not present in episodes.jsonl")
            continue
        if record.get("frames") is not None and int(record["frames"]) != frames:
            problems.append(
                f"{record['raw_episode_key']}: progress frames={record['frames']} "
                f"!= episodes.jsonl length={frames}"
            )
        image_samples: list[int] = []
        if output_index in image_episode_set and frames > 0:
            count = min(args.image_frames_per_episode, frames)
            image_samples = sorted(rng.sample(range(frames), count))
        jobs.append(
            {
                "raw_episode_key": record["raw_episode_key"],
                "task": record["task"],
                "variation": record["variation"],
                "raw_episode_dir": str(
                    split_root / record["task"] / record["variation"] / "episodes" / record["episode"]
                ),
                "parquet_path": str(parquet_path),
                "output_episode_index": output_index,
                "image_frame_samples": image_samples,
            }
        )

    worker_config = {
        "action_stride": args.action_stride,
        "action_repr": args.action_repr,
        "prompt_strategy": args.prompt_strategy,
        "task_index_to_text": task_index_to_text,
        "descriptions_by_variation": descriptions_by_variation,
    }

    # 附带校验 episodes.jsonl 的指令文本与 tasks.jsonl 一致(逐 episode 的 task_index 在 worker 中查)
    for output_index, text in episode_text.items():
        if text not in {r["task"] for r in tasks_meta}:
            problems.append(f"episode {output_index}: instruction not present in tasks.jsonl")

    done = 0
    if args.num_workers <= 1:
        init_worker(str(args.rlbench_root), worker_config)
        for job in jobs:
            problems.extend(verify_episode(job))
            done += 1
            if done % 200 == 0:
                print(f"[verify] {done}/{len(jobs)} episodes checked")
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker,
            initargs=(str(args.rlbench_root), worker_config),
        ) as executor:
            for result in executor.map(verify_episode, jobs, chunksize=8):
                problems.extend(result)
                done += 1
                if done % 200 == 0:
                    print(f"[verify] {done}/{len(jobs)} episodes checked")

    print(
        f"[verify] dataset={dataset_root.name} episodes={len(jobs)} "
        f"image_sampled={len(image_episode_set)} stride={args.action_stride} "
        f"repr={args.action_repr} prompt={args.prompt_strategy}"
    )
    if problems:
        print(f"\n[FAIL] {len(problems)} problem(s):")
        for problem in problems[:100]:
            print(f"  - {problem}")
        if len(problems) > 100:
            print(f"  ... and {len(problems) - 100} more")
        return 1
    print("[PASS] LeRobot dataset is fully consistent with the raw dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
