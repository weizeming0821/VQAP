#!/usr/bin/env python3
"""Verify RLBench_Raw_Dataset cleanliness: task identity, completeness, visual outliers.

校验层次（全部通过才算干净）：
1. meta.json 身份校验：每个 episode 的 task_name / scene_task_name / variation
   必须与其所在目录一致（meta 由修复后的采集脚本写入，记录 scene 中真正加载的任务）。
2. 结构完整性：low_dim_obs.pkl 可读、5 路相机帧数与 demo 步数一致。
3. task_low_dim_state 维度一致性：同一 task 的所有 episode 低维状态维度应一致
   （不同任务物体数不同，维度串台是任务串台的强信号）。
4. 视觉离群检测：同一 task/variation 的 front_rgb 首帧下采样后与中位图比较，
   偏差异常的 episode 会被标记（场景随机化只移动小物体，大件家具/背景占主导）。

用法:
    python tools/verify_raw_dataset.py --root RLBench_Raw_Dataset --split train
    python tools/verify_raw_dataset.py --root RLBench_Raw_Dataset --split val --contact-sheet-dir viz/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RLBENCH_ROOT = REPO_ROOT / "source" / "RLBench"

LOW_DIM_PICKLE = "low_dim_obs.pkl"
EPISODE_META = "meta.json"
CAMERA_FOLDERS = (
    "front_rgb",
    "wrist_rgb",
    "left_shoulder_rgb",
    "right_shoulder_rgb",
    "overhead_rgb",
)


class CompatibleUnpickler(pickle.Unpickler):
    """Handle numpy module path differences in older pickles."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core":
            module = "numpy.core"
        elif module.startswith("numpy._core."):
            module = module.replace("numpy._core.", "numpy.core.", 1)
        return super().find_class(module, name)


def install_rlbench_pickle_support(rlbench_root: Path) -> None:
    """Install minimal rlbench modules so pickle can load Demo/Observation."""
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
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "RLBench_Raw_Dataset")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--rlbench-root", type=Path, default=DEFAULT_RLBENCH_ROOT,
        help="RLBench source root for unpickling Demo objects.",
    )
    parser.add_argument(
        "--visual-z-threshold", type=float, default=6.0,
        help="Robust z-score threshold on first-frame deviation for visual outliers.",
    )
    parser.add_argument(
        "--contact-sheet-dir", type=Path, default=None,
        help="Optional directory to write per-task first-frame contact sheets.",
    )
    return parser.parse_args()


def episode_dirs_of(variation_dir: Path) -> list[Path]:
    episodes_root = variation_dir / "episodes"
    if not episodes_root.is_dir():
        return []
    return sorted(
        (p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode")),
        key=lambda p: int(p.name[len("episode"):]),
    )


def check_meta(episode_dir: Path, task: str, variation_index: int) -> list[str]:
    problems: list[str] = []
    meta_path = episode_dir / EPISODE_META
    if not meta_path.is_file():
        return [f"{episode_dir}: missing {EPISODE_META}"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{episode_dir}: unreadable {EPISODE_META}: {exc}"]
    for key in ("task_name", "scene_task_name"):
        if meta.get(key) != task:
            problems.append(
                f"{episode_dir}: meta {key}={meta.get(key)!r} != directory task {task!r}"
            )
    if meta.get("variation") != variation_index:
        problems.append(
            f"{episode_dir}: meta variation={meta.get('variation')!r} != {variation_index}"
        )
    return problems


def check_structure(episode_dir: Path) -> tuple[list[str], int | None, int | None]:
    """Return (problems, demo step count, task_low_dim_state dim)."""
    problems: list[str] = []
    low_dim_path = episode_dir / LOW_DIM_PICKLE
    if not low_dim_path.is_file():
        return [f"{episode_dir}: missing {LOW_DIM_PICKLE}"], None, None
    try:
        with low_dim_path.open("rb") as handle:
            demo = CompatibleUnpickler(handle).load()
    except Exception as exc:
        return [f"{episode_dir}: failed to unpickle {LOW_DIM_PICKLE}: {exc}"], None, None

    step_count = len(demo)
    if step_count < 2:
        problems.append(f"{episode_dir}: demo has only {step_count} steps")

    low_dim_dim: int | None = None
    state = getattr(demo[0], "task_low_dim_state", None)
    if state is not None:
        low_dim_dim = int(np.asarray(state).size)

    for folder in CAMERA_FOLDERS:
        camera_dir = episode_dir / folder
        if not camera_dir.is_dir():
            problems.append(f"{episode_dir}: missing {folder}")
            continue
        png_count = sum(1 for _ in camera_dir.glob("*.png"))
        if png_count != step_count:
            problems.append(
                f"{episode_dir}: {folder} has {png_count} frames, expected {step_count}"
            )
    return problems, step_count, low_dim_dim


def first_frame_signature(episode_dir: Path) -> np.ndarray | None:
    image_path = episode_dir / "front_rgb" / "0.png"
    if not image_path.is_file():
        return None
    with Image.open(image_path) as image:
        small = image.convert("L").resize((32, 32), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32)


def visual_outliers(
    signatures: dict[str, np.ndarray], z_threshold: float
) -> list[tuple[str, float]]:
    """Flag episodes whose first frame deviates abnormally from the group median."""
    if len(signatures) < 4:
        return []
    stack = np.stack(list(signatures.values()), axis=0)
    median_image = np.median(stack, axis=0)
    deviations = np.asarray(
        [float(np.mean(np.abs(sig - median_image))) for sig in signatures.values()]
    )
    med = float(np.median(deviations))
    mad = float(np.median(np.abs(deviations - med))) or 1e-6
    flagged = []
    for key, dev in zip(signatures.keys(), deviations):
        z = 0.6745 * (dev - med) / mad
        if z > z_threshold:
            flagged.append((key, round(float(z), 1)))
    return flagged


def write_contact_sheet(
    signatures_rgb: dict[str, Path], out_path: Path, tile: int = 112, columns: int = 10
) -> None:
    keys = list(signatures_rgb.keys())
    rows = (len(keys) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * tile), (30, 30, 30))
    for index, key in enumerate(keys):
        with Image.open(signatures_rgb[key]) as image:
            thumb = image.convert("RGB").resize((tile, tile), Image.BILINEAR)
        sheet.paste(thumb, ((index % columns) * tile, (index // columns) * tile))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    args = parse_args()
    split_root = args.root / args.split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_root}")
    install_rlbench_pickle_support(args.rlbench_root)

    tasks = sorted(p.name for p in split_root.iterdir() if p.is_dir())
    all_problems: list[str] = []
    episode_total = 0
    dim_by_task: dict[str, dict[int, int]] = {}

    for task in tasks:
        task_dir = split_root / task
        variation_dirs = sorted(
            (p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("variation")),
            key=lambda p: int(p.name[len("variation"):]),
        )
        task_signatures_by_variation: dict[str, dict[str, np.ndarray]] = {}
        contact_paths: dict[str, Path] = {}
        dim_counts: dict[int, int] = {}

        for variation_dir in variation_dirs:
            variation_index = int(variation_dir.name[len("variation"):])
            signatures: dict[str, np.ndarray] = {}
            for episode_dir in episode_dirs_of(variation_dir):
                episode_total += 1
                key = f"{task}/{variation_dir.name}/{episode_dir.name}"

                all_problems.extend(check_meta(episode_dir, task, variation_index))
                problems, _, low_dim_dim = check_structure(episode_dir)
                all_problems.extend(problems)
                if low_dim_dim is not None:
                    dim_counts[low_dim_dim] = dim_counts.get(low_dim_dim, 0) + 1

                signature = first_frame_signature(episode_dir)
                if signature is None:
                    all_problems.append(f"{episode_dir}: missing front_rgb/0.png")
                else:
                    signatures[key] = signature
                    contact_paths[key] = episode_dir / "front_rgb" / "0.png"
            task_signatures_by_variation[variation_dir.name] = signatures

        dim_by_task[task] = dim_counts
        if len(dim_counts) > 1:
            all_problems.append(
                f"{task}: inconsistent task_low_dim_state dims across episodes: {dim_counts}"
            )

        for variation_name, signatures in task_signatures_by_variation.items():
            for key, z in visual_outliers(signatures, args.visual_z_threshold):
                all_problems.append(
                    f"{key}: first-frame visual outlier (robust z={z}) vs {variation_name} median"
                )

        if args.contact_sheet_dir is not None and contact_paths:
            write_contact_sheet(
                contact_paths,
                args.contact_sheet_dir / f"{args.split}_{task}.png",
            )

    print(f"[verify] split={args.split} tasks={len(tasks)} episodes={episode_total}")
    for task in tasks:
        print(f"[verify] task={task} low_dim_state_dims={dim_by_task[task]}")
    if all_problems:
        print(f"\n[FAIL] {len(all_problems)} problem(s) found:")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1
    print("[PASS] dataset is clean: identity, structure, dims, and visuals all consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
