#!/usr/bin/env python3
"""Convert raw RLBench full trajectories into LeRobot datasets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RAW_ROOT = REPO_ROOT / "RLBench_Raw_Dataset"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "LeRobot_RLBench_Dataset"
DEFAULT_RLBENCH_ROOT = REPO_ROOT / "source" / "RLBench"
MANIFEST_NAME = "manifest.json"
LOW_DIM_PICKLE = "low_dim_obs.pkl"
VARIATION_DESCRIPTIONS = "variation_descriptions.pkl"
EPISODES_DIR = "episodes"

RAW_TO_LEROBOT_CAMERAS = {
    "front_rgb": "observation.images.front",
    "wrist_rgb": "observation.images.wrist",
    "left_shoulder_rgb": "observation.images.left_shoulder",
    "right_shoulder_rgb": "observation.images.right_shoulder",
    "overhead_rgb": "observation.images.overhead",
}


@dataclass(frozen=True)
class EpisodeSpec:
    """Metadata needed to convert one raw episode."""

    task: str
    variation: str
    episode: str
    episode_dir: Path
    instruction: str


class CompatibleUnpickler(pickle.Unpickler):
    """Handle a few module path differences in older pickles."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core":
            module = "numpy.core"
        elif module.startswith("numpy._core."):
            module = module.replace("numpy._core.", "numpy.core.", 1)
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(
        description="Convert raw RLBench trajectories into LeRobot datasets."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory of the raw RLBench dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for the LeRobot datasets.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        required=True,
        help="Dataset split to convert.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of RLBench task ids to convert.",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Optional cap on converted episodes per task.",
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=("longest", "first", "shortest"),
        default="longest",
        help="How to pick one instruction from variation_descriptions.pkl.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS written into the LeRobot metadata.",
    )
    parser.add_argument(
        "--robot-type",
        default="panda",
        help="Robot type written into the LeRobot metadata.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=5,
        help="Number of async image writer processes.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="Number of async image writer threads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output split directory if it already exists.",
    )
    args = parser.parse_args()

    if args.max_episodes_per_task is not None and args.max_episodes_per_task < 1:
        raise ValueError("--max-episodes-per-task must be >= 1.")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1.")
    if args.image_writer_processes < 0:
        raise ValueError("--image-writer-processes must be >= 0.")
    if args.image_writer_threads < 0:
        raise ValueError("--image-writer-threads must be >= 0.")

    return args


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_numeric_suffix(name: str, prefix: str) -> int:
    """Extract the integer suffix from a path component."""
    if not name.startswith(prefix):
        raise ValueError(f"{name!r} does not start with {prefix!r}.")
    return int(name[len(prefix) :])


def discover_tasks(
    split_root: Path,
    manifest: dict[str, Any] | None,
    requested_tasks: list[str] | None,
) -> list[str]:
    """Resolve which tasks should be converted."""
    if requested_tasks:
        tasks = requested_tasks
    elif manifest is not None:
        split_info = manifest.get("splits", {}).get(split_root.name, {})
        tasks = list(split_info.get("tasks", {}).keys()) or manifest.get("task_list", [])
    else:
        tasks = []

    if not tasks:
        tasks = sorted(path.name for path in split_root.iterdir() if path.is_dir())

    missing = [task for task in tasks if not (split_root / task).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Tasks not found under {split_root}: {', '.join(missing)}"
        )
    return tasks


def pick_instruction(descriptions: list[str], strategy: str) -> str:
    """Pick one stable instruction string for the whole episode."""
    cleaned = [text.strip() for text in descriptions if text and text.strip()]
    if not cleaned:
        raise ValueError("variation_descriptions.pkl is empty.")
    if strategy == "first":
        return cleaned[0]
    if strategy == "shortest":
        return min(cleaned, key=len)
    return max(cleaned, key=len)


def load_instruction(variation_dir: Path, strategy: str) -> str:
    """Read one language instruction from a variation directory."""
    desc_path = variation_dir / VARIATION_DESCRIPTIONS
    with desc_path.open("rb") as handle:
        descriptions = pickle.load(handle)
    if not isinstance(descriptions, list):
        raise TypeError(f"{desc_path} should contain a list of strings.")
    return pick_instruction(descriptions, strategy)


def build_episode_specs(
    split_root: Path,
    tasks: list[str],
    prompt_strategy: str,
    max_episodes_per_task: int | None,
) -> list[EpisodeSpec]:
    """Collect episode paths to convert."""
    specs: list[EpisodeSpec] = []

    for task in tasks:
        task_dir = split_root / task
        variation_dirs = sorted(
            (path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith("variation")),
            key=lambda path: get_numeric_suffix(path.name, "variation"),
        )
        converted_for_task = 0

        for variation_dir in variation_dirs:
            instruction = load_instruction(variation_dir, prompt_strategy)
            episodes_root = variation_dir / EPISODES_DIR
            if not episodes_root.is_dir():
                continue

            episode_dirs = sorted(
                (path for path in episodes_root.iterdir() if path.is_dir() and path.name.startswith("episode")),
                key=lambda path: get_numeric_suffix(path.name, "episode"),
            )
            for episode_dir in episode_dirs:
                if (
                    max_episodes_per_task is not None
                    and converted_for_task >= max_episodes_per_task
                ):
                    break

                specs.append(
                    EpisodeSpec(
                        task=task,
                        variation=variation_dir.name,
                        episode=episode_dir.name,
                        episode_dir=episode_dir,
                        instruction=instruction,
                    )
                )
                converted_for_task += 1

            if (
                max_episodes_per_task is not None
                and converted_for_task >= max_episodes_per_task
            ):
                break

    return specs


def resolve_rlbench_root(raw_root: Path, manifest: dict[str, Any] | None) -> Path:
    """Resolve the RLBench source root used for pickle support."""
    manifest_root = None
    if manifest is not None:
        manifest_root = manifest.get("rlbench_root")
    rlbench_root = Path(manifest_root) if manifest_root else DEFAULT_RLBENCH_ROOT
    if not rlbench_root.is_dir():
        raise FileNotFoundError(f"RLBench source root not found: {rlbench_root}")
    return rlbench_root


def install_rlbench_pickle_support(rlbench_root: Path) -> None:
    """Install minimal rlbench modules so pickle can load Demo/Observation."""
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
        spec = importlib.util.spec_from_file_location(
            "rlbench.backend.observation",
            observation_py,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {observation_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["rlbench.backend.observation"] = module
        spec.loader.exec_module(module)


def load_demo(path: Path) -> Any:
    """Load one raw RLBench low_dim_obs.pkl file."""
    with path.open("rb") as handle:
        return CompatibleUnpickler(handle).load()


def normalize_quaternion_sign(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Fix quaternion sign ambiguity by enforcing qw >= 0."""
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float32)
    if quaternion_xyzw.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quaternion_xyzw.shape}")
    if quaternion_xyzw[3] < 0:
        quaternion_xyzw = -quaternion_xyzw
    return quaternion_xyzw


def observation_to_state(observation: Any) -> np.ndarray:
    """Convert one RLBench observation into the 8D state/action vector."""
    pose = np.asarray(observation.gripper_pose, dtype=np.float32)
    if pose.shape != (7,):
        raise ValueError(f"Expected gripper_pose shape (7,), got {pose.shape}")

    xyz = pose[:3]
    quaternion_xyzw = normalize_quaternion_sign(pose[3:7])
    gripper = np.asarray([float(observation.gripper_open)], dtype=np.float32)
    return np.concatenate([xyz, quaternion_xyzw, gripper], axis=0).astype(np.float32)


def load_png_paths(camera_dir: Path, expected_frames: int) -> list[Path]:
    """Collect and validate PNG frame paths for one camera."""
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Camera folder not found: {camera_dir}")

    png_paths = sorted(
        camera_dir.glob("*.png"),
        key=lambda path: int(path.stem),
    )
    if len(png_paths) != expected_frames:
        raise ValueError(
            f"{camera_dir} has {len(png_paths)} frames, expected {expected_frames}."
        )

    for frame_index, image_path in enumerate(png_paths):
        if int(image_path.stem) != frame_index:
            raise ValueError(
                f"{camera_dir} is missing frame {frame_index} or has non-sequential names."
            )
    return png_paths


def load_image(image_path: Path) -> np.ndarray:
    """Read one RGB image as uint8 HWC."""
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def infer_image_shape(specs: list[EpisodeSpec]) -> tuple[int, int, int]:
    """Infer the image shape from the first readable frame."""
    for spec in specs:
        first_image = spec.episode_dir / "front_rgb" / "0.png"
        if first_image.is_file():
            return load_image(first_image).shape
    raise FileNotFoundError("Failed to infer image shape from the raw dataset.")


def create_lerobot_dataset(
    repo_id: str,
    dataset_root: Path,
    image_shape: tuple[int, int, int],
    fps: int,
    robot_type: str,
    image_writer_processes: int,
    image_writer_threads: int,
) -> Any:
    """Create an empty LeRobot dataset for one split."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    height, width, channels = image_shape
    if channels != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {image_shape}")

    features: dict[str, dict[str, Any]] = {
        feature_key: {
            "dtype": "image",
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        }
        for feature_key in RAW_TO_LEROBOT_CAMERAS.values()
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (8,),
        "names": ["action"],
    }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=robot_type,
        fps=fps,
        features=features,
        use_videos=False,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def convert_episode(dataset: Any, spec: EpisodeSpec) -> int:
    """Convert one raw episode and return the number of written frames."""
    low_dim_path = spec.episode_dir / LOW_DIM_PICKLE
    demo = load_demo(low_dim_path)
    num_observations = len(demo)
    if num_observations < 2:
        raise ValueError(f"{low_dim_path} has only {num_observations} observations.")

    camera_frames = {
        feature_key: load_png_paths(
            spec.episode_dir / raw_folder,
            expected_frames=num_observations,
        )
        for raw_folder, feature_key in RAW_TO_LEROBOT_CAMERAS.items()
    }

    for step_index in range(num_observations - 1):
        frame = {
            feature_key: load_image(paths[step_index])
            for feature_key, paths in camera_frames.items()
        }
        frame["observation.state"] = observation_to_state(demo[step_index])
        frame["action"] = observation_to_state(demo[step_index + 1])
        frame["task"] = spec.instruction
        dataset.add_frame(frame)

    dataset.save_episode()
    return num_observations - 1


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Prepare the output directory for one split."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite to rebuild it."
            )
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Entry point."""
    args = parse_args()
    raw_root = args.raw_root.resolve()
    split_root = raw_root / args.split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    manifest = load_json(raw_root / MANIFEST_NAME)
    tasks = discover_tasks(split_root, manifest, args.tasks)
    specs = build_episode_specs(
        split_root=split_root,
        tasks=tasks,
        prompt_strategy=args.prompt_strategy,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    if not specs:
        raise RuntimeError(f"No episodes found under {split_root}")

    output_root = args.output_root.resolve()
    dataset_root = output_root / args.split
    prepare_output_dir(dataset_root, args.overwrite)

    rlbench_root = resolve_rlbench_root(raw_root, manifest)
    install_rlbench_pickle_support(rlbench_root)

    dataset = create_lerobot_dataset(
        repo_id=args.split,
        dataset_root=dataset_root,
        image_shape=infer_image_shape(specs),
        fps=args.fps,
        robot_type=args.robot_type,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    converted_episodes = 0
    converted_frames = 0
    skipped_episodes = 0
    per_task_counts = {task: 0 for task in tasks}

    for spec in specs:
        try:
            episode_frames = convert_episode(dataset, spec)
        except Exception as exc:  # pragma: no cover - data-dependent
            skipped_episodes += 1
            if hasattr(dataset, "clear_episode_buffer"):
                dataset.clear_episode_buffer()
            print(
                "[warn] skip"
                f" split={args.split}"
                f" task={spec.task}"
                f" variation={spec.variation}"
                f" episode={spec.episode}"
                f" reason={exc}",
                file=sys.stderr,
            )
            continue

        converted_episodes += 1
        converted_frames += episode_frames
        per_task_counts[spec.task] += 1

        if converted_episodes % 20 == 0:
            print(
                f"[{args.split}] converted_episodes={converted_episodes} "
                f"converted_frames={converted_frames} skipped={skipped_episodes}"
            )

    if getattr(dataset, "image_writer", None) is not None:
        dataset.stop_image_writer()

    print(
        f"[done] split={args.split} output={dataset_root} "
        f"episodes={converted_episodes} frames={converted_frames} "
        f"skipped={skipped_episodes}"
    )
    for task, count in per_task_counts.items():
        print(f"[task] split={args.split} task={task} episodes={count}")

    print(
        "For later openpi training, set "
        f"HF_LEROBOT_HOME={output_root} and use repo_id={args.split}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
