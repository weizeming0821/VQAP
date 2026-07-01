#!/usr/bin/env python3
"""Convert raw RLBench full trajectories into LeRobot datasets."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
import pickle
import shutil
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

try:
    import datasets as hf_datasets
except ImportError:  # pragma: no cover - optional dependency
    hf_datasets = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RAW_ROOT = REPO_ROOT / "RLBench_Raw_Dataset"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "LeRobot_RLBench_Dataset"
DEFAULT_RLBENCH_ROOT = REPO_ROOT / "source" / "RLBench"
MANIFEST_NAME = "manifest.json"
LOW_DIM_PICKLE = "low_dim_obs.pkl"
VARIATION_DESCRIPTIONS = "variation_descriptions.pkl"
EPISODES_DIR = "episodes"
DEFAULT_NUM_WORKERS = 1
PROGRESS_LOG_NAME = ".conversion_progress.jsonl"
PENDING_STATE_NAME = ".conversion_pending.json"
LOCK_NAME_TEMPLATE = ".{split}.convert.lock"

RAW_TO_LEROBOT_CAMERAS = {
    "front_rgb": "observation.images.front",
    "wrist_rgb": "observation.images.wrist",
    "left_shoulder_rgb": "observation.images.left_shoulder",
    "right_shoulder_rgb": "observation.images.right_shoulder",
    "overhead_rgb": "observation.images.overhead",
}


_WORKER_RLBENCH_ROOT: Path | None = None


@dataclass(frozen=True)
class EpisodeSpec:
    """Metadata needed to convert one raw episode."""

    task: str
    variation: str
    episode: str
    episode_dir: Path
    instruction: str


@dataclass(frozen=True)
class PreparedEpisode:
    """CPU-side payload prepared before serial LeRobot writing."""

    spec: EpisodeSpec
    num_frames: int
    image_paths: dict[str, tuple[str, ...]]
    states: np.ndarray
    actions: np.ndarray


class CompatibleUnpickler(pickle.Unpickler):
    """Handle a few module path differences in older pickles."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core":
            module = "numpy.core"
        elif module.startswith("numpy._core."):
            module = module.replace("numpy._core.", "numpy.core.", 1)
        return super().find_class(module, name)


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Number of worker processes for episode preparation. "
            "LeRobot writing stays single-writer to keep the dataset consistent."
        ),
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
        "--resume",
        action="store_true",
        help="Resume from an existing split output if a progress log is available.",
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
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1.")
    if args.image_writer_processes < 0:
        raise ValueError("--image-writer-processes must be >= 0.")
    if args.image_writer_threads < 0:
        raise ValueError("--image-writer-threads must be >= 0.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")

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


def spec_key(spec: EpisodeSpec) -> str:
    """Build a stable raw episode id for resume tracking."""
    return f"{spec.task}/{spec.variation}/{spec.episode}"


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
            (
                path
                for path in task_dir.iterdir()
                if path.is_dir() and path.name.startswith("variation")
            ),
            key=lambda path: get_numeric_suffix(path.name, "variation"),
        )
        converted_for_task = 0

        for variation_dir in variation_dirs:
            instruction = load_instruction(variation_dir, prompt_strategy)
            episodes_root = variation_dir / EPISODES_DIR
            if not episodes_root.is_dir():
                continue

            episode_dirs = sorted(
                (
                    path
                    for path in episodes_root.iterdir()
                    if path.is_dir() and path.name.startswith("episode")
                ),
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


def init_prepare_worker(rlbench_root: str) -> None:
    """Initialize a preparation worker process."""
    global _WORKER_RLBENCH_ROOT
    _WORKER_RLBENCH_ROOT = Path(rlbench_root)
    install_rlbench_pickle_support(_WORKER_RLBENCH_ROOT)


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


def load_png_paths(camera_dir: Path, expected_frames: int) -> tuple[str, ...]:
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
    return tuple(str(path) for path in png_paths)


def load_image(image_path: str | Path) -> np.ndarray:
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


def expected_features(
    image_shape: tuple[int, int, int],
) -> dict[str, dict[str, Any]]:
    """Build the expected LeRobot feature schema."""
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
    return features


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

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=robot_type,
        fps=fps,
        features=expected_features(image_shape),
        use_videos=False,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def cleanup_stale_images(dataset_root: Path) -> None:
    """Remove temporary image buffers left behind by interrupted runs."""
    images_dir = dataset_root / "images"
    if images_dir.is_dir():
        shutil.rmtree(images_dir)


def dataset_meta_exists(dataset_root: Path) -> bool:
    """Check whether a LeRobot dataset has already been created here."""
    return (dataset_root / "meta" / "info.json").is_file()


def append_resume_record(path: Path, record: dict[str, Any]) -> None:
    """Append one successful conversion record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def load_pending_state(path: Path) -> dict[str, Any] | None:
    """Load the current in-flight episode transaction, if any."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def remove_pending_state(path: Path) -> None:
    """Remove the in-flight transaction marker."""
    if path.exists():
        path.unlink()


def load_resume_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load successful conversion records from previous runs."""
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[warn] ignore malformed resume record at line {line_number}: {path}",
                    file=sys.stderr,
                )
                break
            key = record.get("raw_episode_key")
            if not key:
                raise RuntimeError(
                    f"Progress log record at line {line_number} is missing raw_episode_key."
                )
            if key in records:
                raise RuntimeError(
                    f"Duplicate raw_episode_key found in progress log: {key}"
                )
            if "output_episode_index" not in record:
                raise RuntimeError(
                    f"Progress log record at line {line_number} is missing output_episode_index."
                )
            records[key] = record
    return records


def count_parquet_files(dataset_root: Path) -> int:
    """Count parquet episode files on disk."""
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        return 0
    return sum(1 for _ in data_root.rglob("*.parquet"))


def validate_commit_records(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate commit records and return them sorted by episode index."""
    ordered = sorted(
        records.values(),
        key=lambda record: int(record["output_episode_index"]),
    )
    seen_indices: set[int] = set()
    for expected_index, record in enumerate(ordered):
        output_index = int(record["output_episode_index"])
        if output_index in seen_indices:
            raise RuntimeError(f"Duplicate output_episode_index in progress log: {output_index}")
        if output_index != expected_index:
            raise RuntimeError(
                "Progress log episode indices are not contiguous from 0. "
                f"Expected {expected_index}, got {output_index}."
            )
        seen_indices.add(output_index)
    return ordered


def audit_dataset_state(
    repo_id: str,
    dataset_root: Path,
    commit_records: dict[str, dict[str, Any]],
    pending_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate that dataset files, metadata, and progress bookkeeping agree."""
    if not dataset_meta_exists(dataset_root):
        if commit_records or pending_state is not None:
            raise RuntimeError(
                "Progress bookkeeping exists but the LeRobot dataset metadata is missing."
            )
        return {
            "meta_total_episodes": 0,
            "meta_total_frames": 0,
            "ordered_commits": [],
            "episode_lengths": {},
        }

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    ordered_commits = validate_commit_records(commit_records)
    parquet_count = count_parquet_files(dataset_root)

    if parquet_count != meta.total_episodes:
        raise RuntimeError(
            f"Parquet file count ({parquet_count}) does not match metadata total_episodes "
            f"({meta.total_episodes})."
        )
    if len(meta.episodes) != meta.total_episodes:
        raise RuntimeError(
            f"episodes.jsonl count ({len(meta.episodes)}) does not match metadata total_episodes "
            f"({meta.total_episodes})."
        )

    episode_lengths = {
        int(episode_index): int(episode_dict["length"])
        for episode_index, episode_dict in meta.episodes.items()
    }
    if set(episode_lengths) != set(range(meta.total_episodes)):
        raise RuntimeError("Episode indices in metadata are not contiguous from 0.")

    expected_frames_from_meta = sum(episode_lengths.values())
    if meta.total_frames != expected_frames_from_meta:
        raise RuntimeError(
            f"Metadata total_frames ({meta.total_frames}) does not match summed episode lengths "
            f"({expected_frames_from_meta})."
        )

    commit_count = len(ordered_commits)
    if pending_state is None:
        if meta.total_episodes != commit_count:
            raise RuntimeError(
                f"Metadata total_episodes ({meta.total_episodes}) does not match committed progress "
                f"records ({commit_count})."
            )
        if ordered_commits and all(record.get("frames") is not None for record in ordered_commits):
            commit_frames = sum(int(record["frames"]) for record in ordered_commits)
            if meta.total_frames != commit_frames:
                raise RuntimeError(
                    f"Metadata total_frames ({meta.total_frames}) does not match committed progress "
                    f"frames ({commit_frames})."
                )
    else:
        expected_output_index = int(pending_state["output_episode_index"])
        if commit_count != expected_output_index:
            raise RuntimeError(
                "Pending state does not line up with the committed progress log: "
                f"commit_count={commit_count}, pending.output_episode_index={expected_output_index}."
            )
        if meta.total_episodes not in (expected_output_index, expected_output_index + 1):
            raise RuntimeError(
                "Pending state exists, but metadata total_episodes is inconsistent with it: "
                f"{meta.total_episodes} not in {{{expected_output_index}, {expected_output_index + 1}}}."
            )
        if meta.total_episodes == expected_output_index + 1:
            pending_frames = pending_state.get("frames")
            last_length = episode_lengths[expected_output_index]
            if pending_frames is not None and int(pending_frames) != last_length:
                raise RuntimeError(
                    "Recovered pending episode length does not match metadata: "
                    f"{pending_frames} != {last_length}."
                )

    return {
        "meta_total_episodes": meta.total_episodes,
        "meta_total_frames": meta.total_frames,
        "ordered_commits": ordered_commits,
        "episode_lengths": episode_lengths,
    }


def build_commit_record(
    *,
    spec: EpisodeSpec,
    frames: int | None,
    output_episode_index: int,
    finished_at: str | None,
    recovered_from_pending: bool = False,
) -> dict[str, Any]:
    """Build one canonical progress record."""
    record: dict[str, Any] = {
        "raw_episode_key": spec_key(spec),
        "task": spec.task,
        "variation": spec.variation,
        "episode": spec.episode,
        "frames": frames,
        "output_episode_index": output_episode_index,
        "finished_at": finished_at,
    }
    if recovered_from_pending:
        record["recovered_from_pending"] = True
    return record


def validate_records_match_plan(
    specs: list[EpisodeSpec],
    commit_records: dict[str, dict[str, Any]],
) -> None:
    """Ensure the existing dataset/progress log matches the requested conversion plan."""
    allowed_keys = {spec_key(spec) for spec in specs}
    extra_keys = sorted(set(commit_records) - allowed_keys)
    if extra_keys:
        preview = ", ".join(extra_keys[:5])
        raise RuntimeError(
            "Existing dataset/progress log contains episodes that are not part of the current "
            f"requested plan, e.g. {preview}. Use matching arguments or restart with --overwrite."
        )


def recover_pending_commit_if_needed(
    *,
    repo_id: str,
    dataset_root: Path,
    progress_path: Path,
    pending_path: Path,
    pending_state: dict[str, Any] | None,
    commit_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Repair or clear one interrupted episode transaction."""
    if pending_state is None:
        return commit_records, None

    audit = audit_dataset_state(repo_id, dataset_root, commit_records, pending_state)
    expected_output_index = int(pending_state["output_episode_index"])
    meta_total_episodes = int(audit["meta_total_episodes"])
    spec = EpisodeSpec(
        task=pending_state["task"],
        variation=pending_state["variation"],
        episode=pending_state["episode"],
        episode_dir=Path(pending_state["episode_dir"]),
        instruction=pending_state["instruction"],
    )

    if meta_total_episodes == expected_output_index:
        print(
            f"[resume] clearing unfinished pending episode {pending_state['raw_episode_key']} "
            "because it was not committed into the dataset."
        )
        remove_pending_state(pending_path)
        return commit_records, None

    recovered_frames = audit["episode_lengths"][expected_output_index]
    record = build_commit_record(
        spec=spec,
        frames=recovered_frames,
        output_episode_index=expected_output_index,
        finished_at=utc_now_iso(),
        recovered_from_pending=True,
    )
    append_resume_record(progress_path, record)
    remove_pending_state(pending_path)
    updated_records = dict(commit_records)
    updated_records[record["raw_episode_key"]] = record
    print(
        f"[resume] recovered committed episode {record['raw_episode_key']} "
        "from the pending transaction marker."
    )
    return updated_records, None


def verify_existing_dataset(
    repo_id: str,
    dataset_root: Path,
    image_shape: tuple[int, int, int],
    fps: int,
    robot_type: str,
) -> None:
    """Check that resume is targeting a compatible existing dataset."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    if meta.fps != fps:
        raise RuntimeError(
            f"Existing dataset fps={meta.fps} does not match requested fps={fps}."
        )
    if meta.robot_type != robot_type:
        raise RuntimeError(
            "Existing dataset robot_type does not match the requested value: "
            f"{meta.robot_type!r} != {robot_type!r}"
        )

    expected = expected_features(image_shape)
    for key, feature in expected.items():
        existing = meta.features.get(key)
        if existing is None:
            raise RuntimeError(f"Existing dataset is missing feature {key!r}.")
        if tuple(existing["shape"]) != tuple(feature["shape"]):
            raise RuntimeError(
                f"Existing dataset feature {key!r} has shape {existing['shape']}, "
                f"expected {feature['shape']}."
            )
        if existing["dtype"] != feature["dtype"]:
            raise RuntimeError(
                f"Existing dataset feature {key!r} has dtype {existing['dtype']}, "
                f"expected {feature['dtype']}."
            )


def load_or_create_dataset(
    repo_id: str,
    dataset_root: Path,
    image_shape: tuple[int, int, int],
    fps: int,
    robot_type: str,
    image_writer_processes: int,
    image_writer_threads: int,
    resume: bool,
) -> Any:
    """Open an existing dataset for resume or create a fresh one."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    cleanup_stale_images(dataset_root)

    if resume and dataset_meta_exists(dataset_root):
        verify_existing_dataset(repo_id, dataset_root, image_shape, fps, robot_type)
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        if image_writer_processes or image_writer_threads:
            dataset.start_image_writer(
                num_processes=image_writer_processes,
                num_threads=image_writer_threads,
            )
        return dataset

    return create_lerobot_dataset(
        repo_id=repo_id,
        dataset_root=dataset_root,
        image_shape=image_shape,
        fps=fps,
        robot_type=robot_type,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def prepare_episode(spec: EpisodeSpec) -> PreparedEpisode:
    """Prepare one episode in a worker-friendly form."""
    low_dim_path = spec.episode_dir / LOW_DIM_PICKLE
    demo = load_demo(low_dim_path)
    num_observations = len(demo)
    if num_observations < 2:
        raise ValueError(f"{low_dim_path} has only {num_observations} observations.")

    image_paths = {
        feature_key: load_png_paths(
            spec.episode_dir / raw_folder,
            expected_frames=num_observations,
        )
        for raw_folder, feature_key in RAW_TO_LEROBOT_CAMERAS.items()
    }

    states = np.stack(
        [observation_to_state(demo[index]) for index in range(num_observations - 1)],
        axis=0,
    )
    actions = np.stack(
        [observation_to_state(demo[index + 1]) for index in range(num_observations - 1)],
        axis=0,
    )

    return PreparedEpisode(
        spec=spec,
        num_frames=num_observations - 1,
        image_paths=image_paths,
        states=states,
        actions=actions,
    )


def iter_prepared_episodes(
    specs: list[EpisodeSpec],
    num_workers: int,
    rlbench_root: Path,
) -> Iterator[tuple[EpisodeSpec, PreparedEpisode | Exception]]:
    """Yield prepared episodes in deterministic order."""
    if num_workers <= 1:
        for spec in specs:
            try:
                yield spec, prepare_episode(spec)
            except Exception as exc:  # pragma: no cover - data-dependent
                yield spec, exc
        return

    max_pending = max(num_workers * 2, 1)
    spec_iter = iter(specs)
    futures: deque[tuple[EpisodeSpec, Future[PreparedEpisode]]] = deque()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_prepare_worker,
        initargs=(str(rlbench_root),),
    ) as executor:
        while len(futures) < max_pending:
            try:
                spec = next(spec_iter)
            except StopIteration:
                break
            futures.append((spec, executor.submit(prepare_episode, spec)))

        while futures:
            spec, future = futures.popleft()
            try:
                yield spec, future.result()
            except Exception as exc:  # pragma: no cover - worker/data-dependent
                yield spec, exc

            try:
                next_spec = next(spec_iter)
            except StopIteration:
                continue
            futures.append((next_spec, executor.submit(prepare_episode, next_spec)))


def write_prepared_episode(dataset: Any, prepared: PreparedEpisode) -> int:
    """Write one prepared episode into the LeRobot dataset."""
    for step_index in range(prepared.num_frames):
        frame = {
            feature_key: load_image(paths[step_index])
            for feature_key, paths in prepared.image_paths.items()
        }
        frame["observation.state"] = prepared.states[step_index]
        frame["action"] = prepared.actions[step_index]
        frame["task"] = prepared.spec.instruction
        dataset.add_frame(frame)

    dataset.save_episode()
    return prepared.num_frames


def prepare_output_dir(path: Path, overwrite: bool, resume: bool) -> None:
    """Prepare the output directory for one split."""
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite to rebuild it or --resume to continue."
            )
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def split_lock(lock_path: Path) -> Iterator[None]:
    """Prevent concurrent top-level conversion runs for the same split."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another conversion run is active for this split: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class SimpleProgress:
    """Fallback progress helper when tqdm is unavailable."""

    def __init__(self, total: int, desc: str, initial: int = 0):
        self.total = total
        self.desc = desc
        self.current = initial
        if total > 0:
            print(f"[{desc}] {self.current}/{self.total}")

    def set_postfix(self, **_: Any) -> None:
        return None

    def update(self, value: int = 1) -> None:
        self.current += value
        if self.total > 0:
            print(f"[{self.desc}] {self.current}/{self.total}")

    def close(self) -> None:
        return None


def make_progress(total: int, desc: str, initial: int = 0) -> Any:
    """Create a tqdm bar when possible, otherwise use a simple fallback."""
    if tqdm is None:
        return SimpleProgress(total=total, desc=desc, initial=initial)
    return tqdm(total=total, desc=desc, initial=initial, dynamic_ncols=True)


def disable_nested_progress_bars() -> None:
    """Keep the terminal focused on the converter-level progress bar."""
    if hf_datasets is None:
        return
    disable_fn = getattr(hf_datasets, "disable_progress_bar", None)
    if callable(disable_fn):
        disable_fn()


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
    progress_path = dataset_root / PROGRESS_LOG_NAME
    lock_path = output_root / LOCK_NAME_TEMPLATE.format(split=args.split)
    repo_id = args.split

    with split_lock(lock_path):
        disable_nested_progress_bars()
        prepare_output_dir(dataset_root, args.overwrite, args.resume)

        rlbench_root = resolve_rlbench_root(raw_root, manifest)
        install_rlbench_pickle_support(rlbench_root)

        completed_records: dict[str, dict[str, Any]] = {}
        pending_path = dataset_root / PENDING_STATE_NAME
        if args.resume and dataset_meta_exists(dataset_root):
            if not progress_path.exists():
                raise RuntimeError(
                    "This dataset was created before transactional resume support and has no "
                    f"{PROGRESS_LOG_NAME}. Use --overwrite to rebuild it cleanly."
                )
            completed_records = load_resume_records(progress_path)
            if not completed_records:
                print(
                    f"[resume] {progress_path.name} exists but no valid records were loaded.",
                    file=sys.stderr,
                )
            pending_state = load_pending_state(pending_path)
            completed_records, _ = recover_pending_commit_if_needed(
                repo_id=repo_id,
                dataset_root=dataset_root,
                progress_path=progress_path,
                pending_path=pending_path,
                pending_state=pending_state,
                commit_records=completed_records,
            )
            validate_records_match_plan(specs, completed_records)
            audit_dataset_state(
                repo_id=repo_id,
                dataset_root=dataset_root,
                commit_records=completed_records,
                pending_state=None,
            )

        image_shape = infer_image_shape(specs)
        dataset = load_or_create_dataset(
            repo_id=repo_id,
            dataset_root=dataset_root,
            image_shape=image_shape,
            fps=args.fps,
            robot_type=args.robot_type,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            resume=args.resume,
        )

        completed_keys = set(completed_records)
        pending_specs = [spec for spec in specs if spec_key(spec) not in completed_keys]
        completed_current = len(specs) - len(pending_specs)
        if args.resume:
            print(
                f"[resume] split={args.split} total={len(specs)} "
                f"completed={completed_current} pending={len(pending_specs)}"
            )
        else:
            print(
                f"[start] split={args.split} total={len(specs)} "
                f"pending={len(pending_specs)} num_workers={args.num_workers}"
            )

        converted_episodes = 0
        converted_frames = 0
        skipped_episodes = 0
        per_task_counts = {task: 0 for task in tasks}
        for record in completed_records.values():
            task = record.get("task")
            if task in per_task_counts:
                per_task_counts[task] += 1

        progress = make_progress(
            total=len(pending_specs),
            desc=f"{args.split} convert",
            initial=0,
        )
        progress.set_postfix(
            overall_done=completed_current,
            overall_total=len(specs),
            skipped=skipped_episodes,
        )

        try:
            for plan_index, (spec, prepared_or_exc) in enumerate(
                iter_prepared_episodes(
                    pending_specs,
                    num_workers=args.num_workers,
                    rlbench_root=rlbench_root,
                ),
                start=completed_current,
            ):
                if isinstance(prepared_or_exc, Exception):
                    skipped_episodes += 1
                    print(
                        "[warn] skip"
                        f" split={args.split}"
                        f" task={spec.task}"
                        f" variation={spec.variation}"
                        f" episode={spec.episode}"
                        f" reason={prepared_or_exc}",
                        file=sys.stderr,
                    )
                    progress.update(1)
                    progress.set_postfix(
                        overall_done=completed_current,
                        overall_total=len(specs),
                        skipped=skipped_episodes,
                    )
                    continue

                pending_state = {
                    "raw_episode_key": spec_key(spec),
                    "task": spec.task,
                    "variation": spec.variation,
                    "episode": spec.episode,
                    "episode_dir": str(spec.episode_dir),
                    "instruction": spec.instruction,
                    "plan_index": plan_index,
                    "frames": prepared_or_exc.num_frames,
                    "output_episode_index": dataset.meta.total_episodes,
                    "started_at": utc_now_iso(),
                }
                atomic_write_json(pending_path, pending_state)

                try:
                    episode_frames = write_prepared_episode(dataset, prepared_or_exc)
                except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
                    cleanup_stale_images(dataset_root)
                    raise
                except Exception as exc:  # pragma: no cover - write-path failure
                    cleanup_stale_images(dataset_root)
                    raise RuntimeError(
                        "LeRobot write failed after a pending transaction was recorded. "
                        "The pending state has been preserved for a safe --resume recovery."
                    ) from exc

                record = build_commit_record(
                    spec=spec,
                    frames=episode_frames,
                    output_episode_index=dataset.meta.total_episodes - 1,
                    finished_at=utc_now_iso(),
                )
                append_resume_record(progress_path, record)
                remove_pending_state(pending_path)

                converted_episodes += 1
                converted_frames += episode_frames
                per_task_counts[spec.task] += 1
                completed_records[record["raw_episode_key"]] = record
                completed_keys.add(record["raw_episode_key"])
                completed_current += 1

                progress.update(1)
                progress.set_postfix(
                    overall_done=completed_current,
                    overall_total=len(specs),
                    skipped=skipped_episodes,
                )
        finally:
            progress.close()
            if getattr(dataset, "image_writer", None) is not None:
                dataset.stop_image_writer()

        audit_dataset_state(
            repo_id=repo_id,
            dataset_root=dataset_root,
            commit_records=completed_records,
            pending_state=load_pending_state(pending_path),
        )
        if completed_current != len(specs):
            raise RuntimeError(
                f"Conversion finished with missing episodes: completed={completed_current}, "
                f"expected={len(specs)}, skipped={skipped_episodes}."
            )

        print(
            f"[done] split={args.split} output={dataset_root} "
            f"converted_now={converted_episodes} frames_now={converted_frames} "
            f"skipped_now={skipped_episodes} total_done={completed_current}/{len(specs)}"
        )
        for task, count in per_task_counts.items():
            print(f"[task] split={args.split} task={task} episodes={count}")

        print(
            "For later openpi training, set "
            f"HF_LEROBOT_HOME={output_root} and use repo_id={repo_id}."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
