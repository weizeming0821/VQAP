import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from torch.utils.data import Dataset

try:
    from .view_select import ViewSelector
    from .utils import normalize
except ImportError:
    from view_select import ViewSelector
    from utils import normalize


ALL_VIEWS: List[str] = [
	"front",
	"left_shoulder",
	"right_shoulder",
	"overhead",
	"wrist",
]

LOW_DIM_FIELDS: List[str] = [
	"joint_positions",
	"gripper_pose",
	"joint_velocities",
	"joint_forces",
	"gripper_joint_positions",
	"gripper_touch_forces",
	"gripper_open",
]

IMAGE_SUFFIXES: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
GLOBAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "global.yaml"


"""读取全局配置文件。"""
def load_global_config() -> Dict[str, Any]:
	if not GLOBAL_CONFIG_PATH.is_file():
		return {}

	with GLOBAL_CONFIG_PATH.open("r", encoding="utf-8") as file:
		config = yaml.safe_load(file)

	if config is None:
		return {}
	if not isinstance(config, dict):
		raise ValueError(f"Config file must contain a top-level mapping: {GLOBAL_CONFIG_PATH}")
	return config


"""AtomActionDataset 类。

输入：
    __init__:
		dataset_root: str，数据集根目录路径。Action/Task/VariationX/Phase_XXX
			会额外读取 config/global.yaml 中的 train_actions 配置，
			必须为非空动作列表，并且每个动作都必须存在于数据集目录中。
		views: Optional[Sequence[str]]，可选的视角名称列表，默认为 ALL_VIEWS 中的全部视角。
		top_k: int，选择 top-k 个最优视角进行返回，默认为 1。
		view_selector_kwargs: Optional[Dict[str, Any]]，传递给 ViewSelector 的额外参数字典。
输出：
    __getitem__:
		Action: str，动作标签。
		Task: str，任务名称。
		Variation: str，variation 名称。
		trajectory_data: Dict[str, List[Any]]，按字段组织的轨迹数据字典，每个字段对应一个列表，长度等于轨迹帧数。
		trajectory_length: int，轨迹的帧数。
		selected_views: List[Dict[str, Any]]，长度为小于等于 top_k 的列表，每个元素包含以下键：
			best_view: str，选定的视角名称。
			best_start_image: torch.Tensor，首帧经过 transforms 处理后的图像张量，形状为 [3, H, W]。
			best_end_image: torch.Tensor，末帧经过 transforms 处理后的图像张量，形状为 [3, H, W]。
			best_score: torch.Tensor，0 维标量张量，dtype 由 config/global.yaml 中的 tensor_dtype 控制，分数越大表示视觉变化越明显。
"""
class AtomActionDataset(Dataset):

	def __init__(
		self,
		dataset_root: str,
		views: Optional[Sequence[str]] = None,
		top_k: int = 1,
		view_selector_kwargs: Optional[Dict[str, Any]] = None,
	) -> None:
		
		self.dataset_root = Path(dataset_root).expanduser().resolve()
		if not self.dataset_root.exists():
			raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

		self.selected_actions = self._load_selected_actions()
		self.views = list(views) if views is not None else list(ALL_VIEWS)
		self.top_k = self._normalize_top_k(top_k)
		self.view_selector_kwargs = dict(view_selector_kwargs or {})
		self._view_selector: Optional[ViewSelector] = None
		self.samples: List[Dict[str, Any]] = []
		self.index_dataset()

	"""从全局配置读取需要训练的动作白名单。"""
	def _load_selected_actions(self) -> List[str]:
		config = load_global_config()
		configured_actions = config.get("train_actions")
		if configured_actions is None or not isinstance(configured_actions, (list, tuple)):
			raise ValueError(
				f"train_actions in {GLOBAL_CONFIG_PATH} must be a non-empty list of action names"
			)

		selected_actions: List[str] = []
		seen_actions: set[str] = set()
		for action_name in configured_actions:

			normalized_name = action_name.strip()
			action_key = normalized_name.casefold()
			if action_key in seen_actions:
				continue
			seen_actions.add(action_key)
			selected_actions.append(normalized_name)

		return selected_actions

	"""检查 top_k 合法性，并统一成正整数。"""
	def _normalize_top_k(self, top_k: int) -> int:
		top_k = int(top_k)
		if top_k <= 0 or top_k > len(self.views):
			raise ValueError("top_k must be a positive integer and less than or equal to the number of available views")
		return top_k

	"""按名称排序目录；若目录名以数字结尾，则按数值排序。"""
	def _sort_dirs(self, dirs: Sequence[Path]) -> List[Path]:
		def sort_key(path: Path) -> Tuple[int, Any, str]:
			name = path.name
			index = len(name)
			while index > 0 and name[index - 1].isdigit():
				index -= 1
			if index < len(name):
				prefix = name[:index].casefold()
				suffix = int(name[index:])
				return (0, (prefix, suffix), name)
			return (1, name.casefold(), name)

		return sorted(dirs, key=sort_key)

	"""获取一个目录下的所有 phase 子目录，并按名称排序。"""
	def _get_phase_dirs(self, parent_dir: Path) -> List[Path]:
		return sorted(
			[path for path in parent_dir.iterdir() if path.is_dir() and path.name.lower().startswith("phase_")],
			key=lambda path: path.name,
		)

	"""获取一个 action 目录下的所有 task 子目录，并按名称排序。"""
	def _get_task_dirs(self, action_dir: Path) -> List[Path]:
		task_dirs = [
			path for path in action_dir.iterdir()
			if path.is_dir() and (path / "task_metadata.json").is_file()
		]
		return self._sort_dirs(task_dirs)

	"""获取一个 task 目录下的所有 variation 子目录，并按名称排序。"""
	def _get_variation_dirs(self, task_dir: Path) -> List[Path]:
		variation_dirs = [
			path for path in task_dir.iterdir()
			if path.is_dir() and (path / "variation_metadata.json").is_file()
		]
		return self._sort_dirs(variation_dirs)

	"""根据全局配置筛选需要读取的 action 目录。"""
	def _filter_action_dirs(self, action_dirs: Sequence[Path]) -> List[Path]:
		available_actions = {action_dir.name.casefold(): action_dir for action_dir in action_dirs}
		filtered_action_dirs: List[Path] = []
		missing_actions: List[str] = []
		for action_name in self.selected_actions:
			action_dir = available_actions.get(action_name.casefold())
			if action_dir is None:
				missing_actions.append(action_name)
				continue
			filtered_action_dirs.append(action_dir)

		if missing_actions:
			available_names = ", ".join(action_dir.name for action_dir in action_dirs)
			missing_names = ", ".join(missing_actions)
			raise ValueError(
				f"Configured train_actions not found under {self.dataset_root}: {missing_names}. "
				f"Available actions: {available_names}"
			)

		return filtered_action_dirs

	"""把单个 phase 注册为一个可读取样本。"""
	def _register_phase_sample(
		self,
		action_name: str,
		phase_path: Path,
		task_name: Optional[str] = None,
		variation_name: Optional[str] = None,
	) -> None:
		pkl_path = phase_path / "low_dim_obs.pkl"
		if not pkl_path.is_file():
			return

		view_image_pairs = self._collect_view_image_pairs(phase_path)
		if not view_image_pairs:
			return

		self.samples.append(
			{
				"action": action_name,
				"task": task_name,
				"variation": variation_name,
				"phase_path": str(phase_path),
				"pkl_path": str(pkl_path),
				"view_image_pairs": view_image_pairs,
			}
		)

	"""获取一个 phase 目录下指定视角的 RGB 图像目录路径。"""
	def _get_view_dir(self, phase_path: Path, view_name: str) -> Path:
		return phase_path / f"{view_name}_rgb"

	"""列出一个视角目录下的所有图像文件，并按帧编号排序。"""
	def _list_image_files(self, image_dir: Path) -> List[Path]:
		if not image_dir.is_dir():
			return []

		def sort_key(path: Path) -> Tuple[int, Any]:
			try:
				return (0, int(path.stem))
			except ValueError:
				return (1, path.stem)

		files = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
		return sorted(files, key=sort_key)

	"""收集每个可用视角的起止帧路径。"""
	def _collect_view_image_pairs(self, phase_path: Path) -> Dict[str, Dict[str, str]]:

		view_pairs: Dict[str, Dict[str, str]] = {}
		for view_name in self.views:
			image_files = self._list_image_files(self._get_view_dir(phase_path, view_name))
			if not image_files:
				continue

			view_pairs[view_name] = {
				"start_path": str(image_files[0]),
				"end_path": str(image_files[-1]),
			}
		return view_pairs

	"""遍历数据集构建样本列表。"""
	def index_dataset(self) -> None:
		self.samples = []
		action_dirs = self._sort_dirs(
			[path for path in self.dataset_root.iterdir() if path.is_dir() and (path / "action_metadata.json").is_file()],
		)
		action_dirs = self._filter_action_dirs(action_dirs)
		if not action_dirs:
			raise ValueError(f"No action directories found under: {self.dataset_root}")

		for action_dir in action_dirs:
			action_name = action_dir.name
			task_dirs = self._get_task_dirs(action_dir)

			if task_dirs:
				for task_dir in task_dirs:
					task_name = task_dir.name
					for variation_dir in self._get_variation_dirs(task_dir):
						variation_name = variation_dir.name
						for phase_path in self._get_phase_dirs(variation_dir):
							self._register_phase_sample(
								action_name=action_name,
								phase_path=phase_path,
								task_name=task_name,
								variation_name=variation_name,
							)
				continue

		if not self.samples:
			raise ValueError(f"No valid phase samples found under: {self.dataset_root}")

	"""从 pickle 文件中读取一个 phase 的全部 Observation。"""
	def load_action_data(self, pkl_path: str) -> List[Any]:
		
		phase_pkl_path = Path(pkl_path).expanduser().resolve()
		if not phase_pkl_path.is_file():
			raise FileNotFoundError(f"low_dim_obs.pkl does not exist: {phase_pkl_path}")

		with phase_pkl_path.open("rb") as file:
			observations = pickle.load(file)

		if not isinstance(observations, list) or len(observations) == 0:
			raise ValueError(f"Invalid or empty observation list in: {phase_pkl_path}")
		return observations

	"""将单帧低维属性序列化为 Python 原生类型。"""
	def _serialize_value(self, value: Any) -> Any:
		if value is None:
			return None
		if hasattr(value, "tolist"):
			return value.tolist()
		if isinstance(value, (list, tuple)):
			return [self._serialize_value(item) for item in value]
		if isinstance(value, (str, bool, int, float)):
			return value
		return value

	"""将一个 phase 的 Observation 列表整理为按字段组织的轨迹字典。"""
	def _build_trajectory_data(self, observations: Sequence[Any]) -> Dict[str, List[Any]]:
		trajectory_data: Dict[str, List[Any]] = {field_name: [] for field_name in LOW_DIM_FIELDS}
		for obs in observations:
			for field_name in LOW_DIM_FIELDS:
				value = None if obs is None else getattr(obs, field_name, None)
				trajectory_data[field_name].append(self._serialize_value(value))
		return trajectory_data

	"""获取视角选择器实例"""
	def _get_view_selector(self) -> ViewSelector:
		if self._view_selector is None:
			self._view_selector = ViewSelector(**self.view_selector_kwargs)
		return self._view_selector

	"""为当前 phase 选择 top-k 个最优视角，并返回列表结果。"""
	def view_select(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:

		view_pairs = sample["view_image_pairs"]
		view_names = list(view_pairs.keys())
		if not view_names:
			raise ValueError(f"No available RGB views found for sample: {sample['phase_path']}")

		# 如果只有一个可用视角，直接构造与多视角一致的返回结构，无需执行 DINOv2 打分。
		if len(view_names) == 1:
			only_view = view_names[0]
			only_pair = view_pairs[only_view]
			view_selector = self._get_view_selector()
			return [
				view_selector.build_view_result(
					view_name=only_view,
					start_path=only_pair["start_path"],
					end_path=only_pair["end_path"],
					score=1.0,
				)
			]

		# 多个视角时，调用选择器进行评估和排序
		start_paths = [view_pairs[view_name]["start_path"] for view_name in view_names]
		end_paths = [view_pairs[view_name]["end_path"] for view_name in view_names]
		try:
			selected_views = self._get_view_selector().select_best_view(
				start_paths=start_paths,
				end_paths=end_paths,
				view_names=view_names,
				top_k=self.top_k,
			)
			if len(selected_views) == 0:
				raise RuntimeError("View selector returned an empty result list")
			return selected_views
		except Exception as exc:
			raise RuntimeError(
				f"Failed to select best {self.top_k} view(s) for sample: {sample['phase_path']}. "
				f"Error: {exc}"
			)

	"""返回动作/任务/variation 标签、轨迹字典、轨迹长度和 selected_views 列表。"""
	def __getitem__(self, index: int) -> Dict[str, Any]:

		if index < 0 or index >= len(self.samples):
			raise IndexError(f"Index out of range: {index}")

		sample = self.samples[index]
		observations = self.load_action_data(sample["pkl_path"])
		trajectory_data = normalize(
			trajectory_data=self._build_trajectory_data(observations),
			dataset_root=self.dataset_root,
		)

		# 获取最优视角数据
		selected_views = self.view_select(sample)

		return {
			"Action": sample["action"],
			"Task": sample["task"],
			"Variation": sample["variation"],
			"trajectory_data": trajectory_data,
			"trajectory_length": len(observations),
			"selected_views": selected_views,
		}

	"""返回数据集样本数量。"""
	def __len__(self) -> int:
		return len(self.samples)


# 简单测试
if __name__ == "__main__":
	dataset = AtomActionDataset(dataset_root="AtomAction_Dataset", views=["front"])
	print(f"Dataset contains {len(dataset)} samples.")
	sample = dataset[0]
	print(f"Sample 0 action: {sample['Action']}")
	print(f"Sample 0 task: {sample['Task']}")
	print(f"Sample 0 variation: {sample['Variation']}")
	print(f"Sample 0 trajectory length: {sample['trajectory_length']}")
	print(f"Sample 0 trajectory data keys: {list(sample['trajectory_data'].keys())}")
	for key, value in sample["trajectory_data"].items():
		print(f"  {key}: {type(value)} with {len(value)} frames")
		if len(value) > 0:
			print(f"    First frame value: {value[0]}")
	print(f"Sample 0 selected views: {sample['selected_views']}")

