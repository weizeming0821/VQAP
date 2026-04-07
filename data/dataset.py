import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

try:
	from .view_select import ViewSelector
except ImportError:
	from view_select import ViewSelector


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


"""Action_Primitive_Dataset 类。

输入：
    __init__:
		dataset_root: str，数据集根目录路径，需包含多个动作子目录，每个动作子目录下包含多个 phase 子目录。
		views: Optional[Sequence[str]]，可选的视角名称列表，默认为 ALL_VIEWS 中的全部视角。
		top_k: int，选择 top-k 个最优视角进行返回，默认为 1。
		view_selector_kwargs: Optional[Dict[str, Any]]，传递给 ViewSelector 的额外参数字典。
输出：
    __getitem__:
		action: str，动作标签。
		trajectory_data: Dict[str, List[Any]]，按字段组织的轨迹数据字典，每个字段对应一个列表，长度等于轨迹帧数。
		trajectory_length: int，轨迹的帧数。
		selected_views: List[Dict[str, Any]]，长度为 top_k 的列表，每个元素包含以下键：
			best_view: str，选定的视角名称。
			best_start_image: torch.Tensor，首帧经过 transforms 处理后的图像张量，形状为 [3, H, W]。
			best_end_image: torch.Tensor，末帧经过 transforms 处理后的图像张量，形状为 [3, H, W]。
			best_score: torch.Tensor，0 维标量张量，dtype 由 config/utils.yaml 中的 tensor_dtype 控制，分数越大表示视觉变化越明显。
"""
class Action_Primitive_Dataset(Dataset):

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

		self.views = list(views) if views is not None else list(ALL_VIEWS)
		self.top_k = self._normalize_top_k(top_k)
		self.view_selector_kwargs = dict(view_selector_kwargs or {})
		self._view_selector: Optional[ViewSelector] = None
		self.samples: List[Dict[str, Any]] = []
		self.index_dataset()

	"""检查 top_k 合法性，并统一成正整数。"""
	def _normalize_top_k(self, top_k: int) -> int:
		top_k = int(top_k)
		if top_k <= 0 or top_k > len(self.views):
			raise ValueError("top_k must be a positive integer and less than or equal to the number of available views")
		return top_k

	"""获取一个 action 目录下的所有 phase 子目录，并按名称排序。"""
	def _get_phase_dirs(self, action_dir: Path) -> List[Path]:
		return sorted(
			[path for path in action_dir.iterdir() if path.is_dir() and path.name.startswith("phase_")],
			key=lambda path: path.name,
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
				"frame_count": len(image_files),
			}
		return view_pairs

	"""遍历数据集构建样本列表。"""
	def index_dataset(self) -> None:
		self.samples = []
		action_dirs = sorted(
			[path for path in self.dataset_root.iterdir() if path.is_dir() and (path / "action_metadata.json").is_file()],
			key=lambda path: path.name,
		)
		if not action_dirs:
			raise ValueError(f"No action directories found under: {self.dataset_root}")

		# 遍历根目录下的动作目录
		for action_dir in action_dirs:
			action_name = action_dir.name

			# 遍历动作目录下的 phase 子目录
			for phase_path in self._get_phase_dirs(action_dir):
				pkl_path = phase_path / "low_dim_obs.pkl"
				if not pkl_path.is_file():
					continue

				view_image_pairs = self._collect_view_image_pairs(phase_path)
				if not view_image_pairs:
					continue

				self.samples.append(
					{
						"action": action_name,
						"phase_path": str(phase_path),
						"pkl_path": str(pkl_path),
						"view_image_pairs": view_image_pairs,
					}
				)

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

		# 如果限定范围只有一个视角，直接返回长度为 1 的列表，无需调用选择器。
		if len(view_names) == 1:
			only_view = view_names[0]
			only_pair = view_pairs[only_view]
			return [
				{
					"best_view": only_view,
					"best_start_path": only_pair["start_path"],
					"best_end_path": only_pair["end_path"],
					"best_score": 1.0,  # 只有一个视角时，变化分数默认为 1.0
				}
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

	"""返回动作标签、该 phase 的轨迹字典、轨迹长度和 selected_views 列表。"""
	def __getitem__(self, index: int) -> Dict[str, Any]:

		if index < 0 or index >= len(self.samples):
			raise IndexError(f"Index out of range: {index}")

		sample = self.samples[index]
		observations = self.load_action_data(sample["pkl_path"])
		trajectory_data = self._build_trajectory_data(observations)

		# 获取最优视角数据
		selected_views = self.view_select(sample)

		return {
			"action": sample["action"],
			"trajectory_data": trajectory_data,
			"trajectory_length": len(observations),
			"selected_views": selected_views,
		}

	"""返回数据集样本数量。"""
	def __len__(self) -> int:
		return len(self.samples)


ActionPrimitiveDataset = Action_Primitive_Dataset


# 简单测试
if __name__ == "__main__":
	dataset = Action_Primitive_Dataset(dataset_root="Action_Primitive_Dataset_v0")
	print(f"Dataset contains {len(dataset)} samples.")
	sample = dataset[0]
	print(f"Sample 0 action: {sample['action']}")
	print(f"Sample 0 trajectory length: {sample['trajectory_length']}")
	print(f"Sample 0 trajectory data keys: {list(sample['trajectory_data'].keys())}")
	for key, value in sample["trajectory_data"].items():
		print(f"  {key}: {type(value)} with {len(value)} frames")
		if len(value) > 0:
			print(f"    First frame value: {value[0]}")
	print(f"Sample 0 selected views: {sample['selected_views']}")

