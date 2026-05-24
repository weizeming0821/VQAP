"""独立统计 AtomAction 数据集低维轨迹字段的脚本。

功能：
	- 遍历数据集中每个 phase 的 low_dim_obs.pkl。
	- 跳过 gripper_open，对其余连续动作字段按维度统计。
	- 计算 n、mean、var、std、min、max、q01、q99。
	- 将统计结果写回 dataset、action、task 三层 metadata 的 traj_stats 字段。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import ALL_VIEWS, AtomActionDataset
from data.utils import _to_trajectory_tensor


TRAJ_STATS_KEY = "traj_stats"
EXCLUDED_TRAJECTORY_FIELDS = {"gripper_open"}


"""维护单个字段逐维统计量的在线累加器。"""
class _RunningFieldStats:
	"""用在线方式累计均值、方差、最值。"""
	def __init__(self) -> None:
		self.count = 0
		self.feature_shape: Optional[Tuple[int, ...]] = None
		self.mean: Optional[torch.Tensor] = None
		self.m2: Optional[torch.Tensor] = None
		self.min: Optional[torch.Tensor] = None
		self.max: Optional[torch.Tensor] = None

	"""将一批 [N, D] 数据合并到当前字段统计量中。"""
	def update(self, values: torch.Tensor, feature_shape: Tuple[int, ...], field_name: str) -> None:
		if values.ndim != 2:
			raise ValueError(f"Field {field_name} expects a [N, D] tensor, got shape {tuple(values.shape)}")

		if self.feature_shape is None:
			self.feature_shape = feature_shape
			self.mean = values.mean(dim=0)
			self.m2 = torch.zeros_like(self.mean)
			centered = values - self.mean
			self.m2 += (centered * centered).sum(dim=0)
			self.min = values.amin(dim=0)
			self.max = values.amax(dim=0)
			self.count = int(values.shape[0])
			return

		if self.feature_shape != feature_shape:
			raise ValueError(
				f"Inconsistent feature shape in field {field_name}: "
				f"expected {self.feature_shape}, got {feature_shape}"
			)

		batch_count = int(values.shape[0])
		batch_mean = values.mean(dim=0)
		centered = values - batch_mean
		batch_m2 = (centered * centered).sum(dim=0)

		total_count = self.count + batch_count
		delta = batch_mean - self.mean
		self.mean = self.mean + delta * (batch_count / total_count)
		self.m2 = self.m2 + batch_m2 + (delta * delta) * (self.count * batch_count / total_count)
		self.min = torch.minimum(self.min, values.amin(dim=0))
		self.max = torch.maximum(self.max, values.amax(dim=0))
		self.count = total_count


"""读取 JSON 元数据。"""
def _load_json_mapping(path: Path) -> Dict[str, Any]:
	if not path.is_file():
		return {}

	with path.open("r", encoding="utf-8") as file:
		content = json.load(file)

	if not isinstance(content, dict):
		raise ValueError(f"Metadata file must contain a top-level mapping: {path}")
	return content


"""写回 JSON 元数据。"""
def _save_json_mapping(path: Path, content: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		json.dump(content, file, ensure_ascii=False, indent=2)


"""将单个字段的轨迹序列拼成 [N, D] 张量。"""
def _stack_field_sequence(field_name: str, field_sequence: Any) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, ...]]]:
	feature_shape: Optional[Tuple[int, ...]] = None
	frame_tensors: List[torch.Tensor] = []
	for frame_value in field_sequence:
		frame_tensor = _to_trajectory_tensor(frame_value, torch.float64)
		if frame_tensor is None:
			continue

		current_shape = tuple(frame_tensor.shape)
		if feature_shape is None:
			feature_shape = current_shape
		elif feature_shape != current_shape:
			raise ValueError(
				f"Inconsistent frame shape in field {field_name}: expected {feature_shape}, got {current_shape}"
			)

		frame_tensors.append(frame_tensor.reshape(1, -1))

	if not frame_tensors or feature_shape is None:
		return None, None
	return torch.cat(frame_tensors, dim=0), feature_shape


"""把张量转成 Python 浮点列表。"""
def _tensor_to_float_list(value: torch.Tensor) -> List[float]:
	return [float(item) for item in value.tolist()]


"""返回字段统计对象。"""
def _get_or_create_field_stats(group_stats: Dict[str, _RunningFieldStats], field_name: str) -> _RunningFieldStats:
	field_stats = group_stats.get(field_name)
	if field_stats is None:
		field_stats = _RunningFieldStats()
		group_stats[field_name] = field_stats
	return field_stats


"""返回 key 对应的嵌套字典，不存在时自动创建。"""
def _get_or_create_nested_mapping(container: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
	mapping = container.get(key)
	if mapping is None:
		mapping = {}
		container[key] = mapping
	return mapping


"""汇总一组字段统计量。"""
def _finalize_group_stats(
	group_stats: Dict[str, _RunningFieldStats],
	raw_field_values: Dict[str, List[torch.Tensor]],
) -> Dict[str, Dict[str, Any]]:
	serialized: Dict[str, Dict[str, Any]] = {}
	quantiles = torch.tensor([0.01, 0.99], dtype=torch.float64)

	for field_name in sorted(group_stats.keys()):
		field_stats = group_stats[field_name]
		if field_stats.count == 0:
			continue

		stacked_values = torch.cat(raw_field_values[field_name], dim=0)
		var = field_stats.m2 / field_stats.count
		std = torch.sqrt(var)
		q01, q99 = torch.quantile(stacked_values, quantiles, dim=0)

		serialized[field_name] = {
			"n": int(field_stats.count),
			"mean": _tensor_to_float_list(field_stats.mean),
			"var": _tensor_to_float_list(var),
			"std": _tensor_to_float_list(std),
			"min": _tensor_to_float_list(field_stats.min),
			"max": _tensor_to_float_list(field_stats.max),
			"q01": _tensor_to_float_list(q01),
			"q99": _tensor_to_float_list(q99),
		}

	return serialized


"""遍历数据集样本，返回 action、task 和轨迹字段字典。"""
def _iter_dataset_trajectory_data(dataset: Any) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
	if all(hasattr(dataset, attr) for attr in ("samples", "load_action_data", "_build_trajectory_data")):
		for sample in dataset.samples:
			observations = dataset.load_action_data(sample["pkl_path"])
			yield sample["action"], sample["task"], dataset._build_trajectory_data(observations)
		return

	for index in range(len(dataset)):
		sample = dataset[index]
		yield sample["Action"], sample["Task"], sample["trajectory_data"]


"""返回数据集根目录下全部可索引的动作目录名。"""
def _list_action_names(dataset_root: Path) -> List[str]:
	if not dataset_root.is_dir():
		return []

	return sorted(
		child.name
		for child in dataset_root.iterdir()
		if child.is_dir() and (child / "action_metadata.json").is_file()
	)


"""自动查找仓库内默认可用的数据集根目录。"""
def resolve_dataset_root() -> Path:
	for dataset_name in ("AtomAction_Dataset", "AtomAction_Dataset_re"):
		dataset_root = (PROJECT_ROOT / dataset_name).resolve()
		if _list_action_names(dataset_root):
			return dataset_root

	raise FileNotFoundError("No indexable AtomAction dataset root was found")


"""构建一个仅用于轨迹统计的 AtomActionDataset。"""
def build_statistics_dataset(dataset_root: Path, views: Optional[List[str]] = None) -> AtomActionDataset:
	dataset_root = dataset_root.expanduser().resolve()
	if not dataset_root.exists():
		raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

	dataset = AtomActionDataset.__new__(AtomActionDataset)
	dataset.dataset_root = dataset_root
	dataset.selected_actions = _list_action_names(dataset_root)
	dataset.views = list(views) if views is not None else list(ALL_VIEWS)
	dataset.top_k = 1
	dataset.view_selector_kwargs = {}
	dataset._view_selector = None
	dataset.samples = []
	dataset.index_dataset()
	return dataset


"""统计轨迹字段并写回 dataset、action、task 三层 metadata。"""
def compute_and_save_statistics(dataset: Any) -> None:
	dataset_root = Path(dataset.dataset_root).expanduser().resolve()
	dataset_stats: Dict[str, _RunningFieldStats] = {}
	action_stats: Dict[str, Dict[str, _RunningFieldStats]] = {}
	task_stats: Dict[str, Dict[str, Dict[str, _RunningFieldStats]]] = {}
	task_raw_values: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = {}

	for action_name, task_name, trajectory_data in _iter_dataset_trajectory_data(dataset):
		action_field_stats = _get_or_create_nested_mapping(action_stats, action_name)
		action_task_stats = _get_or_create_nested_mapping(task_stats, action_name)
		task_field_stats = _get_or_create_nested_mapping(action_task_stats, task_name)
		action_task_raw = _get_or_create_nested_mapping(task_raw_values, action_name)
		task_field_raw = _get_or_create_nested_mapping(action_task_raw, task_name)

		for field_name, field_sequence in trajectory_data.items():
			if field_name in EXCLUDED_TRAJECTORY_FIELDS:
				continue

			stacked_values, feature_shape = _stack_field_sequence(field_name, field_sequence)
			if stacked_values is None or feature_shape is None:
				continue

			_get_or_create_field_stats(dataset_stats, field_name).update(stacked_values, feature_shape, field_name)
			_get_or_create_field_stats(action_field_stats, field_name).update(stacked_values, feature_shape, field_name)
			_get_or_create_field_stats(task_field_stats, field_name).update(stacked_values, feature_shape, field_name)
			task_field_raw.setdefault(field_name, []).append(stacked_values)

	dataset_raw_values: Dict[str, List[torch.Tensor]] = {}
	for action_name, task_mapping in task_raw_values.items():
		action_raw_values: Dict[str, List[torch.Tensor]] = {}
		for field_mapping in task_mapping.values():
			for field_name, field_values in field_mapping.items():
				action_raw_values.setdefault(field_name, []).extend(field_values)
				dataset_raw_values.setdefault(field_name, []).extend(field_values)

		action_meta_path = dataset_root / action_name / "action_metadata.json"
		action_meta = _load_json_mapping(action_meta_path)
		action_meta[TRAJ_STATS_KEY] = _finalize_group_stats(action_stats[action_name], action_raw_values)
		_save_json_mapping(action_meta_path, action_meta)

		for task_name, field_mapping in task_mapping.items():
			task_meta_path = dataset_root / action_name / task_name / "task_metadata.json"
			task_meta = _load_json_mapping(task_meta_path)
			task_meta[TRAJ_STATS_KEY] = _finalize_group_stats(task_stats[action_name][task_name], field_mapping)
			_save_json_mapping(task_meta_path, task_meta)

	dataset_meta_path = dataset_root / "dataset_metadata.json"
	dataset_meta = _load_json_mapping(dataset_meta_path)
	dataset_meta[TRAJ_STATS_KEY] = _finalize_group_stats(dataset_stats, dataset_raw_values)
	_save_json_mapping(dataset_meta_path, dataset_meta)


"""解析命令行参数。"""
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="统计轨迹字段并写回 dataset/action/task 各层级 metadata。")
	parser.add_argument("--dataset_root", type=str, default=None, help="数据集根目录，默认自动探测。")
	parser.add_argument("--views", nargs="*", default=None, help="索引 phase 时使用的视角列表，默认使用全部视角。")
	return parser.parse_args()


"""命令行入口。"""
def main() -> None:
	args = parse_args()
	dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else resolve_dataset_root()
	dataset = build_statistics_dataset(dataset_root, args.views)
	compute_and_save_statistics(dataset)
	print(f"Saved trajectory statistics to {dataset_root}")


if __name__ == "__main__":
	main()