import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torchvision import transforms



CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "global_config.yaml"
TRAJ_STATS_KEY = "traj_stats"
NORMALIZATION_EPS = 1e-6
ZSCORE_FIELDS = {"joint_positions", "gripper_joint_positions"}
QUANTILE_FIELDS = {"joint_velocities", "joint_forces", "gripper_touch_forces"}
SUPPORTED_TENSOR_DTYPES = {
	"float16": torch.float16,
	"float32": torch.float32,
	"float64": torch.float64,
	"bfloat16": torch.bfloat16,
 }

"""读取全局配置文件。"""
def load_utils_config() -> Dict[str, Any]:
	if not CONFIG_PATH.is_file():
		return {}

	with CONFIG_PATH.open("r", encoding="utf-8") as file:
		config = yaml.safe_load(file)

	if config is None:
		return {}
	if not isinstance(config, dict):
		raise ValueError(f"Config file must contain a top-level mapping: {CONFIG_PATH}")
	return config


"""读取全局 tensor dtype 配置，未配置时默认返回 torch.float32。"""
def get_configured_tensor_dtype() -> torch.dtype:
	config = load_utils_config()
	dtype_name = str(config.get("tensor_dtype", "float32")).strip().lower()
	if dtype_name not in SUPPORTED_TENSOR_DTYPES:
		supported = ", ".join(sorted(SUPPORTED_TENSOR_DTYPES.keys()))
		raise ValueError(
			f"Unsupported tensor_dtype in {CONFIG_PATH}: {dtype_name}. Supported values: {supported}"
		)
	return SUPPORTED_TENSOR_DTYPES[dtype_name]


"""构建 DINOv2 图像预处理流程。"""
def build_dinov2_transform(input_size: int = 224) -> transforms.Compose:
	if input_size <= 0:
		raise ValueError("input_size must be a positive integer")
	if input_size % 14 != 0:
		raise ValueError("input_size must be a multiple of 14 for DINOv2 models")

	return transforms.Compose(
		[
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)


"""把单帧低维数据转成张量，标量统一扩成 [1]。"""
def _to_trajectory_tensor(value: Any, tensor_dtype: torch.dtype) -> Optional[torch.Tensor]:
	if value is None:
		return None

	tensor = torch.as_tensor(value, dtype=tensor_dtype)
	if tensor.ndim == 0:
		tensor = tensor.unsqueeze(0)
	return tensor


"""将 AtomActionDataset 的样本列表整理成批数据。

输入：
	batch: List[Dict[str, Any]]，每个元素对应 AtomActionDataset.__getitem__ 的返回值。

输出：
	Dict[str, Any]
		Action/Task/Variation: 长度为 B 的列表。
		trajectory_data: Dict[str, torch.Tensor]，每个字段形状为 [B, T, D]。
		trajectory_length: torch.LongTensor，形状为 [B]。
		trajectory_mask: torch.BoolTensor，形状为 [B, T]，True 表示有效帧。
		selected_views: 长度为 B 的列表，保留每个样本原始的视角结果。
"""
def AtomActionDataset_collate_fn(batch: Any) -> Dict[str, Any]:
	if not isinstance(batch, list) or len(batch) == 0:
		raise ValueError("batch must be a non-empty list")

	tensor_dtype = get_configured_tensor_dtype()
	trajectory_length = torch.tensor(
		[int(sample["trajectory_length"]) for sample in batch],
		dtype=torch.long,
	)
	batch_size = len(batch)
	max_length = int(trajectory_length.max().item())
	trajectory_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
	for batch_index, length in enumerate(trajectory_length.tolist()):
		trajectory_mask[batch_index, :length] = True

	trajectory_data: Dict[str, torch.Tensor] = {}
	field_names = list(batch[0]["trajectory_data"].keys())
	for field_name in field_names:
		feature_shape: Tuple[int, ...] = (1,)
		for sample in batch:
			for frame_value in sample["trajectory_data"][field_name]:
				frame_tensor = _to_trajectory_tensor(frame_value, tensor_dtype)
				if frame_tensor is not None:
					feature_shape = tuple(frame_tensor.shape)
					break
			if feature_shape != (1,):
				break

		# 每个字段都补到 [B, T, D]，未使用位置保持 0。
		padded_field = torch.zeros((batch_size, max_length, *feature_shape), dtype=tensor_dtype)
		for batch_index, sample in enumerate(batch):
			field_sequence = sample["trajectory_data"][field_name]
			expected_length = int(trajectory_length[batch_index].item())
			if len(field_sequence) != expected_length:
				raise ValueError(
					f"trajectory_data[{field_name}] length does not match trajectory_length: "
					f"{len(field_sequence)} != {expected_length}"
				)

			for time_index, frame_value in enumerate(field_sequence):
				frame_tensor = _to_trajectory_tensor(frame_value, tensor_dtype)
				if frame_tensor is None:
					continue
				if tuple(frame_tensor.shape) != feature_shape:
					raise ValueError(
						f"Inconsistent frame shape in field {field_name}: "
						f"expected {feature_shape}, got {tuple(frame_tensor.shape)}"
					)
				padded_field[batch_index, time_index] = frame_tensor

		trajectory_data[field_name] = padded_field

	return {
		"Action": [sample["Action"] for sample in batch],
		"Task": [sample["Task"] for sample in batch],
		"Variation": [sample["Variation"] for sample in batch],
		"trajectory_data": trajectory_data,
		"trajectory_length": trajectory_length,
		"trajectory_mask": trajectory_mask,
		"selected_views": [sample["selected_views"] for sample in batch],
	}


"""调用独立脚本中的轨迹统计逻辑。"""
def compute_and_save_statistics(dataset: Any) -> None:
	try:
		from .scripts.compute_dataset_statistics import compute_and_save_statistics as _compute_statistics
	except ImportError:
		from data.scripts.compute_dataset_statistics import compute_and_save_statistics as _compute_statistics

	_compute_statistics(dataset)


"""读取数据集根目录 dataset_metadata.json 中的轨迹统计量。"""
@lru_cache(maxsize=None)
def _load_dataset_traj_stats(dataset_root: str) -> Dict[str, Dict[str, Any]]:
	metadata_path = Path(dataset_root).expanduser().resolve() / "dataset_metadata.json"
	if not metadata_path.is_file():
		raise FileNotFoundError(f"dataset_metadata.json does not exist: {metadata_path}")

	with metadata_path.open("r", encoding="utf-8") as file:
		metadata = json.load(file)

	traj_stats = metadata.get(TRAJ_STATS_KEY)
	if not isinstance(traj_stats, dict) or len(traj_stats) == 0:
		raise ValueError(f"Missing traj_stats in dataset metadata: {metadata_path}")
	return traj_stats


"""将统计量列表转成与参考张量同形状的张量。"""
def _stats_to_tensor(values: Any, reference: torch.Tensor, field_name: str, stat_name: str) -> torch.Tensor:
	stats_tensor = torch.as_tensor(values, dtype=reference.dtype)
	if tuple(stats_tensor.shape) != tuple(reference.shape):
		raise ValueError(
			f"Invalid {stat_name} shape for field {field_name}: "
			f"expected {tuple(reference.shape)}, got {tuple(stats_tensor.shape)}"
		)
	return stats_tensor


"""按分位数线性归一化，使 q01/q99 对应 -1/1。"""
def _quantile_normalize(
	value: torch.Tensor,
	q01: torch.Tensor,
	q99: torch.Tensor,
) -> torch.Tensor:
	denominator = (q99 - q01).clamp_min(NORMALIZATION_EPS)
	return ((value - q01) / denominator) * 2.0 - 1.0


"""按 Z-score 逐维归一化。"""
def _zscore_normalize(
	value: torch.Tensor,
	mean: torch.Tensor,
	std: torch.Tensor,
) -> torch.Tensor:
	return (value - mean) / std.clamp_min(NORMALIZATION_EPS)


"""将 [qx, qy, qz, qw] 四元数转成 6D 旋转表示。"""
def _quaternion_to_rotation6d(quaternion: torch.Tensor) -> torch.Tensor:
	if quaternion.shape != (4,):
		raise ValueError(f"quaternion must have shape (4,), got {tuple(quaternion.shape)}")

	unit_quaternion = quaternion / quaternion.norm().clamp_min(NORMALIZATION_EPS)
	qx, qy, qz, qw = unit_quaternion.unbind(dim=0)
	xx, yy, zz = qx * qx, qy * qy, qz * qz
	xy, xz, yz = qx * qy, qx * qz, qy * qz
	wx, wy, wz = qw * qx, qw * qy, qw * qz

	rotation_matrix = torch.stack(
		[
			torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy))),
			torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx))),
			torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy))),
		],
		dim=0,
	)
	# 取旋转矩阵前两列并按列拼接，输出 [6]。
	return torch.cat((rotation_matrix[:, 0], rotation_matrix[:, 1]), dim=0)


"""轨迹数据预处理函数"""
def normalize(trajectory_data: Dict[str, Any], dataset_root: Any) -> Dict[str, Any]:
	if not isinstance(trajectory_data, dict):
		raise ValueError("trajectory_data must be a dictionary")

	tensor_dtype = get_configured_tensor_dtype()
	traj_stats = _load_dataset_traj_stats(str(dataset_root))
	normalized_data: Dict[str, Any] = {}

	for field_name, field_sequence in trajectory_data.items():
		if field_name == "gripper_open":
			normalized_data[field_name] = list(field_sequence)
			continue

		field_stats = traj_stats.get(field_name)
		if not isinstance(field_stats, dict):
			raise ValueError(f"Missing normalization stats for field: {field_name}")

		normalized_sequence = []
		for frame_value in field_sequence:
			frame_tensor = _to_trajectory_tensor(frame_value, tensor_dtype)
			if frame_tensor is None:
				normalized_sequence.append(None)
				continue

			if field_name == "gripper_pose":
				if frame_tensor.shape != (7,):
					raise ValueError(f"gripper_pose must have shape (7,), got {tuple(frame_tensor.shape)}")
				position_q01 = _stats_to_tensor(field_stats["q01"][:3], frame_tensor[:3], field_name, "q01")
				position_q99 = _stats_to_tensor(field_stats["q99"][:3], frame_tensor[:3], field_name, "q99")
				normalized_position = _quantile_normalize(frame_tensor[:3], position_q01, position_q99)
				rotation6d = _quaternion_to_rotation6d(frame_tensor[3:])
				# gripper_pose 由原始 7 维变成 [位置 3 维 + 旋转 6 维] 共 9 维。
				normalized_sequence.append(torch.cat((normalized_position, rotation6d), dim=0).tolist())
				continue

			if field_name in ZSCORE_FIELDS:
				mean = _stats_to_tensor(field_stats["mean"], frame_tensor, field_name, "mean")
				std = _stats_to_tensor(field_stats["std"], frame_tensor, field_name, "std")
				normalized_frame = _zscore_normalize(frame_tensor, mean, std)
			elif field_name in QUANTILE_FIELDS:
				q01 = _stats_to_tensor(field_stats["q01"], frame_tensor, field_name, "q01")
				q99 = _stats_to_tensor(field_stats["q99"], frame_tensor, field_name, "q99")
				normalized_frame = _quantile_normalize(frame_tensor, q01, q99)
			else:
				normalized_frame = frame_tensor

			normalized_sequence.append(normalized_frame.tolist())

		normalized_data[field_name] = normalized_sequence

	return normalized_data