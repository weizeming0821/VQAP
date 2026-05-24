from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torchvision import transforms



CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "global_config.yaml"
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


"""统计数据均值与方差并保存到 action_metadata.json 中"""
def compute_and_save_statistics(dataset: Any) -> None:
	pass


"""轨迹数据预处理函数"""