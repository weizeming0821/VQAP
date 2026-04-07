from pathlib import Path
from typing import Any, Dict

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

# """重写 Action_Primitive_Dataset 类的数据裁剪函数"""
# def ActionPrimitiveDataset_collate_fn_(batch):
# 	pass



