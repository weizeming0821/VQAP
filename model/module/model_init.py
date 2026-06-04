from typing import Set

import torch.nn as nn

from .utils import RMSNorm


_INIT_METHOD_NAME = "init_parameters"


"""对 VQAP 模型执行统一初始化。

约定：
	1. 预训练子树不参与统一初始化。
	2. 先执行统一基础初始化。
	3. 再调用各模块的 init_parameters() 做二次覆盖。
"""
def apply_vqap_initialization(model: nn.Module) -> None:
	skipped_module_ids = _collect_pretrained_module_ids(model)
	_apply_base_initialization(model, skipped_module_ids)
	_apply_specialized_initialization(model, skipped_module_ids)


"""统一基础模块初始化。"""
def _apply_base_initialization(model: nn.Module, skipped_module_ids: Set[int]) -> None:
	for module in model.modules():
		if id(module) in skipped_module_ids:
			continue

		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			nn.init.ones_(module.weight)
			nn.init.zeros_(module.bias)
		elif isinstance(module, RMSNorm):
			nn.init.ones_(module.weight)


"""执行各模块的项目特化初始化覆盖。"""
def _apply_specialized_initialization(model: nn.Module, skipped_module_ids: Set[int]) -> None:
	for module in model.modules():
		if id(module) in skipped_module_ids:
			continue

		init_method = getattr(module, _INIT_METHOD_NAME, None)
		if callable(init_method):
			init_method()


"""收集所有预训练子树的模块 id，避免统一初始化覆盖已有权重。"""
def _collect_pretrained_module_ids(model: nn.Module) -> Set[int]:
	skipped_module_ids: Set[int] = set()

	for module in model.modules():
		backbone = getattr(module, "backbone", None)
		if isinstance(backbone, nn.Module):
			for pretrained_submodule in backbone.modules():
				skipped_module_ids.add(id(pretrained_submodule))

	return skipped_module_ids
