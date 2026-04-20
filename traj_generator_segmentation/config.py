# -*- coding: utf-8 -*-
"""traj_generator_segmentation 的 YAML 配置加载层。"""

from pathlib import Path
import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'traj_generator_segmentation.yaml'


# 读取 YAML 配置文件并返回顶层字典。
def _load_config():
	if not CONFIG_PATH.is_file():
		raise FileNotFoundError(f'Config file not found: {CONFIG_PATH}')

	with CONFIG_PATH.open('r', encoding='utf-8') as file_obj:
		config = yaml.safe_load(file_obj)

	if config is None:
		raise ValueError(f'Config file is empty: {CONFIG_PATH}')
	if not isinstance(config, dict):
		raise ValueError(f'Config file must contain a top-level mapping: {CONFIG_PATH}')
	return config


# 递归展开嵌套配置，产出 (路径, 叶子值) 形式的条目。
def _flatten_config_items(node, path=()):
	if isinstance(node, dict):
		for key, value in node.items():
			if not isinstance(key, str):
				raise ValueError(f'Config keys must be strings in {CONFIG_PATH}')
			yield from _flatten_config_items(value, path + (key,))
		return

	if not path:
		raise ValueError(f'Config file must contain at least one leaf value: {CONFIG_PATH}')
	yield path, node


# 将叶子键名转换为模块级常量名，例如 run_min_phase_len -> RUN_MIN_PHASE_LEN。
def _leaf_key_to_constant_name(path):
	return path[-1].upper()


# 根据 YAML 叶子节点自动构建可导出的模块常量，并检查重名冲突。
def _build_exported_constants(config):
	exported = {}
	origins = {}

	for path, value in _flatten_config_items(config):
		name = _leaf_key_to_constant_name(path)
		if name in exported:
			prev = '.'.join(origins[name])
			curr = '.'.join(path)
			raise ValueError(
				f'Duplicate exported config key "{name}" from "{prev}" and "{curr}" in {CONFIG_PATH}')
		exported[name] = value
		origins[name] = path

	return exported


# 返回 YAML 配置的原始字典。
def load_traj_generator_segmentation_config():
	return CONFIG


# 按路径读取原始配置值；路径不存在时返回 default。
def get_config_value(*path, default=None):
	if not path:
		return CONFIG

	node = CONFIG
	for key in path:
		if not isinstance(node, dict) or key not in node:
			return default
		node = node[key]
	return node


CONFIG = _load_config()
EXPORTED_CONSTANTS = _build_exported_constants(CONFIG)
globals().update(EXPORTED_CONSTANTS)


__all__ = [
	'CONFIG_PATH',
	'CONFIG',
	'EXPORTED_CONSTANTS',
	'load_traj_generator_segmentation_config',
	'get_config_value',
	*sorted(EXPORTED_CONSTANTS.keys()),
]
