from typing import Iterable, Tuple

import torch.nn as nn

from .decoder import FutureFramePredictor
from .encoder import ChannelEncoder, TransformerEncoderLayer, VisualSelfAttentionLayer
from .flow_matching import (
	FlowMatchingHead,
	FlowMatchingPredictionBranch,
	GripperPredictionBranch,
	VisualFlowMatchingHead,
)
from .utils import AdaptiveModulation, RMSNorm, TrajectoryProjectionMLP, TransformerFFN, ChannelAttention


_DINO_BACKBONE_KEYWORD = "image_encoder.feature_extractor.backbone"


"""对 VQAP 模型执行统一初始化。"""
def apply_vqap_initialization(model: nn.Module) -> None:
	_apply_base_initialization(model)
	_apply_specialized_initialization(model)


"""判断当前模块是否属于 DINOv2 预训练 backbone，应跳过随机初始化。"""
def _should_skip_module(module_name: str) -> bool:
	return _DINO_BACKBONE_KEYWORD in module_name


"""遍历所有需要初始化的模块。"""
def _iter_named_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
	for module_name, module in model.named_modules():
		if _should_skip_module(module_name):
			continue
		yield module_name, module


"""执行基础初始化规则。"""
def _apply_base_initialization(model: nn.Module) -> None:
	for _, module in _iter_named_modules(model):
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


"""执行模块级二次覆盖初始化。"""
def _apply_specialized_initialization(model: nn.Module) -> None:
	for _, module in _iter_named_modules(model):
		if isinstance(module, TrajectoryProjectionMLP):
			_init_trajectory_projection_mlp(module)
		elif isinstance(module, ChannelAttention):
			_init_channel_attention(module)
		elif isinstance(module, ChannelEncoder):
			_init_channel_encoder(module)
		elif isinstance(module, TransformerEncoderLayer):
			_init_sequential_ffn(module.ffn)
		elif isinstance(module, VisualSelfAttentionLayer):
			_init_sequential_ffn(module.ffn)
		elif isinstance(module, TransformerFFN):
			_init_transformer_ffn(module)
		elif isinstance(module, AdaptiveModulation):
			_init_adaptive_modulation(module)
		elif isinstance(module, FlowMatchingPredictionBranch):
			_init_prediction_branch(module)
		elif isinstance(module, GripperPredictionBranch):
			_init_gripper_branch(module)
		elif isinstance(module, FlowMatchingHead):
			_init_flow_matching_head(module)
		elif isinstance(module, VisualFlowMatchingHead):
			_init_visual_flow_matching_head(module)
		elif isinstance(module, FutureFramePredictor):
			_init_future_frame_predictor(module)


"""初始化动作特征投影 MLP。"""
def _init_trajectory_projection_mlp(module: TrajectoryProjectionMLP) -> None:
	first_linear = module.network[0]
	nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(first_linear.bias)


"""初始化通道注意力门控。"""
def _init_channel_attention(module: ChannelAttention) -> None:
	first_linear = module.channel_gate[0]
	last_linear = module.channel_gate[2]
	nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(first_linear.bias)
	nn.init.zeros_(last_linear.weight)
	nn.init.constant_(last_linear.bias, -4.0)


"""初始化 ChannelEncoder 残差 FFN。"""
def _init_channel_encoder(module: ChannelEncoder) -> None:
	first_linear = module.output_ffn[0]
	last_linear = module.output_ffn[-1]
	nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(first_linear.bias)
	nn.init.zeros_(last_linear.weight)
	nn.init.zeros_(last_linear.bias)


"""初始化由 nn.Sequential 表示的 GELU 前馈网络。"""
def _init_sequential_ffn(module: nn.Sequential) -> None:
	first_linear = module[0]
	nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(first_linear.bias)


"""初始化 TransformerFFN。"""
def _init_transformer_ffn(module: TransformerFFN) -> None:
	nn.init.kaiming_normal_(module.input_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(module.input_linear.bias)


"""初始化 AdaRMSNorm 调制层。"""
def _init_adaptive_modulation(module: AdaptiveModulation) -> None:
	nn.init.zeros_(module.projection.weight)
	nn.init.zeros_(module.projection.bias)


"""初始化位置 / 旋转速度场输出分支。"""
def _init_prediction_branch(module: FlowMatchingPredictionBranch) -> None:
	first_linear = module.output_mlp[0]
	last_linear = module.output_mlp[-1]
	nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(first_linear.bias)
	nn.init.normal_(last_linear.weight, std=0.01)
	nn.init.zeros_(last_linear.bias)


"""初始化夹爪预测分支。"""
def _init_gripper_branch(module: GripperPredictionBranch) -> None:
	first_linear = module.network[0]
	last_linear = module.network[-1]
	nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(first_linear.bias)
	nn.init.zeros_(last_linear.weight)
	nn.init.zeros_(last_linear.bias)


"""初始化 AtomAction_NSVQ 的 Flow Matching Head。"""
def _init_flow_matching_head(module: FlowMatchingHead) -> None:
	first_linear = module.condition_fusion[0]
	nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(first_linear.bias)


"""初始化 VASA 的视觉 Flow Matching Head。"""
def _init_visual_flow_matching_head(module: VisualFlowMatchingHead) -> None:
	first_linear = module.condition_fusion[0]
	nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(first_linear.bias)


"""初始化未来帧 patch 预测器。"""
def _init_future_frame_predictor(module: FutureFramePredictor) -> None:
	nn.init.zeros_(module.film_linear.weight)
	nn.init.zeros_(module.film_linear.bias)
	nn.init.kaiming_normal_(module.ffn.fc1.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(module.ffn.fc1.bias)
	nn.init.kaiming_normal_(module.ffn.fc2.weight, mode="fan_in", nonlinearity="relu")
	nn.init.zeros_(module.ffn.fc2.bias)
	nn.init.normal_(module.ffn.fc3.weight, std=0.01)
	nn.init.zeros_(module.ffn.fc3.bias)
