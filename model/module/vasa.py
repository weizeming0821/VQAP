from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from .encoder import ImageEncoder, VisualTransformerEncoder
from .flow_matching import VisualFlowMatchingHead


"""VASA 模块。

输入：
	__init__:
		model_args: Dict[str, Any]
	forward:
		selected_views: 长度为 B 的列表，第 b 个元素是长度为 K 的视角结果列表。
		noisy_ee_trajectory: [B, T_action, 9]，可选；传入时执行视觉条件 Flow Matching Head。
		timestep: [B] 或 [B, 1]，可选；Flow Matching 流时间。
		trajectory_mask: [B, T_action]，可选；True 表示有效动作帧。

输出：
	forward:
		start_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)，其中 P 是 patch 数量。
		end_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)
		img_diff_features: [B, 1, 512] (feature_type=cls) 或 [B, P, 512] (feature_type=patch)
		view_weights: [B, K]
		position_vector_field: [B, T_action, 3]，仅执行 Flow Matching Head 时返回。
		rotation_vector_field: [B, T_action, 6]，仅执行 Flow Matching Head 时返回。
		gripper_logit: [B, T_action, 1]，仅执行 Flow Matching Head 时返回。
"""
class VASA(nn.Module):

	def __init__(self, model_args: Dict[str, Any]):
		super().__init__()
		self.model_args = model_args["VASA"] if "VASA" in model_args else model_args
		self.image_encoder = ImageEncoder(dinov2_cfg=self.model_args["dinov2"])
		self.visual_transformer_encoder = VisualTransformerEncoder(
			input_dim=int(self.model_args["dinov2"]["feature_dim"]),
			hidden_dim=int(self.model_args["transformer_encoder"]["hidden_dim"]),
			num_self_attention_layers=int(self.model_args["transformer_encoder"]["num_self_attention_layers"]),
			num_heads=int(self.model_args["transformer_encoder"]["num_heads"]),
			ffn_dim=int(self.model_args["transformer_encoder"]["ffn_dim"]),
			dropout=float(self.model_args["transformer_encoder"]["dropout"]),
			norm_type=str(self.model_args["transformer_encoder"].get("norm_type", "layernorm")),
		)
		self.flow_matching_head = VisualFlowMatchingHead(config=self.model_args["flow_matching_head"])

	def forward(
		self,
		selected_views: Sequence[Sequence[Dict[str, Any]]],
		noisy_ee_trajectory: Optional[torch.Tensor] = None,
		timestep: Optional[torch.Tensor] = None,
		trajectory_mask: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		
		# 始末帧图像特征提取
		start_img_features, end_img_features, view_weights = self.image_encoder(selected_views)

		# 结尾帧特征查询起始帧特征，输出视觉变化特征
		img_diff_features = self.visual_transformer_encoder(
			end_img_features=end_img_features,
			start_img_features=start_img_features,
		)
		
		# 判断是否执行视觉条件 Flow Matching Head，要求 noisy_ee_trajectory、timestep 和 trajectory_mask 必须同时提供或同时不提供。
		flow_matching_inputs = (noisy_ee_trajectory, timestep, trajectory_mask)
		has_flow_matching_input = any(input_tensor is not None for input_tensor in flow_matching_inputs)
		if has_flow_matching_input:
			if any(input_tensor is None for input_tensor in flow_matching_inputs):
				raise ValueError(
					"noisy_ee_trajectory, timestep and trajectory_mask must be provided together "
					"when running VASA Flow Matching Head"
				)

			# img_diff_features 作为条件，与 noisy_ee_trajectory、timestep 和 trajectory_mask 一起输入 Flow Matching Head，
			# 输出动作级位置和旋转变化向量场以及 gripper 开合 logits。
			flow_matching_outputs = self.flow_matching_head(
				noisy_ee_trajectory=noisy_ee_trajectory,
				timestep=timestep,
				visual_condition=img_diff_features,
				trajectory_mask=trajectory_mask,
			)

			model_outputs = {
			"start_img_features": start_img_features,
			"end_img_features": end_img_features,
			"img_diff_features": img_diff_features,
			"view_weights": view_weights,
			**flow_matching_outputs
		}
		else:
			model_outputs = {
				"start_img_features": start_img_features,
				"end_img_features": end_img_features,
				"img_diff_features": img_diff_features,
				"view_weights": view_weights,
			}
		
		return model_outputs
