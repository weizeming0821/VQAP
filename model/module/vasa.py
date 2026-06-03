from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .decoder import FutureFramePredictor
from .encoder import ImageEncoder, VisualTransformerEncoder
from .flow_matching import VisualFlowMatchingHead

"""VASA 模块。

输入：
	__init__:
		model_args: Dict[str, Any]
	forward:
		selected_views: 长度为 B 的列表，第 b 个元素是长度为 K 的视角结果列表。
		noisy_ee_trajectory: [B, T_action, 9]，输入视觉条件 Flow Matching Head。
		timestep: [B] 或 [B, 1]，Flow Matching 流时间。
		trajectory_mask: [B, T_action]，True 表示有效动作帧。
		global_codeword: [B, 256]，输入未来帧 patch 潜空间预测支路。

输出：
	forward:
		start_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)，其中 P 是 patch 数量。
		end_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)
		img_diff_features: [B, 1, 512] (feature_type=cls) 或 [B, P, 512] (feature_type=patch)
		view_weights: [B, K]
		pred_end_patch_features: [B, 1, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)。
		position_vector_field: [B, T_action, 3]
		rotation_vector_field: [B, T_action, 6]
		gripper_logit: [B, T_action, 1]
"""
class VASA(nn.Module):

	def __init__(self, model_args: Dict[str, Any]):
		super().__init__()
		self.model_args = model_args["VASA"] if "VASA" in model_args else model_args

		transformer_encoder_cfg = self.model_args["transformer_encoder"]
		future_predictor_cfg = self.model_args["future_predictor"]

		self.image_encoder = ImageEncoder(dinov2_cfg=self.model_args["dinov2"])
		self.visual_transformer_encoder = VisualTransformerEncoder(
			input_dim=int(self.model_args["dinov2"]["feature_dim"]),
			hidden_dim=int(transformer_encoder_cfg["hidden_dim"]),
			num_self_attention_layers=int(transformer_encoder_cfg["num_self_attention_layers"]),
			num_heads=int(transformer_encoder_cfg["num_heads"]),
			ffn_dim=int(transformer_encoder_cfg["ffn_dim"]),
			dropout=float(transformer_encoder_cfg["dropout"]),
			norm_type=str(transformer_encoder_cfg.get("norm_type", "layernorm")),
		)
		self.flow_matching_head = VisualFlowMatchingHead(config=self.model_args["flow_matching_head"])
		self.future_predictor = FutureFramePredictor(
			global_code_dim=int(future_predictor_cfg["global_code_dim"]),
			image_feature_dim=int(future_predictor_cfg["image_feature_dim"]),
			ffn_hidden_dim=int(future_predictor_cfg["ffn_hidden_dim"]),
		)

	def forward(
		self,
		selected_views: Sequence[Sequence[Dict[str, Any]]],
		noisy_ee_trajectory: torch.Tensor,
		timestep: torch.Tensor,
		trajectory_mask: torch.Tensor,
		global_codeword: torch.Tensor,
	) -> Dict[str, torch.Tensor]:

		# 始末帧图像特征提取：
		# feature_type=cls   -> [B, 768]
		# feature_type=patch -> [B, P, 768]
		start_img_features, end_img_features, view_weights = self.image_encoder(selected_views)

		# 结尾帧特征查询起始帧特征，输出视觉变化特征：
		# [B, 768]/[B, P, 768] -> [B, 1, 512]/[B, P, 512]
		img_diff_features = self.visual_transformer_encoder(
			end_img_features=end_img_features,
			start_img_features=start_img_features,
		)


		# 未来帧 patch 潜空间预测支路：
		# global_codeword [B, 256] + start_img_features [B, 768]/[B, P, 768]
		# -> pred_end_patch_features [B, 1, 768]/[B, P, 768]
		pred_end_patch_features = self.future_predictor(
			start_img_features=start_img_features,
			global_codeword=global_codeword,
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
			"pred_end_patch_features": pred_end_patch_features,
		}
		model_outputs.update(flow_matching_outputs)

		return model_outputs
