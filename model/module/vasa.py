from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .encoder import ImageEncoder, VisualTransformerEncoder


"""VASA 模块。

当前实现包含两部分：
1. 多视角图像特征提取与融合；
2. 视觉交互 Transformer 编码器，用结尾帧特征查询起始帧特征并输出视觉条件序列。

输入：
	__init__:
		model_args: Dict[str, Any]
	forward:
		selected_views: 长度为 B 的列表，第 b 个元素是长度为 K 的视角结果列表。

输出：
	forward:
		start_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)，其中 P 是 patch 数量。
		end_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)
		img_diff_features: [B, 1, 512] (feature_type=cls) 或 [B, P, 512] (feature_type=patch)
		view_weights: [B, K]
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

	def forward(self, selected_views: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
		start_img_features, end_img_features, view_weights = self.image_encoder(selected_views)
		img_diff_features = self.visual_transformer_encoder(
			end_img_features=end_img_features,
			start_img_features=start_img_features,
		)
		
		model_outputs = {
			"start_img_features": start_img_features,
			"end_img_features": end_img_features,
			"img_diff_features": img_diff_features,
			"view_weights": view_weights,
		}
		
		return model_outputs
