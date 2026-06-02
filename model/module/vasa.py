from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .encoder import ImageEncoder


"""VASA 模块。

当前仅实现多视角图像特征提取与融合，支持 CLS / patch 两种 DINOv2 输出模式。

输入：
	__init__:
		model_args: Dict[str, Any]
	forward:
		selected_views: 长度为 B 的列表，第 b 个元素是长度为 K 的视角结果列表。

输出：
	forward:
		start_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)，其中 P 是 patch 数量。
		end_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)
		view_weights: [B, K]
"""
class VASA(nn.Module):

	def __init__(self, model_args: Dict[str, Any]):
		super().__init__()
		self.model_args = model_args["VASA"] if "VASA" in model_args else model_args
		self.image_encoder = ImageEncoder(dinov2_cfg=self.model_args["dinov2"])

	def forward(self, selected_views: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
		start_img_features, end_img_features, view_weights = self.image_encoder(selected_views)
		
		model_outputs = {
			"start_img_features": start_img_features,
			"end_img_features": end_img_features,
			"view_weights": view_weights,
		}
		
		return model_outputs
