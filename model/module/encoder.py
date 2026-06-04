from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

try:
	from .utils import ChannelAttention, MultiHeadAttention, RMSNorm, RotaryPositionEncoding1D, build_norm_layer
except ImportError:
	from utils import ChannelAttention, MultiHeadAttention, RMSNorm, RotaryPositionEncoding1D, build_norm_layer


SUPPORTED_DINOV2_MODELS = {
	"dinov2_vits14",
	"dinov2_vitb14",
	"dinov2_vitl14",
	"dinov2_vitg14",
	"dinov2_vits14_reg",
	"dinov2_vitb14_reg",
	"dinov2_vitl14_reg",
	"dinov2_vitg14_reg",
}


"""DINOv2 特征提取器。

输入：
	__init__:
		model_name: DINOv2 模型名称。
		repo: torch.hub 仓库地址。
		input_size: 输入图像尺寸。
		feature_dim: 输出特征维度。
		feature_type: 输出特征类型，`cls` 或 `patch`。
	forward:
		images: [N, 3, H, W]

输出：
	forward:
		feature_type=cls: image_features: [N, feature_dim]
		feature_type=patch: image_features: [N, num_patches, feature_dim]
"""
class DINO_FeatureExtractor(nn.Module):

	def __init__(
		self,
		model_name: str,
		repo: str,
		input_size: int,
		feature_dim: int,
		feature_type: str = "cls",
	) -> None:
		super().__init__()
		self.model_name = str(model_name)
		self.repo = str(repo)
		self.input_size = int(input_size)
		self.feature_dim = int(feature_dim)
		self.feature_type = str(feature_type).strip().lower()
		self._validate_model_config()

		try:
			self.backbone = torch.hub.load(self.repo, self.model_name)
		except Exception as exc:
			raise RuntimeError(
				"Failed to load DINOv2 model from torch.hub. "
				f"repo={self.repo}, model={self.model_name}"
			) from exc

	"""验证 DINOv2 模型配置的合法性。"""
	def _validate_model_config(self) -> None:
		if self.repo != "facebookresearch/dinov2":
			raise ValueError(
				"DINO_FeatureExtractor currently only supports the official DINOv2 hub repo: "
				"facebookresearch/dinov2"
			)
		if self.model_name not in SUPPORTED_DINOV2_MODELS:
			supported_models = ", ".join(sorted(SUPPORTED_DINOV2_MODELS))
			raise ValueError(
				"Unsupported DINOv2 model_name. "
				f"Got: {self.model_name}. Supported models: {supported_models}"
			)
		if self.input_size <= 0 or self.input_size % 14 != 0:
			raise ValueError("input_size must be a positive multiple of 14 for DINOv2")
		if self.feature_type not in {"cls", "patch"}:
			raise ValueError("feature_type must be either 'cls' or 'patch'")

	@property
	def device(self) -> torch.device:
		return next(self.backbone.parameters()).device

	@property
	def dtype(self) -> torch.dtype:
		return next(self.backbone.parameters()).dtype

	"""从 DINOv2 forward_features 输出中提取 CLS token 特征。"""
	def _extract_cls_features(self, features: Any) -> torch.Tensor:
		if isinstance(features, dict):
			if "x_norm_clstoken" in features:
				return features["x_norm_clstoken"]
			if "x_prenorm" in features:
				return features["x_prenorm"][:, 0]
			raise ValueError("DINOv2 forward_features output does not contain CLS features")
		return features

	"""从 DINOv2 forward_features 输出中提取 x_norm_patchtokens。"""
	def _extract_patch_features(self, features: Any) -> torch.Tensor:
		if not isinstance(features, dict):
			raise ValueError("Patch feature extraction requires dict output from DINOv2 forward_features")
		if "x_norm_patchtokens" in features:
			return features["x_norm_patchtokens"]
		raise ValueError("DINOv2 forward_features output does not contain patch features")

	def forward(self, images: torch.Tensor) -> torch.Tensor:
		if images.ndim != 4 or images.shape[1] != 3:
			raise ValueError("images must have shape [N, 3, H, W]")
		if images.shape[-2] != self.input_size or images.shape[-1] != self.input_size:
			raise ValueError(
				f"images must have spatial size [{self.input_size}, {self.input_size}]"
			)

		images = images.to(device=self.device, dtype=self.dtype)
		features = self.backbone.forward_features(images)
		if self.feature_type == "cls":
			image_features = self._extract_cls_features(features)
			if image_features.ndim != 2 or image_features.shape[-1] != self.feature_dim:
				raise ValueError(
					f"Expected CLS features with shape [N, {self.feature_dim}], got {tuple(image_features.shape)}"
				)
		else:
			image_features = self._extract_patch_features(features)
			if image_features.ndim != 3 or image_features.shape[-1] != self.feature_dim:
				raise ValueError(
					"Expected patch features with shape "
					f"[N, num_patches, {self.feature_dim}], got {tuple(image_features.shape)}"
				)
		return image_features


"""多视角图像编码器。

输入：
	__init__:
		dinov2_cfg: DINOv2 配置字典。
	forward:
		selected_views: 长度为 B 的列表，第 b 个元素是长度为 K 的视角结果列表。

输出：
	forward:
		fused_start_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)，其中 P 是 patch 数量。
		fused_end_img_features: [B, 768] (feature_type=cls) 或 [B, P, 768] (feature_type=patch)
		view_weights: [B, K]
"""
class ImageEncoder(nn.Module):

	def __init__(self, dinov2_cfg: Dict[str, Any]) -> None:
		super().__init__()
		self.feature_dim = int(dinov2_cfg["feature_dim"])
		self.feature_extractor = DINO_FeatureExtractor(
			model_name=str(dinov2_cfg["model_name"]),
			repo=str(dinov2_cfg["repo"]),
			input_size=int(dinov2_cfg["input_size"]),
			feature_dim=self.feature_dim,
			feature_type=str(dinov2_cfg.get("feature_type", "cls")),
		)

	"""检测输入的 selected_views 是否符合预期的批次格式，并返回每个样本的视角数量 K。"""
	def _validate_batch(self, selected_views: Sequence[Sequence[Dict[str, Any]]]) -> int:
		if not isinstance(selected_views, Sequence) or len(selected_views) == 0:
			raise ValueError("selected_views must be a non-empty batch of view lists")

		num_views = None
		for sample_views in selected_views:
			if not isinstance(sample_views, Sequence) or len(sample_views) == 0:
				raise ValueError("each sample in selected_views must contain at least one view")
			if num_views is None:
				num_views = len(sample_views)
			elif len(sample_views) != num_views:
				raise ValueError("all samples in a batch must share the same number of selected views")

		return int(num_views)

	"""从 selected_views 中提取指定 image_key 的图像数据，并堆叠成 [B, K, 3, H, W] 的张量。"""
	def _stack_image_batch(self, selected_views: Sequence[Sequence[Dict[str, Any]]], image_key: str) -> torch.Tensor:
		try:
			return torch.stack(
				[
					torch.stack([view_entry[image_key] for view_entry in sample_views], dim=0)
					for sample_views in selected_views
				],
				dim=0,
			)
		except Exception as exc:
			raise ValueError(
				f"Failed to stack {image_key} from selected_views. All images must share shape [3, H, W]"
			) from exc

	"""从 selected_views 中提取 best_score，并堆叠成 [B, K] 的张量。"""
	def _stack_score_batch(self, selected_views: Sequence[Sequence[Dict[str, Any]]]) -> torch.Tensor:
		score_rows = []
		for sample_views in selected_views:
			score_rows.append(
				torch.stack(
					[
						torch.as_tensor(view_entry["best_score"], dtype=torch.float32).reshape(())
						for view_entry in sample_views
					],
					dim=0,
				)
			)
		return torch.stack(score_rows, dim=0)

	def forward(self, selected_views: Sequence[Sequence[Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		num_views = self._validate_batch(selected_views)
		start_images = self._stack_image_batch(selected_views, image_key="best_start_image")
		end_images = self._stack_image_batch(selected_views, image_key="best_end_image")

		batch_size, _, channels, height, width = start_images.shape
		flat_start_images = start_images.reshape(batch_size * num_views, channels, height, width)
		flat_end_images = end_images.reshape(batch_size * num_views, channels, height, width)

		start_img_features = self.feature_extractor(flat_start_images)
		end_img_features = self.feature_extractor(flat_end_images)

		# 根据 feature_type 调整特征维度，准备后续的视角融合。
		if self.feature_extractor.feature_type == "cls":
			start_img_features = start_img_features.reshape(batch_size, num_views, self.feature_dim)
			end_img_features = end_img_features.reshape(batch_size, num_views, self.feature_dim)
		else:
			num_patches = start_img_features.shape[1]
			start_img_features = start_img_features.reshape(batch_size, num_views, num_patches, self.feature_dim)
			end_img_features = end_img_features.reshape(batch_size, num_views, num_patches, self.feature_dim)

		# 如果只有一个视角，则直接使用该视角的特征作为融合结果，权重为 1。
		if num_views == 1:
			view_weights = torch.ones(batch_size, 1, device=start_img_features.device, dtype=start_img_features.dtype)
		else:
			view_scores = self._stack_score_batch(selected_views).to(device=start_img_features.device, dtype=start_img_features.dtype)
			view_weights = view_scores / view_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

		# 根据 feature_type 进行加权融合，得到最终的图像特征表示。
		if self.feature_extractor.feature_type == "cls":
			fused_start_img_features = (start_img_features * view_weights.unsqueeze(-1)).sum(dim=1)
			fused_end_img_features = (end_img_features * view_weights.unsqueeze(-1)).sum(dim=1)
		else:
			fused_start_img_features = (start_img_features * view_weights[:, :, None, None]).sum(dim=1)
			fused_end_img_features = (end_img_features * view_weights[:, :, None, None]).sum(dim=1)

		return fused_start_img_features, fused_end_img_features, view_weights
		

"""视觉 Self-Attention 单层。

输入：
	__init__:
		hidden_dim: int，输入与输出维度 C。
		num_heads: int，注意力头数 H。
		ffn_dim: int，前馈网络中间维度。
		dropout: float，dropout 概率。
		norm_type: str，`layernorm` 或 `rmsnorm`。
	forward:
		query_features: [B, T, C]
		attention_mask: [B, T]，True 表示有效 token。

输出：
	forward:
		query_features: [B, T, C]
"""
class VisualSelfAttentionLayer(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_heads: int,
		ffn_dim: int,
		dropout: float = 0.0,
		norm_type: str = "layernorm",
	) -> None:
		super().__init__()
		self.attention_norm = build_norm_layer(hidden_dim, norm_type)
		self.self_attention = MultiHeadAttention(
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			dropout=dropout,
		)
		self.attention_dropout = nn.Dropout(dropout)

		self.ffn_norm = build_norm_layer(hidden_dim, norm_type)
		self.ffn = nn.Sequential(
			nn.Linear(hidden_dim, ffn_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(ffn_dim, hidden_dim),
		)
		self.ffn_dropout = nn.Dropout(dropout)

	"""执行单层视觉自注意力 FFN 的项目特化初始化。"""
	def init_parameters(self) -> None:
		first_linear = self.ffn[0]
		nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
		nn.init.zeros_(first_linear.bias)

	def forward(
		self,
		query_features: torch.Tensor,
		attention_mask: torch.Tensor,
	) -> torch.Tensor:
		if query_features.ndim != 3:
			raise ValueError("query_features must have shape [B, T, C]")
		if attention_mask.ndim != 2 or attention_mask.shape != query_features.shape[:2]:
			raise ValueError("attention_mask must have shape [B, T] matching query_features")

		attention_input = self.attention_norm(query_features)	# [B, T, C] -> [B, T, C]
		attention_output = self.self_attention(
			query=attention_input,
			key=attention_input,
			value=attention_input,
			trajectory_mask=attention_mask,
		)	# [B, T, C] -> [B, T, C]
		query_features = query_features + self.attention_dropout(attention_output)	# [B, T, C] -> [B, T, C]

		ffn_input = self.ffn_norm(query_features)	# [B, T, C] -> [B, T, C]
		ffn_output = self.ffn(ffn_input)	# [B, T, C] -> [B, T, C]
		query_features = self.ffn_dropout(ffn_output)	# [B, T, C] -> [B, T, C]
		
		return query_features


"""VASA 视觉交互 Transformer Encoder。

输入：
	__init__:
		input_dim: int，图像特征维度，默认来自 DINOv2 的 768 维输出。
		hidden_dim: int，视觉交互隐藏维度 C。
		num_self_attention_layers: int，cross-attention 之后的 self-attention 层数。
		num_heads: int，注意力头数 H。
		ffn_dim: int，前馈网络中间维度。
		dropout: float，dropout 概率。
		norm_type: str，`layernorm` 或 `rmsnorm`。
	forward:
		end_img_features: [B, 768] 或 [B, T, 768]
		start_img_features: [B, 768] 或 [B, T, 768]

输出：
	forward:
		img_diff_features: [B, 1, hidden_dim] (CLS 模式) 或 [B, P, hidden_dim] (patch 模式)
"""
class VisualTransformerEncoder(nn.Module):

	def __init__(
		self,
		input_dim: int,
		hidden_dim: int,
		num_self_attention_layers: int,
		num_heads: int,
		ffn_dim: int,
		dropout: float = 0.0,
		norm_type: str = "layernorm",
	) -> None:
		super().__init__()
		if hidden_dim % num_heads != 0:
			raise ValueError("hidden_dim must be divisible by num_heads")

		self.input_dim = int(input_dim)
		self.hidden_dim = int(hidden_dim)
		self.num_self_attention_layers = int(num_self_attention_layers)
		self.num_heads = int(num_heads)
		self.norm_type = str(norm_type).strip().lower()

		if self.norm_type not in {"layernorm", "rmsnorm"}:
			raise ValueError("norm_type must be either 'layernorm' or 'rmsnorm'")

		self.query_projection = nn.Linear(self.input_dim, self.hidden_dim)
		self.key_value_projection = nn.Linear(self.input_dim, self.hidden_dim)
		self.delta_projection = nn.Linear(self.input_dim, self.hidden_dim)

		self.cross_attention_query_norm = build_norm_layer(self.hidden_dim, self.norm_type)
		self.cross_attention_key_value_norm = build_norm_layer(self.hidden_dim, self.norm_type)
		self.cross_attention = MultiHeadAttention(
			hidden_dim=self.hidden_dim,
			num_heads=self.num_heads,
			dropout=float(dropout),
		)
		self.cross_attention_dropout = nn.Dropout(dropout)
		self.self_attention_layers = nn.ModuleList(
			[
				VisualSelfAttentionLayer(
					hidden_dim=self.hidden_dim,
					num_heads=self.num_heads,
					ffn_dim=int(ffn_dim),
					dropout=float(dropout),
					norm_type=self.norm_type,
				)
				for _ in range(self.num_self_attention_layers)
			]
		)

	"""将 [B, C] 或 [B, T, C] 图像特征统一整理为序列形式。

	维度变化：
		[B, C] -> [B, 1, C]
		[B, T, C] -> [B, T, C]
	"""
	def _to_token_sequence(self, image_features: torch.Tensor, feature_name: str) -> torch.Tensor:
		if image_features.ndim == 2:
			if image_features.shape[-1] != self.input_dim:
				raise ValueError(f"{feature_name} last dim must be {self.input_dim}")
			return image_features.unsqueeze(1)
		if image_features.ndim == 3:
			if image_features.shape[-1] != self.input_dim:
				raise ValueError(f"{feature_name} last dim must be {self.input_dim}")
			return image_features
		raise ValueError(f"{feature_name} must have shape [B, {self.input_dim}] or [B, T, {self.input_dim}]")

	def forward(
		self,
		end_img_features: torch.Tensor,
		start_img_features: torch.Tensor,
	) -> torch.Tensor:
		end_tokens = self._to_token_sequence(end_img_features, feature_name="end_img_features")	# [B, 768]/[B, T, 768] -> [B, T, 768]
		start_tokens = self._to_token_sequence(start_img_features, feature_name="start_img_features")	# [B, 768]/[B, T, 768] -> [B, T, 768]

		if end_tokens.shape != start_tokens.shape:
			raise ValueError("end_img_features and start_img_features must share the same [B, T, C] shape")

		delta_tokens = self.delta_projection(end_tokens - start_tokens)	# [B, T, 768] -> [B, T, hidden_dim]

		query_tokens = self.query_projection(end_tokens)	# [B, T, 768] -> [B, T, hidden_dim]
		key_value_tokens = self.key_value_projection(start_tokens)	# [B, T, 768] -> [B, T, hidden_dim]

		key_value_mask = torch.ones(
			key_value_tokens.shape[0],
			key_value_tokens.shape[1],
			device=key_value_tokens.device,
			dtype=torch.bool,
		)	# [B, T]

		cross_attention_query = self.cross_attention_query_norm(query_tokens)	# [B, T, hidden_dim] -> [B, T, hidden_dim]
		cross_attention_key_value = self.cross_attention_key_value_norm(key_value_tokens)	# [B, T, hidden_dim] -> [B, T, hidden_dim]
		cross_attention_output = self.cross_attention(
			query=cross_attention_query,
			key=cross_attention_key_value,
			value=cross_attention_key_value,
			trajectory_mask=key_value_mask,
		)	# [B, T, hidden_dim] -> [B, T, hidden_dim]
		img_diff_features = self.cross_attention_dropout(cross_attention_output) + delta_tokens	# [B, T, hidden_dim] + [B, T, hidden_dim] -> [B, T, hidden_dim]

		for layer in self.self_attention_layers:
			img_diff_features = layer(
				query_features=img_diff_features,
				attention_mask=key_value_mask,
			)	# [B, T, hidden_dim] -> [B, T, hidden_dim]
		return img_diff_features


"""通道编码模块。

输入：
	__init__:
		hidden_dim: int，输入输出维度 C。
		bottleneck_dim: int，通道注意力瓶颈维度。
	forward:
		x: [B, T, C]
		trajectory_mask: [B, T]，True 表示有效帧。

输出：
	forward:
		channel_encoded_x: [B, T, C]
"""
class ChannelEncoder(nn.Module):

	def __init__(self, hidden_dim: int, bottleneck_dim: int) -> None:
		super().__init__()
		self.hidden_dim = int(hidden_dim)
		self.channel_attention = ChannelAttention(
			feature_dim=self.hidden_dim,
			bottleneck_dim=int(bottleneck_dim),
		)
		self.output_ffn = nn.Sequential(
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.LayerNorm(self.hidden_dim),
			nn.GELU(),
			nn.Linear(self.hidden_dim, self.hidden_dim),
		)

	"""执行通道编码残差 FFN 的项目特化初始化。"""
	def init_parameters(self) -> None:
		first_linear = self.output_ffn[0]
		last_linear = self.output_ffn[-1]
		nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
		nn.init.zeros_(first_linear.bias)
		nn.init.zeros_(last_linear.weight)
		nn.init.zeros_(last_linear.bias)

	"""先做逐帧通道注意力，再用 trajectory_mask 屏蔽 padding 位置。

	维度变化：
		x: [B, T, C] -> ChannelAttention -> [B, T, C] -> FFN(2层LLN) -> [B, T, C]
	"""
	def forward(self, x: torch.Tensor, trajectory_mask: torch.Tensor) -> torch.Tensor:
		valid_mask = trajectory_mask.unsqueeze(-1).to(x.dtype)

		# [B, T, C] -> [B, T, C]
		x = x * valid_mask
		x = self.channel_attention(x)
		x = x * valid_mask

		# [B, T, C] -> [B, T, C]
		x = self.output_ffn(x)
		x = x * valid_mask
		return x


"""Transformer Encoder 单层。

输入：
	__init__:
		hidden_dim: int，输入与输出维度 C。
		num_heads: int，注意力头数 H。
		ffn_dim: int，前馈网络中间维度。
		dropout: float，dropout 概率。
		norm_type: str，`layernorm` 或 `rmsnorm`。
	forward:
		x: [B, T, C]
		trajectory_mask: [B, T]
		rope_cos: [1, T, D_h]
		rope_sin: [1, T, D_h]

输出：
	forward:
		x: [B, T, C]
"""
class TransformerEncoderLayer(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_heads: int,
		ffn_dim: int,
		dropout: float = 0.0,
		norm_type: str = "layernorm",
	) -> None:
		super().__init__()
		self.norm_type = str(norm_type).strip().lower()
		self.attention_norm = build_norm_layer(hidden_dim, self.norm_type)
		self.self_attention = MultiHeadAttention(
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			dropout=dropout,
		)
		self.attention_dropout = nn.Dropout(dropout)

		self.ffn_norm = build_norm_layer(hidden_dim, self.norm_type)
		self.ffn = nn.Sequential(
			nn.Linear(hidden_dim, ffn_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(ffn_dim, hidden_dim),
		)
		self.ffn_dropout = nn.Dropout(dropout)

	"""执行单层 Transformer 编码 FFN 的项目特化初始化。"""
	def init_parameters(self) -> None:
		first_linear = self.ffn[0]
		nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
		nn.init.zeros_(first_linear.bias)

	"""执行单层 Pre-LN Transformer 编码。

	维度变化：
		x: [B, T, C] -> self-attention -> [B, T, C] -> FFN -> [B, T, C]
	"""
	def forward(
		self,
		x: torch.Tensor,
		trajectory_mask: torch.Tensor,
		rope_cos: torch.Tensor,
		rope_sin: torch.Tensor,
	) -> torch.Tensor:
		valid_mask = trajectory_mask.unsqueeze(-1).to(x.dtype)

		# [B, T, C] -> [B, T, C]
		attention_input = self.attention_norm(x)
		attention_output = self.self_attention(
			query=attention_input,
			key=attention_input,
			value=attention_input,
			trajectory_mask=trajectory_mask,
			rope_cos=rope_cos,
			rope_sin=rope_sin,
		)
		x = x + self.attention_dropout(attention_output)
		x = x * valid_mask

		# [B, T, C] -> [B, T, C]
		ffn_input = self.ffn_norm(x)
		ffn_output = self.ffn(ffn_input)
		x = x + self.ffn_dropout(ffn_output)
		x = x * valid_mask
		return x


"""多层 Transformer Encoder。

输入：
	__init__:
		hidden_dim: int，输入输出维度 C。
		num_layers: int，编码层数。
		num_heads: int，注意力头数 H。
		ffn_dim: int，前馈网络中间维度。
		dropout: float，dropout 概率。
		rope_theta: float，RoPE 频率基数。
		rope_max_seq_len: int，RoPE 支持的最大序列长度。
		norm_type: str，`layernorm` 或 `rmsnorm`，默认 `layernorm`。
	forward:
		x: [B, T, C]
		trajectory_mask: [B, T]，True 表示有效帧。

输出：
	forward:
		encoded_x: [B, T, C]
"""
class TransformerEncoder(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_layers: int,
		num_heads: int,
		ffn_dim: int,
		dropout: float,
		rope_theta: float,
		rope_max_seq_len: int,
		norm_type: str = "layernorm",
	) -> None:
		super().__init__()
		if hidden_dim % num_heads != 0:
			raise ValueError("hidden_dim must be divisible by num_heads")

		self.hidden_dim = int(hidden_dim)
		self.num_layers = int(num_layers)
		self.num_heads = int(num_heads)
		self.head_dim = self.hidden_dim // self.num_heads
		self.norm_type = str(norm_type).strip().lower()
		if self.norm_type not in {"layernorm", "rmsnorm"}:
			raise ValueError("norm_type must be either 'layernorm' or 'rmsnorm'")

		self.rope = RotaryPositionEncoding1D(
			feature_dim=self.head_dim,
			theta=float(rope_theta),
			max_seq_len=int(rope_max_seq_len),
		)
		self.layers = nn.ModuleList(
			[
				TransformerEncoderLayer(
					hidden_dim=self.hidden_dim,
					num_heads=self.num_heads,
					ffn_dim=int(ffn_dim),
					dropout=float(dropout),
					norm_type=self.norm_type,
				)
				for _ in range(self.num_layers)
			]
		)

	"""执行多层 Transformer Encoder。

	维度变化：
		x: [B, T, C] -> N 层 Encoder -> encoded_x: [B, T, C]
	"""
	def forward(self, x: torch.Tensor, trajectory_mask: torch.Tensor) -> torch.Tensor:

		rope_cos, rope_sin = self.rope(seq_len=x.shape[1], device=x.device, dtype=x.dtype)
		for layer in self.layers:
			x = layer(x, trajectory_mask, rope_cos, rope_sin)
		return x
