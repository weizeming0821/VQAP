import torch
import torch.nn as nn


"""未来帧 patch 潜空间预测 FFN。

输入：
	__init__:
		feature_dim: int，输入与输出 patch 特征维度 C。
		hidden_dim: int，中间层隐藏维度。
	forward:
		patch_features: [B, P, C]

输出：
	forward:
		predicted_patch_features: [B, P, C]
"""
class FuturePredictionFFN(nn.Module):

	def __init__(self, feature_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.feature_dim = int(feature_dim)
		self.hidden_dim = int(hidden_dim)
		self.fc1 = nn.Linear(self.feature_dim, self.hidden_dim)
		self.act1 = nn.GELU()
		self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.act2 = nn.GELU()
		self.fc3 = nn.Linear(self.hidden_dim, self.feature_dim)

	def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
		hidden_features = self.act1(self.fc1(patch_features))	# [B, P, C] -> [B, P, H]
		hidden_features = self.act2(self.fc2(hidden_features))	# [B, P, H] -> [B, P, H]
		predicted_patch_features = self.fc3(hidden_features)	# [B, P, H] -> [B, P, C]
		return predicted_patch_features


"""未来帧 patch 潜空间预测器。

输入：
	__init__:
		global_code_dim: int，全局语义码维度 D_g。
		image_feature_dim: int，图像 patch 特征维度 C。
		ffn_hidden_dim: int，逐 patch 共享 FFN 的隐藏维度。
	forward:
		start_img_features: [B, 768] (CLS 回退模式) 或 [B, P, 768] (patch 模式)
		global_codeword: [B, 256]

输出：
	forward:
		pred_end_patch_features: [B, 1, 768] (CLS 回退模式) 或 [B, P, 768] (patch 模式)
"""
class FutureFramePredictor(nn.Module):

	def __init__(self, global_code_dim: int, image_feature_dim: int, ffn_hidden_dim: int) -> None:
		super().__init__()
		self.global_code_dim = int(global_code_dim)
		self.image_feature_dim = int(image_feature_dim)
		self.feature_norm = nn.LayerNorm(self.image_feature_dim)
		self.film_linear = nn.Linear(self.global_code_dim, self.image_feature_dim * 2)
		self.ffn = FuturePredictionFFN(
			feature_dim=self.image_feature_dim,
			hidden_dim=int(ffn_hidden_dim),
		)

	"""将 CLS 或 patch 特征统一整理为 patch 序列形式。

	维度变化：
		[B, C] -> [B, 1, C]
		[B, P, C] -> [B, P, C]
	"""
	def _to_patch_sequence(self, image_features: torch.Tensor) -> torch.Tensor:
		if image_features.ndim == 2:
			if image_features.shape[-1] != self.image_feature_dim:
				raise ValueError(f"start_img_features last dim must be {self.image_feature_dim}")
			return image_features.unsqueeze(1)
		if image_features.ndim == 3:
			if image_features.shape[-1] != self.image_feature_dim:
				raise ValueError(f"start_img_features last dim must be {self.image_feature_dim}")
			return image_features
		raise ValueError(
			f"start_img_features must have shape [B, {self.image_feature_dim}] or "
			f"[B, P, {self.image_feature_dim}]"
		)

	def forward(
		self,
		start_img_features: torch.Tensor,
		global_codeword: torch.Tensor,
	) -> torch.Tensor:
		start_patch_features = self._to_patch_sequence(start_img_features)	# [B, 768]/[B, P, 768] -> [B, P, 768]

		if global_codeword.ndim != 2 or global_codeword.shape[-1] != self.global_code_dim:
			raise ValueError(f"global_codeword must have shape [B, {self.global_code_dim}]")
		if global_codeword.shape[0] != start_patch_features.shape[0]:
			raise ValueError("global_codeword and start_img_features must share the same batch size")

		global_codeword = global_codeword.to(
			device=start_patch_features.device,
			dtype=start_patch_features.dtype,
		)

		film_parameters = self.film_linear(global_codeword)	# [B, D_g] -> [B, 2C]
		gamma, beta = torch.chunk(film_parameters, chunks=2, dim=-1)	# [B, 2C] -> [B, C] + [B, C]
		gamma = gamma.unsqueeze(1)	# [B, C] -> [B, 1, C]
		beta = beta.unsqueeze(1)	# [B, C] -> [B, 1, C]

		normalized_start_features = self.feature_norm(start_patch_features)	# [B, P, C] -> [B, P, C]
		film_features = (1.0 + gamma) * normalized_start_features + beta	# [B, P, C] -> [B, P, C]
		pred_end_patch_features = self.ffn(film_features)	# [B, P, C] -> [B, P, C]
		return pred_end_patch_features
