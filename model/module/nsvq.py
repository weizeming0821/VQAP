from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
	from .utils import MultiHeadAttention, RotaryPositionEncoding1D
except ImportError:
	from utils import MultiHeadAttention, RotaryPositionEncoding1D


"""带 padding mask 的平均池化。

输入：
	sequence_features: [B, T, C]
	trajectory_mask: [B, T]，True 表示有效帧。

输出：
	pooled_features: [B, C]

维度变化：
	[B, T, C] --mask--> [B, T, C] --sum over T--> [B, C]
"""
def masked_average_pool(sequence_features: torch.Tensor, trajectory_mask: torch.Tensor) -> torch.Tensor:

	valid_mask = trajectory_mask.unsqueeze(-1).to(sequence_features.dtype)
	masked_features = sequence_features * valid_mask

	# [B, T, C] -> [B, C]。分母最小截断到 1，避免理论上的全 padding 样本导致除零。
	valid_counts = valid_mask.sum(dim=1).clamp_min(1.0)
	pooled_features = masked_features.sum(dim=1) / valid_counts
	return pooled_features


"""根据离散码索引计算 perplexity。

输入：
	codebook_indices: 任意形状的 long tensor。
	codebook_size: 码本大小 K。

输出：
	perplexity: 标量 tensor。
"""
def compute_perplexity(codebook_indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
	if codebook_indices.numel() == 0:
		return torch.zeros((), device=codebook_indices.device, dtype=torch.float32)

	flat_indices = codebook_indices.reshape(-1)
	code_histogram = torch.bincount(flat_indices, minlength=int(codebook_size)).to(dtype=torch.float32)
	code_probabilities = code_histogram / code_histogram.sum().clamp_min(1.0)

	non_zero_mask = code_probabilities > 0
	entropy = -(code_probabilities[non_zero_mask] * code_probabilities[non_zero_mask].log()).sum()
	return entropy.exp()


"""NSVQ 量化器。

输入：
	__init__:
		codebook_size: int，码本大小 K。
		codebook_dim: int，码本向量维度 D。
		replace_every: int，死码替换周期。
		discard_threshold: float，低利用率判定阈值。
		replace_noise_scale: float，替换时添加的小噪声尺度。
		eps: float，数值稳定项。
	forward:
		inputs: [N, D]

输出：
	forward:
		quantized_features: [N, D]
		codebook_indices: [N]
		perplexity: 标量 tensor
"""
class NSVQQuantizer(nn.Module):

	def __init__(
		self,
		codebook_size: int,
		codebook_dim: int,
		replace_every: int,
		discard_threshold: float,
		replace_noise_scale: float,
		eps: float = 1e-6,
	) -> None:
		super().__init__()
		if codebook_size <= 0:
			raise ValueError("codebook_size must be positive")
		if codebook_dim <= 0:
			raise ValueError("codebook_dim must be positive")

		self.codebook_size = int(codebook_size)
		self.codebook_dim = int(codebook_dim)
		self.replace_every = int(replace_every)
		self.discard_threshold = float(discard_threshold)
		self.replace_noise_scale = float(replace_noise_scale)
		self.eps = float(eps)

		# 码本直接作为可训练参数存在，不使用 EMA。
		self.codebooks = nn.Parameter(torch.empty(self.codebook_size, self.codebook_dim))  # [K, D]
		self.register_buffer(
			"codebooks_used",
			torch.zeros(self.codebook_size, dtype=torch.float32),  # [K]，每个码字累计被分配的次数
			persistent=True,
		)

	"""均匀分布初始化，范围与码本大小成反比。码本越大，初始化值越接近 0。"""
	def reset_parameters(self) -> None:
		init_bound = 1.0 / float(self.codebook_size)
		nn.init.uniform_(self.codebooks, -init_bound, init_bound)

	"""执行 NSVQ 码本的项目特化初始化。"""
	def init_parameters(self) -> None:
		self.reset_parameters()

	"""计算输入与整张码本的平方欧氏距离。

	维度变化：
		inputs: [N, D]
		codebooks: [K, D]
		squared_distances: [N, K]
	"""
	def compute_codebook_distances(self, inputs: torch.Tensor) -> torch.Tensor:
		if inputs.shape[-1] != self.codebook_dim:
			raise ValueError("inputs feature dim must match codebook_dim")

		input_norm = (inputs ** 2).sum(dim=-1, keepdim=True)	# [N, D] -> [N, 1]
		codebook_norm = (self.codebooks ** 2).sum(dim=-1).unsqueeze(0)	# [K, D] -> [1, K]
		squared_distances = input_norm + codebook_norm - 2.0 * inputs @ self.codebooks.t()	# [N, 1] + [1, K] - 2 * [N, D] @ [D, K] -> [N, K]
		return squared_distances

	"""按离散索引查表取码本向量。"""
	def lookup_codewords(self, codebook_indices: torch.Tensor) -> torch.Tensor:
		flat_indices = codebook_indices.reshape(-1)                                   # [...] -> [N]
		selected_codewords = self.codebooks.index_select(dim=0, index=flat_indices)   # [N] -> [N, D]
		return selected_codewords.view(*codebook_indices.shape, self.codebook_dim)    # [N, D] -> [..., D]

	"""输入检查与索引规范化。确保 codebook_indices 是整数类型且在合法范围内。"""
	def normalize_codebook_indices(self, codebook_indices: torch.Tensor) -> torch.Tensor:
		if torch.is_floating_point(codebook_indices) or codebook_indices.dtype == torch.bool:
			raise ValueError("codebook_indices must use an integer dtype")

		normalized_indices = codebook_indices.to(device=self.codebooks.device, dtype=torch.long)
		if normalized_indices.numel() == 0:
			return normalized_indices

		if normalized_indices.min() < 0 or normalized_indices.max() >= self.codebook_size:
			raise ValueError("codebook_indices must be in [0, codebook_size)")
		return normalized_indices

	"""统计本批次码字使用频次并累加到全局计数 buffer，供死码检测使用。"""
	@torch.no_grad()
	def update_codebook_usage(self, codebook_indices: torch.Tensor) -> None:
		flat_indices = codebook_indices.reshape(-1)		# [N]，展平为一维，N = 批次内所有样本数
		usage_increment = torch.bincount(flat_indices, minlength=self.codebook_size) 	# [K]，第 k 位 = 码字 k 在本批次中被分配到的次数
		self.codebooks_used.add_(usage_increment.to(dtype=self.codebooks_used.dtype, device=self.codebooks_used.device))	# [K]

	"""训练时执行 NSVQ；推理时要求按给定索引直接查表。"""
	def forward(
		self,
		inputs: Optional[torch.Tensor] = None,
		codebook_indices: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# 独立查表路径：仅当提供 codebook_indices 时有效，inputs 会被忽略。
		if codebook_indices is not None:
			if self.training:
				raise ValueError("codebook_indices can only be provided during evaluation")

			normalized_indices = self.normalize_codebook_indices(codebook_indices)
			quantized_features = self.lookup_codewords(normalized_indices)
			perplexity = compute_perplexity(normalized_indices, self.codebook_size)
			return quantized_features, normalized_indices, perplexity

		if not self.training:
			raise ValueError("codebook_indices must be provided during evaluation")

		if inputs is None:
			raise ValueError("inputs must be provided when codebook_indices is not given")

		if inputs.shape[-1] != self.codebook_dim:
			raise ValueError("inputs feature dim must match codebook_dim")

		# [N, D] x [K, D] -> [N, K]，随后沿 K 维做硬最近邻分配。
		squared_distances = self.compute_codebook_distances(inputs)
		codebook_indices = torch.argmin(squared_distances, dim=-1)	# [N]，每个输入样本分配到的最近邻码字索引

		# [N] -> [N, D]，得到最近邻码本向量 e_{k*}。
		nearest_codewords = self.lookup_codewords(codebook_indices)

		# NSVQ：用与量化残差等范数的随机噪声替代不可微的硬量化残差。
		residual = inputs - nearest_codewords                                 # [N, D]，量化残差 h - e_{k*}
		random_noise = torch.randn_like(inputs)                               # [N, D]，标准正态噪声
		residual_norm = residual.norm(dim=-1, keepdim=True)                   # [N, 1]，每个样本的残差 L2 范数
		noise_norm = random_noise.norm(dim=-1, keepdim=True)                  # [N, 1]，每个样本的噪声 L2 范数
		vq_error = residual_norm / (noise_norm + self.eps) * random_noise     # [N, D]，缩放至与残差等范数，替代不可微的硬量化误差

		# 训练时返回 q = h + vq_error。
		quantized_features = inputs + vq_error                                # [N, D]
		self.update_codebook_usage(codebook_indices)

		perplexity = compute_perplexity(codebook_indices, self.codebook_size)
		return quantized_features, codebook_indices, perplexity

	"""每 replace_every 步检查一次码本使用情况，替换掉使用率低于 discard_threshold 的码本。"""
	@torch.no_grad()
	def replace_unused_codebooks(self) -> torch.Tensor:
		if self.replace_every <= 0:
			raise ValueError("replace_every must be positive when replace_unused_codebooks is enabled")

		usage_ratio = self.codebooks_used.to(dtype=torch.float32) / float(self.replace_every)  # [K]，每个码字在过去 replace_every 步内的平均使用率
		unused_mask = usage_ratio < self.discard_threshold                                      # [K]，bool，True = 低利用率死码
		dead_code_indices = unused_mask.nonzero(as_tuple=False).squeeze(-1)                    # [M]，死码下标，M = 死码数量
		num_replaced = dead_code_indices.numel()                                                # 标量，本次需替换的死码数

		if num_replaced == 0:
			self.codebooks_used.zero_()
			return torch.zeros((), device=self.codebooks.device, dtype=torch.long)

		active_code_indices = (~unused_mask).nonzero(as_tuple=False).squeeze(-1)               # [A]，活跃码字下标，A = 活跃码字数
		if active_code_indices.numel() == 0:
			# 若整张码本都低利用率，则整体加小扰动，避免从空活跃集合采样。
			self.codebooks.add_(self.replace_noise_scale * torch.randn_like(self.codebooks))   # [K, D]，in-place 加扰动
		else:
			# 从活跃码本中随机采样替换死码，添加小扰动后直接覆盖。这样做既能保持码本多样性，又能避免死码长期不更新。
			sampled_active_indices = active_code_indices[
				torch.randint(
					low=0,
					high=active_code_indices.numel(),
					size=(num_replaced,),
					device=self.codebooks.device,
				)
			]                                                                                   # [M]，从活跃集中有放回采样的下标
			replacement_codewords = self.codebooks.index_select(0, sampled_active_indices)     # [M, D]，对应的活跃码字
			replacement_codewords = replacement_codewords + self.replace_noise_scale * torch.randn_like(replacement_codewords)  # [M, D]，加小扰动避免完全重复
			self.codebooks.data.index_copy_(0, dead_code_indices, replacement_codewords)       # [K, D]，将死码位置覆盖为扰动后的替换码字

		self.codebooks_used.zero_()
		return torch.tensor(num_replaced, device=self.codebooks.device, dtype=torch.long)


"""全局语义码本模块。

输入：
	encoded_trajectory_features: [B, T, C]，训练或特征量化路径下必需。
	trajectory_mask: [B, T]，训练或特征量化路径下必需。
	codebook_indices: [B]，独立查表路径下可直接输入。

输出：
	特征量化路径：
		pooled_global_features: [B, C]，masked average pooling 后、投影前的全局特征。
		global_feature: [B, D]，投影到码本空间后的全局特征。
		global_codeword: [B, D]，NSVQ 量化后的全局码向量。
		global_codeindex: [B]，全局码索引。
		global_perplexity: 标量 tensor。
	独立查表路径：
		global_codeword: [B, D]
		global_codeindex: [B]
		global_perplexity: 标量 tensor。
"""
class GlobalCodebookModule(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		codebook_dim: int,
		codebook_size: int,
		replace_every: int,
		discard_threshold: float,
		replace_noise_scale: float,
		eps: float,
	) -> None:
		super().__init__()
		self.hidden_dim = int(hidden_dim)
		self.codebook_dim = int(codebook_dim)

		self.projection = nn.Linear(self.hidden_dim, self.codebook_dim)
		self.quantizer = NSVQQuantizer(
			codebook_size=int(codebook_size),
			codebook_dim=self.codebook_dim,
			replace_every=int(replace_every),
			discard_threshold=float(discard_threshold),
			replace_noise_scale=float(replace_noise_scale),
			eps=float(eps),
		)

	def lookup_codewords(self, codebook_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
		if codebook_indices.ndim != 1:
			raise ValueError("global codebook_indices must have shape [B]")

		quantized_global_features, global_codebook_indices, global_perplexity = self.quantizer(
			codebook_indices=codebook_indices,
		)
		return {
			"global_codeword": quantized_global_features,
			"global_codeindex": global_codebook_indices,
			"global_perplexity": global_perplexity,
		}

	def forward(
		self,
		encoded_trajectory_features: Optional[torch.Tensor] = None,
		trajectory_mask: Optional[torch.Tensor] = None,
		codebook_indices: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		if codebook_indices is not None and encoded_trajectory_features is None and trajectory_mask is None:
			return self.lookup_codewords(codebook_indices)

		if not self.training and codebook_indices is None:
			raise ValueError("global codebook_indices must be provided during evaluation")

		if encoded_trajectory_features is None or trajectory_mask is None:
			raise ValueError(
				"encoded_trajectory_features and trajectory_mask must be provided when using the feature quantization path"
			)

		batch_size = encoded_trajectory_features.shape[0]
		if codebook_indices is not None and codebook_indices.shape != (batch_size,):
			raise ValueError("global codebook_indices must have shape [B]")

		# [B, T, C] + [B, T] -> [B, C]
		pooled_global_features = masked_average_pool(encoded_trajectory_features, trajectory_mask)

		# [B, C] -> [B, D]
		projected_global_features = self.projection(pooled_global_features)

		# [B, D] + [B] -> [B, D] + [B] + Tensor
		quantized_global_features, global_codebook_indices, global_perplexity = self.quantizer(
			projected_global_features,
			codebook_indices=codebook_indices,
		)
		return {
			"pooled_global_features": pooled_global_features,
			"global_feature": projected_global_features,
			"global_codeword": quantized_global_features,
			"global_codeindex": global_codebook_indices,
			"global_perplexity": global_perplexity,
		}

	@torch.no_grad()
	def replace_unused_codebooks(self) -> torch.Tensor:
		return self.quantizer.replace_unused_codebooks()


"""细节码本模块。

输入：
	encoded_trajectory_features: [B, T, C]，训练或特征量化路径下必需。
	trajectory_mask: [B, T]，训练或特征量化路径下必需，只用于屏蔽 key/value 侧 padding 帧。
	codebook_indices: [B, N_detail]，独立查表路径下可直接输入。

输出：
	特征量化路径：
		detail_query_features: [B, N_detail, C]，cross-attention 后、投影前的细节 query 特征。
		detail_features: [B, N_detail, D]，投影到码本空间后的细节特征。
		detail_codewords: [B, N_detail, D]，NSVQ 量化后的细节码向量。
		detail_codeindices: [B, N_detail]，细节码索引矩阵。
		detail_perplexity: 标量 tensor。
	独立查表路径：
		detail_codewords: [B, N_detail, D]
		detail_codeindices: [B, N_detail]
		detail_perplexity: 标量 tensor。
"""
class DetailCodebookModule(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_queries: int,
		num_heads: int,
		codebook_dim: int,
		codebook_size: int,
		dropout: float,
		rope_theta: float,
		rope_max_seq_len: int,
		replace_every: int,
		discard_threshold: float,
		replace_noise_scale: float,
		eps: float,
	) -> None:
		super().__init__()
		if hidden_dim % num_heads != 0:
			raise ValueError("hidden_dim must be divisible by num_heads")

		self.hidden_dim = int(hidden_dim)
		self.num_queries = int(num_queries)
		self.num_heads = int(num_heads)
		self.codebook_dim = int(codebook_dim)
		self.codebook_size = int(codebook_size)
		self.head_dim = self.hidden_dim // self.num_heads

		# Learnable query 本身就承担“细节槽位”的语义承载与位置区分作用。
		self.learnable_queries = nn.Parameter(torch.randn(self.num_queries, self.hidden_dim) * 0.02)
		self.cross_attention = MultiHeadAttention(
			hidden_dim=self.hidden_dim,
			num_heads=self.num_heads,
			dropout=float(dropout),
		)
		self.rope = RotaryPositionEncoding1D(
			feature_dim=self.head_dim,
			theta=float(rope_theta),
			max_seq_len=max(int(rope_max_seq_len), self.num_queries),
		)
		self.projection = nn.Linear(self.hidden_dim, self.codebook_dim)
		self.quantizer = NSVQQuantizer(
			codebook_size=self.codebook_size,
			codebook_dim=self.codebook_dim,
			replace_every=int(replace_every),
			discard_threshold=float(discard_threshold),
			replace_noise_scale=float(replace_noise_scale),
			eps=float(eps),
		)

	"""执行细节查询槽位的项目特化初始化。"""
	def init_parameters(self) -> None:
		nn.init.normal_(self.learnable_queries, mean=0.0, std=0.02)

	def lookup_codewords(self, codebook_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
		if codebook_indices.ndim != 2 or codebook_indices.shape[1] != self.num_queries:
			raise ValueError("detail codebook_indices must have shape [B, num_queries]")

		quantized_detail_features, detail_codebook_indices, detail_perplexity = self.quantizer(
			codebook_indices=codebook_indices,
		)
		return {
			"detail_codewords": quantized_detail_features,
			"detail_codeindices": detail_codebook_indices,
			"detail_codeindexs": detail_codebook_indices,
			"detail_perplexity": detail_perplexity,
		}

	def forward(
		self,
		encoded_trajectory_features: Optional[torch.Tensor] = None,
		trajectory_mask: Optional[torch.Tensor] = None,
		codebook_indices: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		if codebook_indices is not None and encoded_trajectory_features is None and trajectory_mask is None:
			return self.lookup_codewords(codebook_indices)

		if not self.training and codebook_indices is None:
			raise ValueError("detail codebook_indices must be provided during evaluation")

		if encoded_trajectory_features is None or trajectory_mask is None:
			raise ValueError(
				"encoded_trajectory_features and trajectory_mask must be provided when using the feature quantization path"
			)

		batch_size, seq_len, _ = encoded_trajectory_features.shape
		if codebook_indices is not None and codebook_indices.shape != (batch_size, self.num_queries):
			raise ValueError("detail codebook_indices must have shape [B, num_queries]")

		# [N_detail, C] -> [1, N_detail, C] -> [B, N_detail, C]
		detail_queries = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)

		# query 长度为 N_detail，key/value 长度为 T；RoPE 统一按较长序列缓存后再切片。
		rope_seq_len = max(self.num_queries, seq_len)
		rope_cos, rope_sin = self.rope(seq_len=rope_seq_len, device=encoded_trajectory_features.device, dtype=encoded_trajectory_features.dtype)

		# [B, N_detail, C] x [B, T, C] -> [B, N_detail, C]
		detail_query_features = self.cross_attention(
			query=detail_queries,
			key=encoded_trajectory_features,
			value=encoded_trajectory_features,
			trajectory_mask=trajectory_mask,
			rope_cos=rope_cos,
			rope_sin=rope_sin,
		)

		# [B, N_detail, C] -> [B, N_detail, D]
		projected_detail_features = self.projection(detail_query_features)

		# 量化器按 [N, D] 处理，因此先展平成 [B * N_detail, D]，量化后再 reshape 回去。
		flat_projected_detail_features = projected_detail_features.reshape(batch_size * self.num_queries, self.codebook_dim)
		flat_quantized_detail_features, flat_detail_codebook_indices, detail_perplexity = self.quantizer(flat_projected_detail_features)

		quantized_detail_features = flat_quantized_detail_features.view(batch_size, self.num_queries, self.codebook_dim)	# [B, N_detail, D]
		detail_codebook_indices = flat_detail_codebook_indices.view(batch_size, self.num_queries)	# [B, N_detail]

		return {
			"detail_query_features": detail_query_features,
			"detail_features": projected_detail_features,
			"detail_codewords": quantized_detail_features,
			"detail_codeindices": detail_codebook_indices,
			"detail_codeindexs": detail_codebook_indices,
			"detail_perplexity": detail_perplexity,
		}

	@torch.no_grad()
	def replace_unused_codebooks(self) -> torch.Tensor:
		return self.quantizer.replace_unused_codebooks()
