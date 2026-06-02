import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


"""RMSNorm 归一化模块。

输入：
	__init__:
		feature_dim: int，输入特征维度。
		eps: float，数值稳定项。
	forward:
		x: [..., C]

输出：
	forward:
		与输入同形状的归一化结果。
"""
class RMSNorm(nn.Module):

	def __init__(self, feature_dim: int, eps: float = 1e-6) -> None:
		super().__init__()

		self.feature_dim = int(feature_dim)
		self.eps = float(eps)
		self.weight = nn.Parameter(torch.ones(self.feature_dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.shape[-1] != self.feature_dim:
			raise ValueError("input feature dim must match feature_dim")

		rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
		return x * rms * self.weight


"""按配置构造归一化层。

输入：
	feature_dim: int，隐藏维度。
	norm_type: str，`layernorm` 或 `rmsnorm`。
	eps: float，数值稳定项。

输出：
	归一化模块实例。
"""
def build_norm_layer(feature_dim: int, norm_type: str, eps: float = 1e-6) -> nn.Module:
	normalized_norm_type = str(norm_type).strip().lower()
	if normalized_norm_type == "layernorm":
		return nn.LayerNorm(feature_dim)
	if normalized_norm_type == "rmsnorm":
		return RMSNorm(feature_dim, eps=eps)
	raise ValueError(f"Unsupported norm_type: {norm_type}")


"""将序列 mask 作用到 [B, T, C] 张量。

输入：
	sequence_tensor: [B, T, C]
	trajectory_mask: [B, T]，True 表示有效帧。

输出：
	masked_tensor: [B, T, C]
"""
def apply_sequence_mask(sequence_tensor: torch.Tensor, trajectory_mask: torch.Tensor) -> torch.Tensor:
	if sequence_tensor.shape[:2] != trajectory_mask.shape:
		raise ValueError("sequence_tensor and trajectory_mask must share the same [B, T] shape")

	valid_mask = trajectory_mask.unsqueeze(-1).to(sequence_tensor.dtype)
	return sequence_tensor * valid_mask


"""Flow Matching 时间步正弦嵌入。

输入：
	__init__:
		embedding_dim: int，输出嵌入维度，必须为偶数。
		min_period: float，最小周期。
		max_period: float，最大周期。
	forward:
		timestep: [B] 或 [B, 1]。

输出：
	forward:
		time_embedding: [B, embedding_dim]
"""
class SinusoidalTimestepEmbedding(nn.Module):

	def __init__(self, embedding_dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> None:
		super().__init__()
		if embedding_dim <= 0 or embedding_dim % 2 != 0:
			raise ValueError("embedding_dim must be a positive even integer")

		self.embedding_dim = int(embedding_dim)
		self.min_period = float(min_period)
		self.max_period = float(max_period)

	def forward(self, timestep: torch.Tensor) -> torch.Tensor:
		if timestep.ndim == 2 and timestep.shape[-1] == 1:
			timestep = timestep.squeeze(-1)
		if timestep.ndim != 1:
			raise ValueError("timestep must have shape [B] or [B, 1]")

		half_dim = self.embedding_dim // 2
		fractions = torch.linspace(0.0, 1.0, half_dim, device=timestep.device, dtype=torch.float32)
		periods = self.min_period * (self.max_period / self.min_period) ** fractions
		radians = timestep.to(dtype=torch.float32).unsqueeze(-1) * (2.0 * math.pi / periods)
		return torch.cat((radians.sin(), radians.cos()), dim=-1)


"""Transformer 风格前馈网络。

输入：
	__init__:
		hidden_dim: int，输入输出维度。
		ffn_dim: int，中间隐藏维度。
		dropout: float，dropout 概率。
	forward:
		x: [B, T, C]

输出：
	forward:
		ffn_output: [B, T, C]
"""
class TransformerFFN(nn.Module):

	def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.input_linear = nn.Linear(hidden_dim, ffn_dim)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.output_linear = nn.Linear(ffn_dim, hidden_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.input_linear(x)
		x = self.activation(x)
		x = self.dropout(x)
		return self.output_linear(x)


"""根据条件向量生成 AdaRMSNorm 所需的 scale / shift / gate。

输入：
	__init__:
		condition_dim: int，条件向量维度。
		hidden_dim: int，目标隐藏维度。
	forward:
		condition: [B, condition_dim]

输出：
	forward:
		gamma: [B, 1, hidden_dim]
		beta: [B, 1, hidden_dim]
		gate: [B, 1, hidden_dim]
"""
class AdaptiveModulation(nn.Module):

	def __init__(self, condition_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.hidden_dim = int(hidden_dim)
		self.projection = nn.Linear(int(condition_dim), self.hidden_dim * 3)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.zeros_(self.projection.weight)
		nn.init.zeros_(self.projection.bias)

	def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if condition.ndim != 2:
			raise ValueError("condition must have shape [B, condition_dim]")

		gamma, beta, gate = self.projection(condition).chunk(3, dim=-1)
		return gamma.unsqueeze(1), beta.unsqueeze(1), gate.unsqueeze(1)


"""AdaRMSNorm 条件归一化模块。

输入：
	__init__:
		hidden_dim: int，输入隐藏维度。
		condition_dim: int，条件向量维度。
		eps: float，RMSNorm 数值稳定项。
	forward:
		x: [B, T, C]
		condition: [B, condition_dim]

输出：
	forward:
		conditioned_x: [B, T, C]
		gate: [B, 1, C]
"""
class AdaRMSNorm(nn.Module):

	def __init__(self, hidden_dim: int, condition_dim: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.norm = RMSNorm(hidden_dim, eps=eps)
		self.modulation = AdaptiveModulation(condition_dim, hidden_dim)

	def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		gamma, beta, gate = self.modulation(condition)
		conditioned_x = (1.0 + gamma) * self.norm(x) + beta
		return conditioned_x, gate

"""动作轨迹投影模块。

输入：
	__init__:
		input_dim: int，输入特征维度。
		hidden_dim: int，中间隐藏维度。
		output_dim: int，输出特征维度。
	forward:
		x: torch.Tensor，形状为 [B, T, input_dim]。

输出：
	forward:
		投影后的特征，形状为 [B, T, output_dim]。
"""
class TrajectoryProjectionMLP(nn.Module):

	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, output_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)


"""通道注意力模块。

输入：
	__init__:
		feature_dim: int，输入输出特征维度 C。
		bottleneck_dim: int，瓶颈层维度。
	forward:
		x: [B, T, C]

输出：
	forward:
		channel_encoded_x: [B, T, C]
"""
class ChannelAttention(nn.Module):

	"""初始化逐时间步通道注意力模块。"""
	def __init__(self, feature_dim: int, bottleneck_dim: int) -> None:
		super().__init__()
		self.feature_dim = int(feature_dim)
		self.bottleneck_dim = int(bottleneck_dim)
		self.channel_gate = nn.Sequential(
			nn.Linear(self.feature_dim, self.bottleneck_dim),
			nn.GELU(),
			nn.Linear(self.bottleneck_dim, self.feature_dim),
			nn.Sigmoid(),
		)

	"""对每个时间步独立执行通道注意力。

	维度变化：
		x: [B, T, C] -> [B*T, C] -> channel gate -> [B*T, C] -> [B, T, C]
	"""
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, seq_len, feature_dim = x.shape
		x_flat = x.reshape(batch_size * seq_len, feature_dim)
		channel_weights = self.channel_gate(x_flat)
		x_flat = x_flat + x_flat * channel_weights
		return x_flat.reshape(batch_size, seq_len, feature_dim)


"""1D RoPE 缓存模块。

输入：
	__init__:
		feature_dim: int，RoPE 对应的特征维度，必须是偶数。
		theta: float，RoPE 的频率基数。
		max_seq_len: int，支持的最大序列长度。
	forward:
		seq_len: int，当前序列长度。

输出：
	forward:
		cos: [1, T, D]
		sin: [1, T, D]

当前仅负责初始化和缓存时序位置对应的 cos/sin，后续注意力模块可直接复用。
"""
class RotaryPositionEncoding1D(nn.Module):

	"""初始化 RoPE 模块。"""
	def __init__(self, feature_dim: int, theta: float = 10000.0, max_seq_len: int = 2048) -> None:
		super().__init__()
		if feature_dim <= 0 or feature_dim % 2 != 0:
			raise ValueError("feature_dim must be a positive even integer")
		if max_seq_len <= 0:
			raise ValueError("max_seq_len must be a positive integer")

		self.feature_dim = int(feature_dim)
		self.theta = float(theta)
		self.max_seq_len = int(max_seq_len)

		inv_freq = 1.0 / (
			self.theta ** (torch.arange(0, self.feature_dim, 2, dtype=torch.float32) / self.feature_dim)
		)
		self.register_buffer("inv_freq", inv_freq, persistent=False)
		self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
		self.register_buffer("_sin_cached", torch.empty(0), persistent=False)
		self._cached_seq_len = 0

	"""对输入张量施加 RoPE。

	输入：
		x: [..., T, D] 或兼容广播的张量，D 必须为偶数。
		cos/sin: [1, T, D]。
	输出：
		与 x 同形状。
	"""
	@staticmethod
	def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
		if x.shape[-1] != cos.shape[-1] or x.shape[-1] != sin.shape[-1]:
			raise ValueError("x, cos and sin must share the same feature dimension")

		x_rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape_as(x).contiguous()
		return x * cos + x_rotated * sin

	"""构建指定长度的 RoPE cos/sin 缓存。"""
	def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
		positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
		freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device=device))

		# [T, D/2] -> [T, D]，保证奇偶位共享同一频率。
		cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1).unsqueeze(0)
		sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1).unsqueeze(0)

		self._cos_cached = cos.to(dtype=dtype)
		self._sin_cached = sin.to(dtype=dtype)
		self._cached_seq_len = seq_len

	"""返回指定序列长度的 RoPE cos/sin 缓存。

	输出：
		cos: [1, T, D]
		sin: [1, T, D]
	"""
	def forward(
		self,
		seq_len: int,
		device: Optional[torch.device] = None,
		dtype: Optional[torch.dtype] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		if seq_len <= 0:
			raise ValueError("seq_len must be a positive integer")
		if seq_len > self.max_seq_len:
			raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

		device = device or self.inv_freq.device
		dtype = dtype or self.inv_freq.dtype

		if (
			self._cached_seq_len < seq_len
			or self._cos_cached.device != device
			or self._cos_cached.dtype != dtype
		):
			self._build_cache(seq_len=seq_len, device=device, dtype=dtype)

		return self._cos_cached[:, :seq_len], self._sin_cached[:, :seq_len]


"""多头注意力模块。

输入：
	__init__:
		hidden_dim: int，输入与输出特征维度 C。
		num_heads: int，注意力头数 H。
		dropout: float，注意力权重的 dropout 概率。
	forward:
		query: [B, T_q, C]
		key: [B, T_k, C]
		value: [B, T_k, C]
		trajectory_mask: [B, T_k]，True 表示有效帧。
		rope_cos: [1, T, D_h]
		rope_sin: [1, T, D_h]

输出：
	forward:
		attention_output: [B, T_q, C]
"""
class MultiHeadAttention(nn.Module):

	"""初始化多头注意力模块。"""
	def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
		super().__init__()

		if hidden_dim % num_heads != 0:
			raise ValueError("hidden_dim must be divisible by num_heads")

		self.hidden_dim = int(hidden_dim)
		self.num_heads = int(num_heads)
		self.head_dim = self.hidden_dim // self.num_heads

		self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.attention_dropout = nn.Dropout(dropout)

	"""执行带 RoPE 和 padding mask 的多头注意力。

	输入：
		query: [B, T_q, C]
		key/value: [B, T_k, C]
		trajectory_mask: [B, T_k]
		rope_cos/sin: 可选，[1, T, D_h]；当为 None 时不使用 RoPE
	输出：
		attention_output: [B, T_q, C]
	"""
	def forward(
		self,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		trajectory_mask: torch.Tensor,
		rope_cos: Optional[torch.Tensor] = None,
		rope_sin: Optional[torch.Tensor] = None,
	) -> torch.Tensor:

		batch_size, query_length, _ = query.shape
		key_length = key.shape[1]
		if (rope_cos is None) != (rope_sin is None):
			raise ValueError("rope_cos and rope_sin must both be provided or both be None")

		# [B, T, C] -> [B, H, T, D_h]
		query_states = self.q_proj(query).view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = self.k_proj(key).view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
		value_states = self.v_proj(value).view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)

		# RoPE 为可选项；视觉 patch 交互编码器默认不使用额外位置编码。
		if rope_cos is not None and rope_sin is not None:
			query_states = RotaryPositionEncoding1D.apply_rotary(
				query_states,
				rope_cos[:, :query_length],
				rope_sin[:, :query_length],
			)
			key_states = RotaryPositionEncoding1D.apply_rotary(
				key_states,
				rope_cos[:, :key_length],
				rope_sin[:, :key_length],
			)

		# [B, H, T_q, D_h] x [B, H, D_h, T_k] -> [B, H, T_q, T_k]
		attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
		key_padding_mask = (~trajectory_mask).unsqueeze(1).unsqueeze(2)
		attention_scores = attention_scores.masked_fill(key_padding_mask, torch.finfo(attention_scores.dtype).min)

		# [B, H, T_q, T_k] -> [B, H, T_q, T_k]
		attention_weights = torch.softmax(attention_scores, dim=-1)
		attention_weights = attention_weights * trajectory_mask.unsqueeze(1).unsqueeze(2).to(attention_weights.dtype)
		attention_weights = self.attention_dropout(attention_weights)

		# [B, H, T_q, T_k] x [B, H, T_k, D_h] -> [B, H, T_q, D_h]
		attention_output = torch.matmul(attention_weights, value_states)
		# [B, H, T_q, D_h] -> [B, T_q, C]
		attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_length, self.hidden_dim)
		return self.out_proj(attention_output)


