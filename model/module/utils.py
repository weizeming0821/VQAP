import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

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
		rope_cos/sin: [1, T, D_h]
	输出：
		attention_output: [B, T_q, C]
	"""
	def forward(
		self,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		trajectory_mask: torch.Tensor,
		rope_cos: torch.Tensor,
		rope_sin: torch.Tensor,
	) -> torch.Tensor:

		batch_size, query_length, _ = query.shape
		key_length = key.shape[1]

		# [B, T, C] -> [B, H, T, D_h]
		query_states = self.q_proj(query).view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = self.k_proj(key).view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
		value_states = self.v_proj(value).view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)

		# RoPE 位置编码
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


