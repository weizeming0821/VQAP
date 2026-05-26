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
		x: [B, T, D] 或兼容广播的张量，D 必须为偶数。
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
