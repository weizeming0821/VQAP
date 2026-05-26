from typing import Optional

import torch
import torch.nn as nn

try:
	from .utils import MultiHeadAttention, RotaryPositionEncoding1D
except ImportError:
	from utils import MultiHeadAttention, RotaryPositionEncoding1D


"""Transformer Encoder 单层。

输入：
	__init__:
		hidden_dim: int，输入与输出维度 C。
		num_heads: int，注意力头数 H。
		ffn_dim: int，前馈网络中间维度。
		dropout: float，dropout 概率。
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

	def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.attention_norm = nn.LayerNorm(hidden_dim)
		self.self_attention = MultiHeadAttention(
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			dropout=dropout,
		)
		self.attention_dropout = nn.Dropout(dropout)

		self.ffn_norm = nn.LayerNorm(hidden_dim)
		self.ffn = nn.Sequential(
			nn.Linear(hidden_dim, ffn_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(ffn_dim, hidden_dim),
		)
		self.ffn_dropout = nn.Dropout(dropout)

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
	) -> None:
		super().__init__()
		if hidden_dim % num_heads != 0:
			raise ValueError("hidden_dim must be divisible by num_heads")

		self.hidden_dim = int(hidden_dim)
		self.num_layers = int(num_layers)
		self.num_heads = int(num_heads)
		self.head_dim = self.hidden_dim // self.num_heads

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
