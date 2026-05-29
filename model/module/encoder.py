import torch
import torch.nn as nn

try:
	from .utils import ChannelAttention, MultiHeadAttention, RMSNorm, RotaryPositionEncoding1D
except ImportError:
	from utils import ChannelAttention, MultiHeadAttention, RMSNorm, RotaryPositionEncoding1D


def build_transformer_norm(hidden_dim: int, norm_type: str) -> nn.Module:
	normalized_norm_type = str(norm_type).strip().lower()
	if normalized_norm_type == "layernorm":
		return nn.LayerNorm(hidden_dim)
	if normalized_norm_type == "rmsnorm":
		return RMSNorm(hidden_dim)
	raise ValueError(f"Unsupported norm_type: {norm_type}")


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
		self.attention_norm = build_transformer_norm(hidden_dim, self.norm_type)
		self.self_attention = MultiHeadAttention(
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			dropout=dropout,
		)
		self.attention_dropout = nn.Dropout(dropout)

		self.ffn_norm = build_transformer_norm(hidden_dim, self.norm_type)
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