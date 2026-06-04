from typing import Any, Dict, Iterable, Optional, Set

import torch
import torch.nn as nn

from .utils import (
	AdaRMSNorm,
	MultiHeadAttention,
	RotaryPositionEncoding1D,
	SinusoidalTimestepEmbedding,
	TransformerFFN,
	apply_sequence_mask,
	build_norm_layer,
)


"""Flow Matching 主干单层。

输入：
	__init__:
		hidden_dim: int，主干隐藏维度。
		num_heads: int，注意力头数。
		ffn_dim: int，前馈隐藏维度。
		condition_dim: int，条件向量维度。
		dropout: float，dropout 概率。
		use_detail_cross_attention: bool，是否在本层注入细节码。
	forward:
		x: [B, T, C]
		condition: [B, C]
		detail_context: [B, N_detail, C]
		trajectory_mask: [B, T]
		detail_mask: [B, N_detail]
		rope_cos: [1, L, D_h]
		rope_sin: [1, L, D_h]

输出：
	forward:
		updated_x: [B, T, C]
"""
class FlowMatchingDecoderLayer(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_heads: int,
		ffn_dim: int,
		condition_dim: int,
		dropout: float,
		use_detail_cross_attention: bool,
	) -> None:
		super().__init__()
		self.use_detail_cross_attention = bool(use_detail_cross_attention)

		self.self_attention_norm = AdaRMSNorm(hidden_dim, condition_dim)
		self.self_attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
		self.self_attention_dropout = nn.Dropout(dropout)

		if self.use_detail_cross_attention:
			self.cross_attention_norm = AdaRMSNorm(hidden_dim, condition_dim)
			self.cross_attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
			self.cross_attention_dropout = nn.Dropout(dropout)

		self.ffn_norm = AdaRMSNorm(hidden_dim, condition_dim)
		self.ffn = TransformerFFN(hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout)
		self.ffn_dropout = nn.Dropout(dropout)

	def forward(
		self,
		x: torch.Tensor,
		condition: torch.Tensor,
		detail_context: torch.Tensor,
		trajectory_mask: torch.Tensor,
		detail_mask: torch.Tensor,
		rope_cos: torch.Tensor,
		rope_sin: torch.Tensor,
	) -> torch.Tensor:
		attention_input, attention_gate = self.self_attention_norm(x, condition)
		attention_output = self.self_attention(
			query=attention_input,
			key=attention_input,
			value=attention_input,
			trajectory_mask=trajectory_mask,
			rope_cos=rope_cos,
			rope_sin=rope_sin,
		)
		x = x + attention_gate * self.self_attention_dropout(attention_output)
		x = apply_sequence_mask(x, trajectory_mask)

		if self.use_detail_cross_attention:
			cross_attention_input, cross_attention_gate = self.cross_attention_norm(x, condition)
			cross_attention_output = self.cross_attention(
				query=cross_attention_input,
				key=detail_context,
				value=detail_context,
				trajectory_mask=detail_mask,
				rope_cos=rope_cos,
				rope_sin=rope_sin,
			)
			x = x + cross_attention_gate * self.cross_attention_dropout(cross_attention_output)
			x = apply_sequence_mask(x, trajectory_mask)

		ffn_input, ffn_gate = self.ffn_norm(x, condition)
		ffn_output = self.ffn(ffn_input)
		x = x + ffn_gate * self.ffn_dropout(ffn_output)
		x = apply_sequence_mask(x, trajectory_mask)
		return x


"""Flow Matching 主干解码器。

输入：
	__init__:
		hidden_dim: int，主干隐藏维度。
		num_layers: int，主干层数。
		num_heads: int，注意力头数。
		ffn_dim: int，前馈隐藏维度。
		condition_dim: int，条件向量维度。
		detail_injection_layers: Iterable[int]，注入细节码的层编号（1-based）。
		dropout: float，dropout 概率。
		rope_theta: float，RoPE 频率基数。
		rope_max_seq_len: int，RoPE 最大长度。
	forward:
		x: [B, T, C]
		condition: [B, C]
		detail_context: [B, N_detail, C]
		trajectory_mask: [B, T]

输出：
	forward:
		decoder_hidden: [B, T, C]
"""
class FlowMatchingDecoder(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_layers: int,
		num_heads: int,
		ffn_dim: int,
		condition_dim: int,
		detail_injection_layers: Iterable[int],
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
		self.detail_injection_layers = self._normalize_detail_injection_layers(detail_injection_layers)

		self.rope = RotaryPositionEncoding1D(
			feature_dim=self.head_dim,
			theta=float(rope_theta),
			max_seq_len=int(rope_max_seq_len),
		)
		self.layers = nn.ModuleList(
			[
				FlowMatchingDecoderLayer(
					hidden_dim=self.hidden_dim,
					num_heads=self.num_heads,
					ffn_dim=int(ffn_dim),
					condition_dim=int(condition_dim),
					dropout=float(dropout),
					use_detail_cross_attention=layer_index in self.detail_injection_layers,
				)
				for layer_index in range(self.num_layers)
			]
		)

	"""对 detail_injection_layers 进行规范化，转换为 0-based 层索引集合，并验证合法性。"""
	def _normalize_detail_injection_layers(self, detail_injection_layers: Iterable[int]) -> Set[int]:
		normalized_layers = set()
		for layer_index in detail_injection_layers:
			integer_layer_index = int(layer_index)
			if integer_layer_index < 1 or integer_layer_index > self.num_layers:
				raise ValueError("detail_injection_layers must use 1-based layer indices within [1, num_layers]")
			normalized_layers.add(integer_layer_index - 1)
		return normalized_layers

	def forward(
		self,
		x: torch.Tensor,
		condition: torch.Tensor,
		detail_context: torch.Tensor,
		trajectory_mask: torch.Tensor,
	) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		detail_mask = torch.ones(
			batch_size,
			detail_context.shape[1],
			dtype=torch.bool,
			device=detail_context.device,
		)

		rope_seq_len = max(seq_len, detail_context.shape[1])
		rope_cos, rope_sin = self.rope(seq_len=rope_seq_len, device=x.device, dtype=x.dtype)

		for layer in self.layers:
			x = layer(x, condition, detail_context, trajectory_mask, detail_mask, rope_cos, rope_sin)
		return x


"""视觉序列条件 Flow Matching 主干单层。

与 AtomAction_NSVQ 的 FlowMatchingDecoderLayer 不同，本层的条件来源不是
global/detail codeword，而是 VASA 输出的 img_diff_features 序列。

输入：
	__init__:
		hidden_dim: int，主干隐藏维度 C。
		num_heads: int，注意力头数 H。
		ffn_dim: int，前馈隐藏维度。
		condition_dim: int，时间条件向量维度。
		dropout: float，dropout 概率。
		use_visual_cross_attention: bool，是否在本层注入视觉序列条件。
	forward:
		x: [B, T_action, C]
		condition: [B, C]
		visual_context: [B, T_vis, C]
		trajectory_mask: [B, T_action]
		visual_context_mask: [B, T_vis]
		rope_cos: [1, T_action, D_h]
		rope_sin: [1, T_action, D_h]

输出：
	forward:
		updated_x: [B, T_action, C]
"""
class VisualConditionedFlowMatchingDecoderLayer(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_heads: int,
		ffn_dim: int,
		condition_dim: int,
		dropout: float,
		use_visual_cross_attention: bool,
	) -> None:
		super().__init__()
		self.use_visual_cross_attention = bool(use_visual_cross_attention)

		self.self_attention_norm = AdaRMSNorm(hidden_dim, condition_dim)
		self.self_attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
		self.self_attention_dropout = nn.Dropout(dropout)

		if self.use_visual_cross_attention:
			self.cross_attention_norm = AdaRMSNorm(hidden_dim, condition_dim)
			self.cross_attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
			self.cross_attention_dropout = nn.Dropout(dropout)

		self.ffn_norm = AdaRMSNorm(hidden_dim, condition_dim)
		self.ffn = TransformerFFN(hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout)
		self.ffn_dropout = nn.Dropout(dropout)

	def forward(
		self,
		x: torch.Tensor,
		condition: torch.Tensor,
		visual_context: torch.Tensor,
		trajectory_mask: torch.Tensor,
		visual_context_mask: torch.Tensor,
		rope_cos: torch.Tensor,
		rope_sin: torch.Tensor,
	) -> torch.Tensor:
		
		# 动作轨迹内部建模：[B, T_action, C] -> self-attention -> [B, T_action, C]
		attention_input, attention_gate = self.self_attention_norm(x, condition)
		attention_output = self.self_attention(
			query=attention_input,
			key=attention_input,
			value=attention_input,
			trajectory_mask=trajectory_mask,
			rope_cos=rope_cos,
			rope_sin=rope_sin,
		)
		x = x + attention_gate * self.self_attention_dropout(attention_output)
		x = apply_sequence_mask(x, trajectory_mask)

		if self.use_visual_cross_attention:
			# 视觉条件注入：
			# Q: 动作 token [B, T_action, C]
			# K/V: 图像差分 token [B, T_vis, C]
			# 输出仍对齐动作时间步 [B, T_action, C]
			cross_attention_input, cross_attention_gate = self.cross_attention_norm(x, condition)
			cross_attention_output = self.cross_attention(
				query=cross_attention_input,
				key=visual_context,
				value=visual_context,
				trajectory_mask=visual_context_mask,
			)
			x = x + cross_attention_gate * self.cross_attention_dropout(cross_attention_output)
			x = apply_sequence_mask(x, trajectory_mask)

		# 逐动作时间步 FFN：[B, T_action, C] -> [B, T_action, C]
		ffn_input, ffn_gate = self.ffn_norm(x, condition)
		ffn_output = self.ffn(ffn_input)
		x = x + ffn_gate * self.ffn_dropout(ffn_output)
		x = apply_sequence_mask(x, trajectory_mask)
		return x


"""视觉序列条件 Flow Matching 主干解码器。

输入：
	__init__:
		hidden_dim: int，主干隐藏维度 C。
		num_layers: int，主干层数。
		num_heads: int，注意力头数 H。
		ffn_dim: int，前馈隐藏维度。
		condition_dim: int，时间条件向量维度。
		condition_injection_layers: Iterable[int]，视觉条件注入层编号（1-based）。
		dropout: float，dropout 概率。
		rope_theta: float，RoPE 频率基数。
		rope_max_seq_len: int，RoPE 最大长度。
	forward:
		x: [B, T_action, C]
		condition: [B, C]
		visual_context: [B, T_vis, C]
		trajectory_mask: [B, T_action]
		visual_context_mask: [B, T_vis]

输出：
	forward:
		decoder_hidden: [B, T_action, C]
"""
class VisualConditionedFlowMatchingDecoder(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_layers: int,
		num_heads: int,
		ffn_dim: int,
		condition_dim: int,
		condition_injection_layers: Iterable[int],
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
		self.condition_injection_layers = self._normalize_condition_injection_layers(condition_injection_layers)

		self.rope = RotaryPositionEncoding1D(
			feature_dim=self.head_dim,
			theta=float(rope_theta),
			max_seq_len=int(rope_max_seq_len),
		)
		self.layers = nn.ModuleList(
			[
				VisualConditionedFlowMatchingDecoderLayer(
					hidden_dim=self.hidden_dim,
					num_heads=self.num_heads,
					ffn_dim=int(ffn_dim),
					condition_dim=int(condition_dim),
					dropout=float(dropout),
					use_visual_cross_attention=layer_index in self.condition_injection_layers,
				)
				for layer_index in range(self.num_layers)
			]
		)

	"""将 1-based condition_injection_layers 转换为 0-based 层索引集合。"""
	def _normalize_condition_injection_layers(self, condition_injection_layers: Iterable[int]) -> Set[int]:
		normalized_layers = set()
		for layer_index in condition_injection_layers:
			integer_layer_index = int(layer_index)
			if integer_layer_index < 1 or integer_layer_index > self.num_layers:
				raise ValueError("condition_injection_layers must use 1-based layer indices within [1, num_layers]")
			normalized_layers.add(integer_layer_index - 1)
		return normalized_layers

	def forward(
		self,
		x: torch.Tensor,
		condition: torch.Tensor,
		visual_context: torch.Tensor,
		trajectory_mask: torch.Tensor,
		visual_context_mask: torch.Tensor,
	) -> torch.Tensor:
		rope_cos, rope_sin = self.rope(seq_len=x.shape[1], device=x.device, dtype=x.dtype)

		for layer in self.layers:
			x = layer(
				x=x,
				condition=condition,
				visual_context=visual_context,
				trajectory_mask=trajectory_mask,
				visual_context_mask=visual_context_mask,
				rope_cos=rope_cos,
				rope_sin=rope_sin,
			)
		return x


"""Flow Matching 输出分支中的单层注意力细化模块。

输入：
	__init__:
		hidden_dim: int，输入输出维度。
		num_heads: int，注意力头数。
		ffn_dim: int，前馈隐藏维度。
		dropout: float，dropout 概率。
		norm_type: str，归一化类型。
	forward:
		x: [B, T, C]
		trajectory_mask: [B, T]
		rope_cos: [1, T, D_h]
		rope_sin: [1, T, D_h]

输出：
	forward:
		refined_x: [B, T, C]
"""
class FlowMatchingRefinementLayer(nn.Module):

	def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float, norm_type: str) -> None:
		super().__init__()
		self.attention_norm = build_norm_layer(hidden_dim, norm_type)
		self.self_attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
		self.attention_dropout = nn.Dropout(dropout)

		self.ffn_norm = build_norm_layer(hidden_dim, norm_type)
		self.ffn = TransformerFFN(hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout)
		self.ffn_dropout = nn.Dropout(dropout)

	def forward(
		self,
		x: torch.Tensor,
		trajectory_mask: torch.Tensor,
		rope_cos: torch.Tensor,
		rope_sin: torch.Tensor,
	) -> torch.Tensor:
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
		x = apply_sequence_mask(x, trajectory_mask)

		ffn_input = self.ffn_norm(x)
		ffn_output = self.ffn(ffn_input)
		x = x + self.ffn_dropout(ffn_output)
		x = apply_sequence_mask(x, trajectory_mask)
		return x


"""Flow Matching 位置 / 旋转输出分支。

输入：
	__init__:
		hidden_dim: int，输入隐藏维度。
		num_layers: int，注意力细化层数。
		num_heads: int，注意力头数。
		ffn_dim: int，分支内部 FFN 隐藏维度。
		dropout: float，dropout 概率。
		rope_theta: float，RoPE 频率基数。
		rope_max_seq_len: int，RoPE 最大长度。
		output_dim: int，输出向量场维度。
	forward:
		hidden_states: [B, T, C]
		trajectory_mask: [B, T]

输出：
	forward:
		branch_hidden: [B, T, C]
		branch_prediction: [B, T, output_dim]
"""
class FlowMatchingPredictionBranch(nn.Module):

	def __init__(
		self,
		hidden_dim: int,
		num_layers: int,
		num_heads: int,
		ffn_dim: int,
		dropout: float,
		rope_theta: float,
		rope_max_seq_len: int,
		output_dim: int,
	) -> None:
		super().__init__()
		if hidden_dim % num_heads != 0:
			raise ValueError("hidden_dim must be divisible by num_heads")

		self.hidden_dim = int(hidden_dim)
		self.num_heads = int(num_heads)
		self.head_dim = self.hidden_dim // self.num_heads
		self.rope = RotaryPositionEncoding1D(
			feature_dim=self.head_dim,
			theta=float(rope_theta),
			max_seq_len=int(rope_max_seq_len),
		)
		self.layers = nn.ModuleList(
			[
				FlowMatchingRefinementLayer(
					hidden_dim=self.hidden_dim,
					num_heads=self.num_heads,
					ffn_dim=int(ffn_dim),
					dropout=float(dropout),
					norm_type="rmsnorm",
				)
				for _ in range(int(num_layers))
			]
		)
		self.output_mlp = nn.Sequential(
			nn.Linear(self.hidden_dim, int(ffn_dim)),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(int(ffn_dim), int(output_dim)),
		)

	"""执行向量场输出头的项目特化初始化。"""
	def init_parameters(self) -> None:
		first_linear = self.output_mlp[0]
		last_linear = self.output_mlp[-1]
		nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
		nn.init.zeros_(first_linear.bias)
		nn.init.normal_(last_linear.weight, std=0.01)
		nn.init.zeros_(last_linear.bias)

	def forward(self, hidden_states: torch.Tensor, trajectory_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		branch_hidden = apply_sequence_mask(hidden_states, trajectory_mask)
		rope_cos, rope_sin = self.rope(seq_len=branch_hidden.shape[1], device=branch_hidden.device, dtype=branch_hidden.dtype)
		for layer in self.layers:
			branch_hidden = layer(branch_hidden, trajectory_mask, rope_cos, rope_sin)
		branch_hidden = apply_sequence_mask(branch_hidden, trajectory_mask)

		branch_prediction = self.output_mlp(branch_hidden)
		branch_prediction = apply_sequence_mask(branch_prediction, trajectory_mask)
		return branch_hidden, branch_prediction


"""夹爪开关预测分支。

输入：
	__init__:
		input_dim: int，输入隐藏维度。
		hidden_dim: int，中间隐藏维度。
		output_dim: int，输出维度，默认 1。
		dropout: float，dropout 概率。
	forward:
		position_hidden: [B, T, C]
		trajectory_mask: [B, T]

输出：
	forward:
		gripper_logit: [B, T, output_dim]
"""
class GripperPredictionBranch(nn.Module):

	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(int(input_dim), int(hidden_dim)),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(int(hidden_dim), int(output_dim)),
		)

	"""执行夹爪预测分支的项目特化初始化。"""
	def init_parameters(self) -> None:
		first_linear = self.network[0]
		last_linear = self.network[-1]
		nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
		nn.init.zeros_(first_linear.bias)
		nn.init.zeros_(last_linear.weight)
		nn.init.zeros_(last_linear.bias)

	def forward(self, position_hidden: torch.Tensor, trajectory_mask: torch.Tensor) -> torch.Tensor:
		gripper_logit = self.network(position_hidden)
		return apply_sequence_mask(gripper_logit, trajectory_mask)


"""AtomAction_NSVQ 的 Flow Matching Head。

输入：
	__init__:
		config: Dict[str, Any]，来自 model.yaml 的 flow_matching_head 配置。
		rope_theta: float，RoPE 频率基数。
		rope_max_seq_len: int，RoPE 最大长度。
	forward:
		noisy_ee_trajectory: [B, T, 9]
		timestep: [B] 或 [B, 1]
		global_codeword: [B, 256]
		detail_codewords: [B, 9, 256]
		trajectory_mask: [B, T]

输出：
	forward:
		position_vector_field: [B, T, 3]
		rotation_vector_field: [B, T, 6]
		gripper_logit: [B, T, 1]
"""
class FlowMatchingHead(nn.Module):

	def __init__(self, config: Dict[str, Any], rope_theta: float, rope_max_seq_len: int) -> None:
		super().__init__()
		self.input_dim = int(config["input_dim"])
		self.hidden_dim = int(config["hidden_dim"])
		self.time_embed_dim = int(config["time_embed_dim"])
		self.global_code_dim = int(config["global_code_dim"])
		self.detail_code_dim = int(config["detail_code_dim"])
		self.num_detail_tokens = int(config["num_detail_tokens"])
		self.condition_hidden_dim = int(config["condition_hidden_dim"])
		self.dropout = float(config["dropout"])

		position_branch_cfg = config["position_branch"]
		rotation_branch_cfg = config["rotation_branch"]
		gripper_branch_cfg = config["gripper_branch"]

		self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
		self.timestep_embedding = SinusoidalTimestepEmbedding(
			embedding_dim=self.time_embed_dim,
			min_period=float(config["time_min_period"]),
			max_period=float(config["time_max_period"]),
		)
		self.condition_fusion = nn.Sequential(
			nn.Linear(self.global_code_dim + self.time_embed_dim, self.condition_hidden_dim),
			nn.GELU(),
			nn.Linear(self.condition_hidden_dim, self.hidden_dim),
		)
		self.detail_projection = nn.Linear(self.detail_code_dim, self.hidden_dim)

		self.decoder = FlowMatchingDecoder(
			hidden_dim=self.hidden_dim,
			num_layers=int(config["num_layers"]),
			num_heads=int(config["num_heads"]),
			ffn_dim=int(config["ffn_dim"]),
			condition_dim=self.hidden_dim,
			detail_injection_layers=config["detail_injection_layers"],
			dropout=self.dropout,
			rope_theta=float(rope_theta),
			rope_max_seq_len=int(rope_max_seq_len),
		)
		self.position_branch = FlowMatchingPredictionBranch(
			hidden_dim=self.hidden_dim,
			num_layers=int(position_branch_cfg["num_layers"]),
			num_heads=int(config["num_heads"]),
			ffn_dim=int(position_branch_cfg["ffn_dim"]),
			dropout=self.dropout,
			rope_theta=float(rope_theta),
			rope_max_seq_len=int(rope_max_seq_len),
			output_dim=int(position_branch_cfg["output_dim"]),
		)
		self.rotation_branch = FlowMatchingPredictionBranch(
			hidden_dim=self.hidden_dim,
			num_layers=int(rotation_branch_cfg["num_layers"]),
			num_heads=int(config["num_heads"]),
			ffn_dim=int(rotation_branch_cfg["ffn_dim"]),
			dropout=self.dropout,
			rope_theta=float(rope_theta),
			rope_max_seq_len=int(rope_max_seq_len),
			output_dim=int(rotation_branch_cfg["output_dim"]),
		)
		self.gripper_branch = GripperPredictionBranch(
			input_dim=self.hidden_dim,
			hidden_dim=int(gripper_branch_cfg["hidden_dim"]),
			output_dim=int(gripper_branch_cfg["output_dim"]),
			dropout=self.dropout,
		)

	"""执行 Flow Matching 条件融合层的项目特化初始化。"""
	def init_parameters(self) -> None:
		first_linear = self.condition_fusion[0]
		nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
		nn.init.zeros_(first_linear.bias)

	def forward(
		self,
		noisy_ee_trajectory: torch.Tensor,
		timestep: torch.Tensor,
		global_codeword: torch.Tensor,
		detail_codewords: torch.Tensor,
		trajectory_mask: torch.Tensor,
	) -> Dict[str, torch.Tensor]:
		if noisy_ee_trajectory.ndim != 3 or noisy_ee_trajectory.shape[-1] != self.input_dim:
			raise ValueError("noisy_ee_trajectory must have shape [B, T, input_dim]")
		if trajectory_mask.ndim != 2 or noisy_ee_trajectory.shape[:2] != trajectory_mask.shape:
			raise ValueError("trajectory_mask must have shape [B, T] matching noisy_ee_trajectory")

		decoder_input = self.input_projection(noisy_ee_trajectory)
		decoder_input = apply_sequence_mask(decoder_input, trajectory_mask)

		time_embedding = self.timestep_embedding(timestep).to(device=global_codeword.device, dtype=global_codeword.dtype)
		condition = torch.cat((global_codeword, time_embedding), dim=-1)
		condition = self.condition_fusion(condition)

		detail_context = self.detail_projection(detail_codewords)
		decoder_hidden = self.decoder(decoder_input, condition, detail_context, trajectory_mask)
		decoder_hidden = apply_sequence_mask(decoder_hidden, trajectory_mask)

		position_hidden, position_vector_field = self.position_branch(decoder_hidden, trajectory_mask)
		_, rotation_vector_field = self.rotation_branch(decoder_hidden, trajectory_mask)
		gripper_logit = self.gripper_branch(position_hidden, trajectory_mask)
		ee_vector_field = torch.cat((position_vector_field, rotation_vector_field), dim=-1)
		ee_vector_field = apply_sequence_mask(ee_vector_field, trajectory_mask)

		return {
			"position_vector_field": position_vector_field,
			"rotation_vector_field": rotation_vector_field,
			"gripper_logit": gripper_logit,
		}


"""VASA 的视觉序列条件 Flow Matching Head。

输入：
	__init__:
		config: Dict[str, Any]，来自 model.yaml 的 VASA.flow_matching_head 配置。
	forward:
		noisy_ee_trajectory: [B, T_action, 9]
		timestep: [B] 或 [B, 1]
		visual_condition: [B, T_vis, C_vis]，对应 img_diff_features。
		trajectory_mask: [B, T_action]
		visual_condition_mask: [B, T_vis]，可选；None 时默认所有视觉 token 有效。

输出：
	forward:
		position_vector_field: [B, T_action, 3]
		rotation_vector_field: [B, T_action, 6]
		gripper_logit: [B, T_action, 1]
"""
class VisualFlowMatchingHead(nn.Module):

	def __init__(self, config: Dict[str, Any]) -> None:
		super().__init__()
		self.input_dim = int(config["input_dim"])
		self.hidden_dim = int(config["hidden_dim"])
		self.time_embed_dim = int(config["time_embed_dim"])
		self.visual_condition_dim = int(config["visual_condition_dim"])
		self.condition_hidden_dim = int(config["condition_hidden_dim"])
		self.num_layers = int(config["num_layers"])
		self.dropout = float(config["dropout"])

		position_branch_cfg = config["position_branch"]
		rotation_branch_cfg = config["rotation_branch"]
		gripper_branch_cfg = config["gripper_branch"]

		condition_injection_layers = config.get("condition_injection_layers")
		if condition_injection_layers is None:
			condition_injection_layers = list(range(1, self.num_layers + 1))

		self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
		self.timestep_embedding = SinusoidalTimestepEmbedding(
			embedding_dim=self.time_embed_dim,
			min_period=float(config["time_min_period"]),
			max_period=float(config["time_max_period"]),
		)
		self.condition_fusion = nn.Sequential(
			nn.Linear(self.time_embed_dim, self.condition_hidden_dim),
			nn.GELU(),
			nn.Linear(self.condition_hidden_dim, self.hidden_dim),
		)
		self.visual_condition_projection = nn.Linear(self.visual_condition_dim, self.hidden_dim)

		self.decoder = VisualConditionedFlowMatchingDecoder(
			hidden_dim=self.hidden_dim,
			num_layers=self.num_layers,
			num_heads=int(config["num_heads"]),
			ffn_dim=int(config["ffn_dim"]),
			condition_dim=self.hidden_dim,
			condition_injection_layers=condition_injection_layers,
			dropout=self.dropout,
			rope_theta=float(config["rope_theta"]),
			rope_max_seq_len=int(config["rope_max_seq_len"]),
		)
		self.position_branch = FlowMatchingPredictionBranch(
			hidden_dim=self.hidden_dim,
			num_layers=int(position_branch_cfg["num_layers"]),
			num_heads=int(config["num_heads"]),
			ffn_dim=int(position_branch_cfg["ffn_dim"]),
			dropout=self.dropout,
			rope_theta=float(config["rope_theta"]),
			rope_max_seq_len=int(config["rope_max_seq_len"]),
			output_dim=int(position_branch_cfg["output_dim"]),
		)
		self.rotation_branch = FlowMatchingPredictionBranch(
			hidden_dim=self.hidden_dim,
			num_layers=int(rotation_branch_cfg["num_layers"]),
			num_heads=int(config["num_heads"]),
			ffn_dim=int(rotation_branch_cfg["ffn_dim"]),
			dropout=self.dropout,
			rope_theta=float(config["rope_theta"]),
			rope_max_seq_len=int(config["rope_max_seq_len"]),
			output_dim=int(rotation_branch_cfg["output_dim"]),
		)
		self.gripper_branch = GripperPredictionBranch(
			input_dim=self.hidden_dim,
			hidden_dim=int(gripper_branch_cfg["hidden_dim"]),
			output_dim=int(gripper_branch_cfg["output_dim"]),
			dropout=self.dropout,
		)

	"""执行视觉 Flow Matching 条件融合层的项目特化初始化。"""
	def init_parameters(self) -> None:
		first_linear = self.condition_fusion[0]
		nn.init.kaiming_normal_(first_linear.weight, mode="fan_in", nonlinearity="relu")
		nn.init.zeros_(first_linear.bias)

	def forward(
		self,
		noisy_ee_trajectory: torch.Tensor,
		timestep: torch.Tensor,
		visual_condition: torch.Tensor,
		trajectory_mask: torch.Tensor,
		visual_condition_mask: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		if noisy_ee_trajectory.ndim != 3 or noisy_ee_trajectory.shape[-1] != self.input_dim:
			raise ValueError("noisy_ee_trajectory must have shape [B, T_action, input_dim]")
		if trajectory_mask.ndim != 2 or noisy_ee_trajectory.shape[:2] != trajectory_mask.shape:
			raise ValueError("trajectory_mask must have shape [B, T_action] matching noisy_ee_trajectory")
		if visual_condition.ndim != 3 or visual_condition.shape[-1] != self.visual_condition_dim:
			raise ValueError("visual_condition must have shape [B, T_vis, visual_condition_dim]")
		if visual_condition.shape[0] != noisy_ee_trajectory.shape[0]:
			raise ValueError("visual_condition and noisy_ee_trajectory must share batch size")

		if visual_condition_mask is None:
			visual_condition_mask = torch.ones(
				visual_condition.shape[0],
				visual_condition.shape[1],
				dtype=torch.bool,
				device=visual_condition.device,
			)
		elif visual_condition_mask.ndim != 2 or visual_condition_mask.shape != visual_condition.shape[:2]:
			raise ValueError("visual_condition_mask must have shape [B, T_vis] matching visual_condition")

		# 加噪末端轨迹输入投影：[B, T_action, 9] -> [B, T_action, C]
		decoder_input = self.input_projection(noisy_ee_trajectory)
		trajectory_mask = trajectory_mask.to(device=decoder_input.device)
		visual_condition_mask = visual_condition_mask.to(device=decoder_input.device)
		decoder_input = apply_sequence_mask(decoder_input, trajectory_mask)

		# 流时间条件：[B] / [B, 1] -> [B, time_embed_dim] -> [B, C]
		time_embedding = self.timestep_embedding(timestep).to(device=decoder_input.device, dtype=decoder_input.dtype)
		condition = self.condition_fusion(time_embedding)

		# 视觉差分序列条件：[B, T_vis, C_vis] -> [B, T_vis, C]
		visual_condition = visual_condition.to(device=decoder_input.device, dtype=decoder_input.dtype)
		visual_context = self.visual_condition_projection(visual_condition)
		visual_context = apply_sequence_mask(visual_context, visual_condition_mask)

		# 视觉条件主干：[B, T_action, C] x [B, T_vis, C] -> [B, T_action, C]
		decoder_hidden = self.decoder(
			x=decoder_input,
			condition=condition,
			visual_context=visual_context,
			trajectory_mask=trajectory_mask,
			visual_context_mask=visual_condition_mask,
		)
		decoder_hidden = apply_sequence_mask(decoder_hidden, trajectory_mask)

		# 三个输出分支：
		# position: [B, T_action, C] -> [B, T_action, 3]
		# rotation: [B, T_action, C] -> [B, T_action, 6]
		# gripper:  [B, T_action, C] -> [B, T_action, 1]
		position_hidden, position_vector_field = self.position_branch(decoder_hidden, trajectory_mask)
		_, rotation_vector_field = self.rotation_branch(decoder_hidden, trajectory_mask)
		gripper_logit = self.gripper_branch(position_hidden, trajectory_mask)

		return {
			"position_vector_field": position_vector_field,
			"rotation_vector_field": rotation_vector_field,
			"gripper_logit": gripper_logit,
		}
