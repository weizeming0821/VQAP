from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .module.atomaction_nsvq import AtomAction_NSVQ
from .module.model_init import apply_vqap_initialization
from .module.utils import apply_sequence_mask
from .module.vasa import VASA


"""VQAP 顶层模型。

输入：
	__init__:
		model_args: Dict[str, Any]，来自 model.yaml 的模型配置。
		train_args: Dict[str, Any]，来自 train.yaml 的训练配置。
	forward:
		trajectory_data: Dict[str, torch.Tensor]
		trajectory_mask: [B, T]，True 表示有效动作帧。
		selected_views: 长度为 B 的列表，第 b 个元素是长度为 K 的视角结果列表。

输出：
	forward:
		shared_outputs:
			clean_ee_trajectory: [B, T, 9]
			noise: [B, T, 9]
			noisy_ee_trajectory: [B, T, 9]
			timestep: [B]
			trajectory_mask: [B, T]
			target_position_vector_field: [B, T, 3]
			target_rotation_vector_field: [B, T, 6]
			target_gripper_state: [B, T, 1]
		atomaction_outputs:
			AtomAction_NSVQ.forward 的输出字典。
		vasa_outputs:
			VASA.forward 的输出字典。
"""
class VQAP(nn.Module):

	def __init__(self, model_args: Dict[str, Any], train_args: Dict[str, Any]):
		super().__init__()
		self.model_args = model_args
		self.train_args = train_args
		self.noise_std = float(self.train_args["flow_matching"]["noise_std"])

		self.atomaction_nsvq = AtomAction_NSVQ(model_args=self.model_args)
		self.vasa = VASA(model_args=self.model_args)
		self._init_model_parameters()

	"""执行 VQAP 全模型统一初始化。"""
	def _init_model_parameters(self) -> None:
		apply_vqap_initialization(self)

	"""构造 AtomAction 与 VASA 共用的 Flow Matching 训练目标。

	维度变化：
		clean_ee_trajectory: [B, T, 9]
		timestep: [B]
		noise: [B, T, 9]
		noisy_ee_trajectory: [B, T, 9]
		target_vector_field: [B, T, 9] -> pos [B, T, 3] + rot [B, T, 6]
	"""
	def _build_flow_matching_targets(
		self,
		trajectory_data: Dict[str, torch.Tensor],
		trajectory_mask: torch.Tensor,
	) -> Dict[str, torch.Tensor]:
		clean_ee_trajectory = trajectory_data["gripper_pose"]
		target_gripper_state = trajectory_data["gripper_open"]

		trajectory_mask = trajectory_mask.to(device=clean_ee_trajectory.device)
		valid_mask = trajectory_mask.unsqueeze(-1).to(clean_ee_trajectory.dtype)	# [B, T] -> [B, T, 1]
		clean_ee_trajectory = clean_ee_trajectory * valid_mask	# [B, T, 9] -> [B, T, 9]
		target_gripper_state = target_gripper_state.to(dtype=clean_ee_trajectory.dtype) * valid_mask	# [B, T, 1] -> [B, T, 1]

		batch_size = clean_ee_trajectory.shape[0]
		timestep = torch.rand(
			batch_size,
			device=clean_ee_trajectory.device,
			dtype=clean_ee_trajectory.dtype,
		)	# [B]
		noise = torch.randn_like(clean_ee_trajectory) * self.noise_std	# [B, T, 9]
		noise = noise * valid_mask	# [B, T, 9] -> [B, T, 9]

		timestep_weights = timestep.view(batch_size, 1, 1)	# [B] -> [B, 1, 1]
		noisy_ee_trajectory = (
			timestep_weights * clean_ee_trajectory + (1.0 - timestep_weights) * noise
		)	# [B, T, 9]
		noisy_ee_trajectory = apply_sequence_mask(noisy_ee_trajectory, trajectory_mask)	# [B, T, 9] -> [B, T, 9]

		target_vector_field = clean_ee_trajectory - noise	# [B, T, 9] -> [B, T, 9]
		target_vector_field = apply_sequence_mask(target_vector_field, trajectory_mask)	# [B, T, 9] -> [B, T, 9]

		return {
			"clean_ee_trajectory": clean_ee_trajectory,
			"noise": noise,
			"noisy_ee_trajectory": noisy_ee_trajectory,
			"timestep": timestep,
			"trajectory_mask": trajectory_mask,
			"target_position_vector_field": target_vector_field[..., :3],	# [B, T, 9] -> [B, T, 3]
			"target_rotation_vector_field": target_vector_field[..., 3:],	# [B, T, 9] -> [B, T, 6]
			"target_gripper_state": target_gripper_state,
		}

	def forward(
		self,
		trajectory_data: Dict[str, torch.Tensor],
		trajectory_mask: torch.Tensor,
		selected_views: Sequence[Sequence[Dict[str, Any]]],
	) -> Dict[str, Dict[str, torch.Tensor]]:
		shared_outputs = self._build_flow_matching_targets(
			trajectory_data=trajectory_data,
			trajectory_mask=trajectory_mask,
		)

		atomaction_outputs = self.atomaction_nsvq(
			trajectory_data=trajectory_data,
			trajectory_mask=shared_outputs["trajectory_mask"],
			noisy_ee_trajectory=shared_outputs["noisy_ee_trajectory"],
			timestep=shared_outputs["timestep"],
		)

		vasa_outputs = self.vasa(
			selected_views=selected_views,
			noisy_ee_trajectory=shared_outputs["noisy_ee_trajectory"],
			timestep=shared_outputs["timestep"],
			trajectory_mask=shared_outputs["trajectory_mask"],
			global_codeword=atomaction_outputs["global_codeword"],
		)

		return {
			"shared_outputs": shared_outputs,
			"atomaction_outputs": atomaction_outputs,
			"vasa_outputs": vasa_outputs,
		}

	"""计算 VQAP 总参数量与可训练参数量。"""
	def print_module_params(self) -> Dict[str, int]:
		total_params = sum(param.numel() for param in self.parameters())
		trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
		print(f"VQAP total params: {total_params}")
		print(f"VQAP trainable params: {trainable_params}")
		return {
			"total": total_params,
			"trainable": trainable_params,
		}

	"""显式触发 AtomAction 双码本的死码替换。"""
	@torch.no_grad()
	def replace_unused_codebooks(self, used_steps: int) -> Dict[str, torch.Tensor]:
		return self.atomaction_nsvq.replace_unused_codebooks(used_steps=used_steps)
