from typing import Any, Dict

import torch
import torch.nn.functional as F


"""将 [B, C] 或 [B, P, C] 的图像特征统一整理为 patch 序列形式。

维度变化：
	[B, C] -> [B, 1, C]
	[B, P, C] -> [B, P, C]
"""
def _to_patch_sequence(image_features: torch.Tensor, feature_name: str) -> torch.Tensor:
	if image_features.ndim == 2:
		return image_features.unsqueeze(1)
	if image_features.ndim == 3:
		return image_features
	raise ValueError(f"{feature_name} must have shape [B, C] or [B, P, C]")


"""计算按有效时间步聚合的向量场 MSE。

输入：
	prediction: [B, T, D]
	target: [B, T, D]
	trajectory_mask: [B, T]

输出：
	标量损失。
"""
def masked_mse_loss(
	prediction: torch.Tensor,
	target: torch.Tensor,
	trajectory_mask: torch.Tensor,
) -> torch.Tensor:
	if prediction.shape != target.shape:
		raise ValueError("prediction and target must share the same shape")
	if prediction.ndim != 3:
		raise ValueError("prediction and target must have shape [B, T, D]")
	if trajectory_mask.ndim != 2 or prediction.shape[:2] != trajectory_mask.shape:
		raise ValueError("trajectory_mask must have shape [B, T] matching prediction")

	squared_error = (prediction - target).pow(2).sum(dim=-1)	# [B, T, D] -> [B, T]
	valid_mask = trajectory_mask.to(dtype=prediction.dtype)	# [B, T]
	return (squared_error * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


"""计算按有效时间步聚合的 BCE with logits。

输入：
	logits: [B, T, 1]
	target: [B, T, 1]
	trajectory_mask: [B, T]

输出：
	标量损失。
"""
def masked_bce_with_logits_loss(
	logits: torch.Tensor,
	target: torch.Tensor,
	trajectory_mask: torch.Tensor,
) -> torch.Tensor:
	if logits.shape != target.shape:
		raise ValueError("logits and target must share the same shape")
	if logits.ndim != 3 or logits.shape[-1] != 1:
		raise ValueError("logits and target must have shape [B, T, 1]")
	if trajectory_mask.ndim != 2 or logits.shape[:2] != trajectory_mask.shape:
		raise ValueError("trajectory_mask must have shape [B, T] matching logits")

	bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none").squeeze(-1)	# [B, T, 1] -> [B, T]
	valid_mask = trajectory_mask.to(dtype=logits.dtype)	# [B, T]
	return (bce_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


"""计算全局支路与细节支路的表示分离损失。

输入：
	pooled_global_features: [B, C]
	detail_query_features: [B, N_detail, C]

输出：
	标量损失。
"""
def compute_separation_loss(
	pooled_global_features: torch.Tensor,
	detail_query_features: torch.Tensor,
) -> torch.Tensor:
	if pooled_global_features.ndim != 2:
		raise ValueError("pooled_global_features must have shape [B, C]")
	if detail_query_features.ndim != 3:
		raise ValueError("detail_query_features must have shape [B, N_detail, C]")
	if pooled_global_features.shape[0] != detail_query_features.shape[0]:
		raise ValueError("pooled_global_features and detail_query_features must share batch size")
	if pooled_global_features.shape[-1] != detail_query_features.shape[-1]:
		raise ValueError("pooled_global_features and detail_query_features must share feature dim")

	mean_detail_features = detail_query_features.mean(dim=1)	# [B, N_detail, C] -> [B, C]
	return F.cosine_similarity(pooled_global_features, mean_detail_features, dim=-1).abs().mean()


"""计算未来帧 patch 逐位置 cosine 对齐损失。

输入：
	pred_end_patch_features: [B, 1, C] 或 [B, P, C]
	end_img_features: [B, C] 或 [B, P, C]

输出：
	标量损失。
"""
def compute_future_patch_loss(
	pred_end_patch_features: torch.Tensor,
	end_img_features: torch.Tensor,
) -> torch.Tensor:
	pred_patch_features = _to_patch_sequence(
		pred_end_patch_features,
		feature_name="pred_end_patch_features",
	)	# [B, C]/[B, P, C] -> [B, P, C]
	target_patch_features = _to_patch_sequence(
		end_img_features,
		feature_name="end_img_features",
	).detach()	# [B, C]/[B, P, C] -> [B, P, C]

	if pred_patch_features.shape != target_patch_features.shape:
		raise ValueError("pred_end_patch_features and end_img_features must share the same patch shape")

	patch_cosine = F.cosine_similarity(pred_patch_features, target_patch_features, dim=-1)	# [B, P, C] -> [B, P]
	return (1.0 - patch_cosine).mean()


"""按 epoch 计算未来帧分支权重调度。

	Stage 0: 线性从 0 增长到 future_stage0_max
	Stage 1 前 future_stage1_ramp_epochs 个 epoch: 线性从 future_stage0_max 增长到 future_stage1_max
	Stage 1 后续 epoch: 固定为 future_stage1_max
"""
def compute_future_weight_schedule(
	loss_cfg: Dict[str, Any],
	stage_cfg: Dict[str, Any],
	epoch: int,
) -> float:
	stage0_epochs = int(stage_cfg["stage0_epochs"])
	stage1_ramp_epochs = int(loss_cfg["future_stage1_ramp_epochs"])
	future_stage0_max = float(loss_cfg["future_stage0_max"])
	future_stage1_max = float(loss_cfg["future_stage1_max"])
	epoch_index = int(epoch) + 1

	if stage0_epochs <= 0:
		raise ValueError("stage0_epochs must be positive")
	if epoch_index <= stage0_epochs:
		return future_stage0_max * float(epoch_index) / float(stage0_epochs)

	stage1_epoch = epoch_index - stage0_epochs
	if stage1_ramp_epochs <= 0:
		return future_stage1_max
	if stage1_epoch <= stage1_ramp_epochs:
		ramp_ratio = float(stage1_epoch) / float(stage1_ramp_epochs)
		return future_stage0_max + (future_stage1_max - future_stage0_max) * ramp_ratio
	return future_stage1_max


"""计算 AtomAction-NSVQ 动作重构损失 L_AP。"""
def compute_atomaction_reconstruction_loss(
	shared_outputs: Dict[str, torch.Tensor],
	atomaction_outputs: Dict[str, torch.Tensor],
	loss_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
	loss_fm_pos_ap = masked_mse_loss(
		prediction=atomaction_outputs["position_vector_field"],
		target=shared_outputs["target_position_vector_field"],
		trajectory_mask=shared_outputs["trajectory_mask"],
	)
	loss_fm_rot_ap = masked_mse_loss(
		prediction=atomaction_outputs["rotation_vector_field"],
		target=shared_outputs["target_rotation_vector_field"],
		trajectory_mask=shared_outputs["trajectory_mask"],
	)
	loss_bce_grip_ap = masked_bce_with_logits_loss(
		logits=atomaction_outputs["gripper_logit"],
		target=shared_outputs["target_gripper_state"],
		trajectory_mask=shared_outputs["trajectory_mask"],
	)
	loss_sep = compute_separation_loss(
		pooled_global_features=atomaction_outputs["pooled_global_features"],
		detail_query_features=atomaction_outputs["detail_query_features"],
	)

	lambda_rot = float(loss_cfg["lambda_rot"])
	lambda_grip = float(loss_cfg["lambda_grip"])
	lambda_sep = float(loss_cfg["lambda_sep"])
	loss_ap = loss_fm_pos_ap + lambda_rot * loss_fm_rot_ap + lambda_grip * loss_bce_grip_ap + lambda_sep * loss_sep

	return {
		"loss_fm_pos_ap": loss_fm_pos_ap,
		"loss_fm_rot_ap": loss_fm_rot_ap,
		"loss_bce_grip_ap": loss_bce_grip_ap,
		"loss_sep": loss_sep,
		"loss_ap": loss_ap,
	}


"""计算视觉-动作对齐损失 L_AG。"""
def compute_action_grounding_loss(
	shared_outputs: Dict[str, torch.Tensor],
	vasa_outputs: Dict[str, torch.Tensor],
	loss_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
	loss_fm_pos_ag = masked_mse_loss(
		prediction=vasa_outputs["position_vector_field"],
		target=shared_outputs["target_position_vector_field"],
		trajectory_mask=shared_outputs["trajectory_mask"],
	)
	loss_fm_rot_ag = masked_mse_loss(
		prediction=vasa_outputs["rotation_vector_field"],
		target=shared_outputs["target_rotation_vector_field"],
		trajectory_mask=shared_outputs["trajectory_mask"],
	)
	loss_bce_grip_ag = masked_bce_with_logits_loss(
		logits=vasa_outputs["gripper_logit"],
		target=shared_outputs["target_gripper_state"],
		trajectory_mask=shared_outputs["trajectory_mask"],
	)

	lambda_rot = float(loss_cfg["lambda_rot"])
	lambda_grip = float(loss_cfg["lambda_grip"])
	loss_ag = loss_fm_pos_ag + lambda_rot * loss_fm_rot_ag + lambda_grip * loss_bce_grip_ag

	return {
		"loss_fm_pos_ag": loss_fm_pos_ag,
		"loss_fm_rot_ag": loss_fm_rot_ag,
		"loss_bce_grip_ag": loss_bce_grip_ag,
		"loss_ag": loss_ag,
	}
