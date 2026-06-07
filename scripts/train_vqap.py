import argparse
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from data.dataset import AtomActionDataset
from data.utils import AtomActionDataset_collate_fn
from model.vqap import VQAP
from utils.init_logger_wandb import finish_wandb, init_logger, init_wandb
from utils.loss_func import (
	compute_action_grounding_loss,
	compute_atomaction_reconstruction_loss,
	compute_future_patch_loss,
	compute_future_weight_schedule,
)


EPOCH_METRIC_KEYS = (
	"loss_total",
	"loss_ap",
	"loss_fm_pos_ap",
	"loss_fm_rot_ap",
	"loss_bce_grip_ap",
	"loss_sep",
	"loss_ag",
	"loss_fm_pos_ag",
	"loss_fm_rot_ag",
	"loss_bce_grip_ag",
	"loss_future",
	"perplexity_g",
	"perplexity_d",
)


"""读取 YAML 配置文件，并校验顶层必须为字典。"""
def load_yaml_config(path: str) -> Dict[str, Any]:
	config_path = Path(path).expanduser()
	with config_path.open("r", encoding="utf-8") as file:
		config = yaml.safe_load(file)

	if config is None:
		return {}
	if not isinstance(config, dict):
		raise ValueError(f"Config file must contain a top-level mapping: {config_path}")
	return config


"""设置随机种子。DDP 下用 seed + rank 让每个进程有独立但可复现的随机序列。"""
def set_random_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


"""按阶段构造线性 warmup + cosine decay 的学习率比例函数。"""
def make_lr_lambda(warmup_epochs: int, total_epochs: int, min_ratio: float) -> Any:
	def lr_lambda(epoch: int) -> float:
		if epoch < warmup_epochs:
			return float(epoch + 1) / float(max(1, warmup_epochs))

		progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
		cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
		return float(min_ratio) + (1.0 - float(min_ratio)) * cosine_ratio

	return lr_lambda


"""把 batch 中需要上卡的张量移到当前 device，selected_views 保持 Python 列表结构不变。"""
def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
	trajectory_data = {
		field_name: field_tensor.to(device=device, non_blocking=True)
		for field_name, field_tensor in batch["trajectory_data"].items()
	}
	return {
		"Action": batch["Action"],
		"Task": batch["Task"],
		"Variation": batch["Variation"],
		"trajectory_data": trajectory_data,
		"trajectory_length": batch["trajectory_length"].to(device=device, non_blocking=True),
		"trajectory_mask": batch["trajectory_mask"].to(device=device, non_blocking=True),
		"selected_views": batch["selected_views"],
	}


class VQAPTrainer:

	def __init__(self, args: argparse.Namespace):

		# 训练配置参数读取
		self.args = args
		self.model_args = load_yaml_config(args.model_config)
		self.train_args = load_yaml_config(args.config)
		self.global_args = load_yaml_config(args.global_config)

		experiment_cfg = self.train_args["experiment"]
		self.exp_name = args.exp_name or str(experiment_cfg.get("exp_name", "vqap_pretrain"))
		self.rank = 0
		self.local_rank = 0
		self.world_size = 1
		self.distributed = False
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# 分布式环境初始化
		self._init_distributed()

		# 随机种子设置
		self.seed = int(experiment_cfg.get("seed", 42))
		set_random_seed(self.seed + self.rank)

		# CUDA 性能优化设置
		runtime_cfg = self.train_args["runtime"]
		if self.device.type == "cuda":
			torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
			torch.backends.cuda.matmul.allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))
			os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

		# 混合精度设置
		self.precision = str(runtime_cfg.get("precision", "bf16")).lower()
		self.autocast_dtype: Optional[torch.dtype]
		if self.device.type != "cuda" or self.precision == "fp32":
			self.autocast_dtype = None
		elif self.precision == "bf16":
			self.autocast_dtype = torch.bfloat16
		elif self.precision == "fp16":
			self.autocast_dtype = torch.float16
		else:
			raise ValueError(f"Unsupported precision setting: {self.precision}")
		self.grad_scaler = torch.amp.GradScaler(
			device="cuda",
			enabled=self.device.type == "cuda" and self.precision == "fp16",
		)

		# 日志与 checkpoint 目录初始化
		checkpoint_cfg = self.train_args["checkpoint"]
		logging_cfg = self.train_args["logging"]
		self.ckpt_dir = Path(checkpoint_cfg["root_dir"]).expanduser() / self.exp_name
		self.ckpt_dir.mkdir(parents=True, exist_ok=True)
		self.logger = init_logger(
			rank=self.rank,
			exp_name=self.exp_name,
			log_dir=str(logging_cfg.get("log_dir", "log")),
			is_resume=bool(args.resume),
		)

		# 训练阶段与优化器调度设置
		self.stage0_epochs = int(self.train_args["stage"]["stage0_epochs"])
		self.stage1_epochs = int(self.train_args["stage"]["stage1_epochs"])
		self.total_epochs = self.stage0_epochs + self.stage1_epochs
		self.save_every_epochs = int(checkpoint_cfg["save_every_epochs"])
		self.grad_clip_norm = float(runtime_cfg["grad_clip_norm"])

		# 断点续训设置
		self.resume_path = self._resolve_resume_path()
		self.resume_state = self._load_resume_state()
		self.current_stage = int(self.resume_state["stage"]) if self.resume_state is not None else 0
		self.start_epoch = int(self.resume_state["epoch"]) if self.resume_state is not None else 0
		self.global_step = int(self.resume_state.get("global_step", 0)) if self.resume_state is not None else 0
		self.best_lap = float(self.resume_state.get("best_lap", float("inf"))) if self.resume_state is not None else float("inf")
		self.best_ltotal = float(self.resume_state.get("best_ltotal", float("inf"))) if self.resume_state is not None else float("inf")

		# wandb 的启停只影响实验记录，不影响训练主逻辑。
		wandb_cfg = dict(self.train_args)
		if args.disable_wandb:
			wandb_cfg["wandb"] = dict(wandb_cfg.get("wandb", {}))
			wandb_cfg["wandb"]["enable"] = False
		self.wandb_run = init_wandb(
			rank=self.rank,
			exp_name=self.exp_name,
			cfg=wandb_cfg,
			ckpt_dir=str(self.ckpt_dir),
			is_resume=self.resume_state is not None,
		)

		# 模型、数据、优化器的初始化
		self.train_dataset, self.train_sampler, self.train_loader = self._init_dataset()
		self.model = self._init_model(apply_lora=self.current_stage == 0)
		if self.resume_state is not None:
			self.model.load_state_dict(self.resume_state["model"], strict=True)
		if self.current_stage == 1:
			# Stage 1 续训时不会再恢复 LoRA 结构，而是直接基于 merge 后的主干继续训练。
			self._freeze_stage1_modules(self.model)
		self._wrap_model_for_ddp()

		self.optimizer = self._init_optimizer(stage=self.current_stage)
		self.scheduler = self._init_scheduler(stage=self.current_stage)
		if self.resume_state is not None:
			self.optimizer.load_state_dict(self.resume_state["optimizer"])
			self.scheduler.load_state_dict(self.resume_state["scheduler"])

		if self.rank == 0:
			self.logger.info(
				f"Trainer initialized | stage={self.current_stage} | start_epoch={self.start_epoch} | "
				f"global_step={self.global_step} | dataset_size={len(self.train_dataset)}"
			)

	"""初始化单机/多卡训练环境，并设置当前进程使用的 GPU。"""
	def _init_distributed(self) -> None:
		self.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
		if not self.distributed:
			if self.device.type == "cuda":
				self.local_rank = 0
				torch.cuda.set_device(self.local_rank)
			return

		if self.device.type != "cuda":
			raise RuntimeError("Distributed training requires CUDA devices")

		self.local_rank = int(os.environ["LOCAL_RANK"])
		self.rank = int(os.environ["RANK"])
		self.world_size = int(os.environ["WORLD_SIZE"])
		torch.cuda.set_device(self.local_rank)
		dist.init_process_group(backend="nccl", init_method="env://")

	"""解析断点续训路径：显式传参优先，其次读取 train.yaml，最后回退到 latest.pth。"""
	def _resolve_resume_path(self) -> Optional[Path]:
		resume_cfg = self.train_args.get("resume", {})
		config_resume_path = resume_cfg.get("path")
		resume_path = self.args.resume_path or config_resume_path
		if resume_path is None:
			if self.args.resume:
				return self.ckpt_dir / "latest.pth"
			return None
		return Path(resume_path).expanduser()

	"""按需加载 checkpoint 状态，但不在这里做 model/optimizer 的恢复。"""
	def _load_resume_state(self) -> Optional[Dict[str, Any]]:
		if self.resume_path is None:
			return None
		if not self.resume_path.is_file():
			raise FileNotFoundError(f"Resume checkpoint not found: {self.resume_path}")
		if self.rank == 0:
			self.logger.info(f"Loading checkpoint from {self.resume_path}")
		return torch.load(self.resume_path, map_location="cpu")

	"""构建唯一训练集与 DataLoader，数据集超参全部来自 global.yaml。"""
	def _init_dataset(self) -> tuple[AtomActionDataset, Optional[DistributedSampler], DataLoader]:
		dataset_cfg = self.global_args["atomactiondataset"]
		top_k = int(dataset_cfg.get("top_k", dataset_cfg.get("tok_k", 1)))
		views = dataset_cfg.get("views")
		view_selector_kwargs = dict(dataset_cfg.get("view_selector", {}))
		dataset = AtomActionDataset(
			dataset_root=str(dataset_cfg["dataset_root"]),
			views=views,
			top_k=top_k,
			view_selector_kwargs=view_selector_kwargs,
		)

		data_cfg = self.train_args["data"]
		sampler: Optional[DistributedSampler]
		if self.distributed:
			sampler = DistributedSampler(dataset, shuffle=True, drop_last=bool(data_cfg.get("drop_last", False)))
		else:
			sampler = None

		num_workers = int(data_cfg["num_workers"])
		persistent_workers = bool(data_cfg.get("persistent_workers", False)) and num_workers > 0
		dataloader_kwargs: Dict[str, Any] = {
			"batch_size": int(data_cfg["batch_size"]),
			"shuffle": sampler is None,
			"sampler": sampler,
			"drop_last": bool(data_cfg.get("drop_last", False)),
			"num_workers": num_workers,
			"pin_memory": bool(data_cfg.get("pin_memory", True)),
			"persistent_workers": persistent_workers,
			"collate_fn": AtomActionDataset_collate_fn,
		}
		if num_workers > 0 and "prefetch_factor" in data_cfg:
			dataloader_kwargs["prefetch_factor"] = int(data_cfg["prefetch_factor"])

		dataloader = DataLoader(dataset, **dataloader_kwargs)
		return dataset, sampler, dataloader

	"""构建 VQAP 模型，并在 Stage 0 起始时把 LoRA 仅挂到 DINOv2 backbone。"""
	def _init_model(self, apply_lora: bool) -> VQAP:
		model = VQAP(model_args=self.model_args, train_args=self.train_args).to(self.device)
		if apply_lora:
			self._apply_dinov2_lora(model)
		return model

	"""统一拿到未包 DDP 的真实模型对象，便于访问子模块与保存权重。"""
	def _get_model_module(self) -> VQAP:
		if isinstance(self.model, DDP):
			return self.model.module
		return self.model

	"""临时拆开 DDP 包装，用于 stage 切换时重建 backbone / optimizer。"""
	def _unwrap_ddp_model(self) -> VQAP:
		if isinstance(self.model, DDP):
			self.model = self.model.module
		return self.model

	"""在多卡场景下重新包裹 DDP；单卡或 CPU 场景下保持原模型不变。"""
	def _wrap_model_for_ddp(self) -> None:
		if self.distributed:
			self.model = DDP(
				self.model,
				device_ids=[self.local_rank],
				output_device=self.local_rank,
				find_unused_parameters=True,
				gradient_as_bucket_view=True,
			)

	"""返回唯一允许挂 LoRA 的位置：VASA 内部 DINOv2 backbone。"""
	def _get_dinov2_backbone(self, model: VQAP) -> torch.nn.Module:
		return model.vasa.image_encoder.feature_extractor.backbone

	"""用 PEFT 把 LoRA 仅挂到 DINOv2 backbone 的目标注意力层。"""
	def _apply_dinov2_lora(self, model: VQAP) -> None:
		lora_cfg = self.train_args["lora"]
		backbone = self._get_dinov2_backbone(model)
		peft_config = LoraConfig(
			r=int(lora_cfg["r"]),
			lora_alpha=int(lora_cfg["alpha"]),
			lora_dropout=float(lora_cfg["dropout"]),
			bias=str(lora_cfg.get("bias", "none")),
			target_modules=list(lora_cfg["target_modules"]),
		)
		peft_backbone = get_peft_model(backbone, peft_config)
		model.vasa.image_encoder.feature_extractor.backbone = peft_backbone
		if self.rank == 0:
			self.logger.info("Applied LoRA to DINOv2 backbone")
			peft_backbone.print_trainable_parameters()

	"""批量设置某个子模块的 requires_grad，供 stage 切换时复用。"""
	def _set_module_requires_grad(self, module: torch.nn.Module, requires_grad: bool) -> None:
		for parameter in module.parameters():
			parameter.requires_grad = requires_grad

	"""进入 Stage 1 后冻结视觉对齐支路，只保留码本路径与 future predictor 继续训练。"""
	def _freeze_stage1_modules(self, model: VQAP) -> None:
		self._set_module_requires_grad(self._get_dinov2_backbone(model), False)
		self._set_module_requires_grad(model.vasa.visual_transformer_encoder, False)
		self._set_module_requires_grad(model.vasa.flow_matching_head, False)
		self._set_module_requires_grad(model.vasa.future_predictor, True)
		self._set_module_requires_grad(model.atomaction_nsvq, True)

	"""Stage 1 中把冻结的视觉子模块切到 eval，避免 dropout 等训练态扰动。"""
	def _set_stage1_eval_modes(self) -> None:
		model = self._get_model_module()
		self._get_dinov2_backbone(model).eval()
		model.vasa.visual_transformer_encoder.eval()
		model.vasa.flow_matching_head.eval()

	"""判断某个参数是否应当跳过 weight decay。"""
	def _should_skip_weight_decay(self, name: str, parameter: torch.nn.Parameter) -> bool:
		name_lower = name.lower()
		if parameter.ndim < 2:
			return True
		if name.endswith(".bias"):
			return True
		if "norm" in name_lower:
			return True
		if "embedding" in name_lower:
			return True
		if "codebooks" in name_lower:
			return True
		return False

	"""把同一类参数按 decay / no-decay 拆分成 AdamW 参数组。"""
	def _build_param_groups(
		self,
		named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
		group_prefix: str,
		lr: float,
		beta2: float,
	) -> list[Dict[str, Any]]:
		optimizer_cfg = self.train_args["optimizer"]
		decay_params = []
		no_decay_params = []
		for name, parameter in named_parameters:
			if not parameter.requires_grad:
				continue
			if self._should_skip_weight_decay(name, parameter):
				no_decay_params.append(parameter)
			else:
				decay_params.append(parameter)

		param_groups = []
		if decay_params:
			param_groups.append(
				{
					"params": decay_params,
					"lr": lr,
					"betas": (float(optimizer_cfg["beta1"]), beta2),
					"eps": float(optimizer_cfg["eps"]),
					"weight_decay": float(optimizer_cfg["weight_decay"]),
					"group_name": f"{group_prefix}_decay",
				}
			)
		if no_decay_params:
			param_groups.append(
				{
					"params": no_decay_params,
					"lr": lr,
					"betas": (float(optimizer_cfg["beta1"]), beta2),
					"eps": float(optimizer_cfg["eps"]),
					"weight_decay": 0.0,
					"group_name": f"{group_prefix}_nodecay",
				}
			)
		return param_groups

	"""按当前 stage 构建优化器。

	Stage 0:
		main 参数组 + LoRA 参数组
	Stage 1:
		只有仍需训练的主干参数组
	"""
	def _init_optimizer(self, stage: int) -> AdamW:
		model = self._get_model_module()
		lr_cfg = self.train_args["lr"]
		optimizer_cfg = self.train_args["optimizer"]

		main_named_parameters = []
		lora_named_parameters = []
		for name, parameter in model.named_parameters():
			if not parameter.requires_grad:
				continue
			if "lora_" in name:
				lora_named_parameters.append((name, parameter))
			else:
				main_named_parameters.append((name, parameter))

		param_groups = []
		if stage == 0:
			param_groups.extend(
				self._build_param_groups(
					named_parameters=main_named_parameters,
					group_prefix="main",
					lr=float(lr_cfg["stage0_main"]),
					beta2=float(optimizer_cfg["main_beta2"]),
				)
			)
			param_groups.extend(
				self._build_param_groups(
					named_parameters=lora_named_parameters,
					group_prefix="lora",
					lr=float(lr_cfg["stage0_lora"]),
					beta2=float(optimizer_cfg["lora_beta2"]),
				)
			)
		else:
			param_groups.extend(
				self._build_param_groups(
					named_parameters=main_named_parameters,
					group_prefix="main",
					lr=float(lr_cfg["stage1_main"]),
					beta2=float(optimizer_cfg["main_beta2"]),
				)
			)

		if not param_groups:
			raise RuntimeError("No trainable parameters were found when building the optimizer")
		return AdamW(param_groups)

	"""为当前 stage 构建独立的 warmup + cosine scheduler。"""
	def _init_scheduler(self, stage: int) -> LambdaLR:
		scheduler_cfg = self.train_args["scheduler"]
		total_epochs = self.stage0_epochs if stage == 0 else self.stage1_epochs
		return LambdaLR(
			self.optimizer,
			make_lr_lambda(
				warmup_epochs=int(scheduler_cfg["warmup_epochs"]),
				total_epochs=total_epochs,
				min_ratio=float(scheduler_cfg["min_lr_ratio"]),
			),
		)

	"""根据 precision 配置返回 autocast 上下文；fp32 时退化为空上下文。"""
	def _get_autocast_context(self) -> Any:
		if self.autocast_dtype is None:
			return nullcontext()
		return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

	"""读取当前 epoch 对应的 future loss 权重。"""
	def _get_lambda_future(self, epoch: int) -> float:
		return compute_future_weight_schedule(
			loss_cfg=self.train_args["loss"],
			stage_cfg=self.train_args["stage"],
			epoch=epoch,
		)

	"""执行一次完整前向，并按当前 stage 组装总损失。

	关键点：
	- `L_AP` 全程启用
	- `L_AG` 仅在 Stage 0 加入总损失
	- `L_future` 权重由当前 epoch 的调度结果决定
	"""
	def _compute_loss(self, batch: Dict[str, Any], epoch: int) -> Dict[str, torch.Tensor]:
		loss_cfg = self.train_args["loss"]
		model_outputs = self.model(
			trajectory_data=batch["trajectory_data"],
			trajectory_mask=batch["trajectory_mask"],
			selected_views=batch["selected_views"],
		)
		shared_outputs = model_outputs["shared_outputs"]
		atomaction_outputs = model_outputs["atomaction_outputs"]
		vasa_outputs = model_outputs["vasa_outputs"]

		atomaction_loss_outputs = compute_atomaction_reconstruction_loss(
			shared_outputs=shared_outputs,
			atomaction_outputs=atomaction_outputs,
			loss_cfg=loss_cfg,
		)
		loss_future = compute_future_patch_loss(
			pred_end_patch_features=vasa_outputs["pred_end_patch_features"],
			end_img_features=vasa_outputs["end_img_features"],
		)
		lambda_future = self._get_lambda_future(epoch)

		# 总损失从动作重构主线开始，再按训练阶段决定是否叠加视觉对齐分支。
		loss_total = float(loss_cfg["lambda_ap"]) * atomaction_loss_outputs["loss_ap"] + lambda_future * loss_future
		zero = loss_total.new_zeros(())
		if self.current_stage == 0:
			action_grounding_loss_outputs = compute_action_grounding_loss(
				shared_outputs=shared_outputs,
				vasa_outputs=vasa_outputs,
				loss_cfg=loss_cfg,
			)
			loss_total = loss_total + float(loss_cfg["lambda_ag"]) * action_grounding_loss_outputs["loss_ag"]
		else:
			action_grounding_loss_outputs = {
				"loss_fm_pos_ag": zero,
				"loss_fm_rot_ag": zero,
				"loss_bce_grip_ag": zero,
				"loss_ag": zero,
			}

		loss_outputs: Dict[str, torch.Tensor] = {}
		loss_outputs.update(atomaction_loss_outputs)
		loss_outputs.update(action_grounding_loss_outputs)
		loss_outputs["loss_future"] = loss_future
		loss_outputs["lambda_future"] = loss_total.new_tensor(lambda_future)
		loss_outputs["loss_total"] = loss_total
		loss_outputs["perplexity_g"] = atomaction_outputs["global_perplexity"]
		loss_outputs["perplexity_d"] = atomaction_outputs["detail_perplexity"]
		return loss_outputs

	"""提取当前 optimizer 中各参数组的学习率，用于日志记录。"""
	def _get_lr_logs(self) -> Dict[str, float]:
		lr_logs: Dict[str, float] = {}
		for param_group in self.optimizer.param_groups:
			group_name = str(param_group.get("group_name", "group"))
			if group_name.startswith("main") and "lr/main" not in lr_logs:
				lr_logs["lr/main"] = float(param_group["lr"])
			elif group_name.startswith("lora") and "lr/lora" not in lr_logs:
				lr_logs["lr/lora"] = float(param_group["lr"])
		return lr_logs

	"""把 epoch 转成更直观的 1-based 记录步数，供 wandb / checkpoint 日志复用。"""
	def _get_epoch_log_step(self, epoch: int) -> int:
		return int(epoch) + 1

	"""执行单个 epoch 的训练循环，并聚合 epoch 级平均指标。"""
	def _train_epoch(self, epoch: int) -> Dict[str, float]:
		self.model.train()
		if self.current_stage == 1:
			self._set_stage1_eval_modes()

		metric_sums = {metric_name: 0.0 for metric_name in EPOCH_METRIC_KEYS}
		num_steps = 0
		last_grad_norm_value = 0.0

		for batch in self.train_loader:
			# `selected_views` 保持列表结构，只有轨迹张量会被搬到 device 上。
			batch = move_batch_to_device(batch, self.device)
			self.optimizer.zero_grad(set_to_none=True)
			with self._get_autocast_context():
				loss_outputs = self._compute_loss(batch, epoch=epoch)
				loss_total = loss_outputs["loss_total"]

			# fp16 时使用 GradScaler；bf16/fp32 直接常规反传。
			if self.grad_scaler.is_enabled():
				self.grad_scaler.scale(loss_total).backward()
				self.grad_scaler.unscale_(self.optimizer)
				grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
				self.grad_scaler.step(self.optimizer)
				self.grad_scaler.update()
			else:
				loss_total.backward()
				grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
				self.optimizer.step()

			last_grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
			for metric_name in EPOCH_METRIC_KEYS:
				metric_sums[metric_name] += float(loss_outputs[metric_name].detach().item())
			num_steps += 1

			self.global_step += 1

		metric_names = list(EPOCH_METRIC_KEYS)
		local_metrics = torch.tensor(
			[metric_sums[metric_name] for metric_name in metric_names] + [float(num_steps)],
			device=self.device,
			dtype=torch.float64,
		)
		if self.distributed:
			dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)

		global_batch_count = max(local_metrics[-1].item(), 1.0)
		epoch_metrics = {
			metric_name: local_metrics[index].item() / global_batch_count
			for index, metric_name in enumerate(metric_names)
		}
		grad_norm_tensor = torch.tensor([last_grad_norm_value], device=self.device, dtype=torch.float64)
		if self.distributed:
			dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM)
			grad_norm_tensor /= float(self.world_size)
		epoch_metrics["grad_norm"] = float(grad_norm_tensor.item())
		epoch_metrics["lambda_future"] = self._get_lambda_future(epoch)
		return epoch_metrics

	"""在多卡模式下广播主进程更新后的张量状态。"""
	def _broadcast_tensor(self, tensor: torch.Tensor) -> None:
		if self.distributed:
			dist.broadcast(tensor, src=0)

	"""执行一次真实的死码替换，并把更新后的码本同步给所有进程。"""
	def _replace_dead_codebooks(self) -> Dict[str, int]:
		model = self._get_model_module()
		global_quantizer = model.atomaction_nsvq.global_codebook_module.quantizer
		detail_quantizer = model.atomaction_nsvq.detail_codebook_module.quantizer

		if self.distributed:
			dist.all_reduce(global_quantizer.codebooks_used, op=dist.ReduceOp.SUM)
			dist.all_reduce(detail_quantizer.codebooks_used, op=dist.ReduceOp.SUM)

		replaced_g = 0
		replaced_d = 0
		if self.rank == 0:
			replace_outputs = model.replace_unused_codebooks()
			replaced_g = int(replace_outputs["replaced_codebooks_g"].item())
			replaced_d = int(replace_outputs["replaced_codebooks_d"].item())

		replaced_tensor = torch.tensor([replaced_g, replaced_d], device=self.device, dtype=torch.long)
		self._broadcast_tensor(replaced_tensor)
		self._broadcast_tensor(global_quantizer.codebooks)
		self._broadcast_tensor(global_quantizer.codebooks_used)
		self._broadcast_tensor(detail_quantizer.codebooks)
		self._broadcast_tensor(detail_quantizer.codebooks_used)
		return {
			"replaced_codebooks_g": int(replaced_tensor[0].item()),
			"replaced_codebooks_d": int(replaced_tensor[1].item()),
		}

	"""根据 perplexity 阈值或固定间隔，决定当前 epoch 是否执行死码替换。"""
	def _maybe_replace_dead_codebooks(self, epoch: int, epoch_metrics: Dict[str, float]) -> Dict[str, int]:
		codebook_cfg = self.train_args["codebook"]
		perplexity_g_threshold = float(codebook_cfg["perplexity_g_threshold"])
		perplexity_d_threshold = float(codebook_cfg["perplexity_d_threshold"])
		replace_interval_epochs = int(codebook_cfg["replace_interval_epochs"])

		trigger_by_perplexity = (
			epoch_metrics["perplexity_g"] < perplexity_g_threshold
			or epoch_metrics["perplexity_d"] < perplexity_d_threshold
		)
		trigger_by_interval = replace_interval_epochs > 0 and (epoch + 1) % replace_interval_epochs == 0
		if not (trigger_by_perplexity or trigger_by_interval):
			return {
				"replaced_codebooks_g": 0,
				"replaced_codebooks_d": 0,
			}
		return self._replace_dead_codebooks()

	"""用先写临时文件再原子替换的方式保存 checkpoint，避免中途中断写坏文件。"""
	def _atomic_torch_save(self, payload: Dict[str, Any], path: Path) -> None:
		tmp_path = path.with_suffix(path.suffix + ".tmp")
		torch.save(payload, tmp_path)
		tmp_path.replace(path)

	"""保存 latest / best / codebook 三类 checkpoint。"""
	def _save_checkpoint(self, epoch: int, epoch_metrics: Dict[str, float]) -> None:
		if self.rank != 0:
			return

		model = self._get_model_module()
		log_step = self._get_epoch_log_step(epoch)
		checkpoint_payload = {
			"epoch": epoch + 1,
			"global_step": self.global_step,
			"stage": self.current_stage,
			"model": model.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"scheduler": self.scheduler.state_dict(),
			"best_lap": self.best_lap,
			"best_ltotal": self.best_ltotal,
			"model_args": self.model_args,
			"train_args": self.train_args,
			"global_args": self.global_args,
			"wandb_run_id": getattr(self.wandb_run, "id", None),
		}
		self._atomic_torch_save(checkpoint_payload, self.ckpt_dir / "latest.pth")

		# `best_lap` 更关注动作重构质量，`best_ltotal` 则保留当前训练目标下的综合最优。
		if epoch_metrics["loss_ap"] < self.best_lap:
			self.best_lap = epoch_metrics["loss_ap"]
			checkpoint_payload["best_lap"] = self.best_lap
			checkpoint_payload["best_ltotal"] = self.best_ltotal
			self._atomic_torch_save(checkpoint_payload, self.ckpt_dir / "best_lap.pth")
			if self.wandb_run is not None:
				self.wandb_run.log({"ckpt/best_lap": self.best_lap}, step=log_step)

		if epoch_metrics["loss_total"] < self.best_ltotal:
			self.best_ltotal = epoch_metrics["loss_total"]
			checkpoint_payload["best_lap"] = self.best_lap
			checkpoint_payload["best_ltotal"] = self.best_ltotal
			self._atomic_torch_save(checkpoint_payload, self.ckpt_dir / "best_ltotal.pth")
			if self.wandb_run is not None:
				self.wandb_run.log({"ckpt/best_ltotal": self.best_ltotal}, step=log_step)

		codebook_payload = {
			"epoch": epoch + 1,
			"global_step": self.global_step,
			"stage": self.current_stage,
			"perplexity_g": epoch_metrics["perplexity_g"],
			"perplexity_d": epoch_metrics["perplexity_d"],
			"global_codebook": model.atomaction_nsvq.global_codebook_module.quantizer.state_dict(),
			"detail_codebook": model.atomaction_nsvq.detail_codebook_module.quantizer.state_dict(),
		}
		self._atomic_torch_save(codebook_payload, self.ckpt_dir / "codebook.pth")

	"""执行 Stage 0 -> Stage 1 切换。

	关键步骤：
	- merge 并移除 LoRA 结构
	- 冻结视觉对齐支路
	- 重建 optimizer / scheduler
	- 重置 best 指标，避免跨 stage 直接比较
	"""
	def _setup_stage1(self, epoch: int) -> None:
		if self.current_stage != 0:
			return

		if self.rank == 0:
			self.logger.info("Switching from Stage 0 to Stage 1")
		if self.distributed:
			dist.barrier()

		model = self._unwrap_ddp_model()
		backbone = self._get_dinov2_backbone(model)
		if not isinstance(backbone, PeftModel):
			raise TypeError("Stage 1 setup expects the DINOv2 backbone to be a PeftModel before merge")

		merged_backbone = backbone.merge_and_unload()
		model.vasa.image_encoder.feature_extractor.backbone = merged_backbone
		self._freeze_stage1_modules(model)
		self.model = model
		self._wrap_model_for_ddp()
		self.current_stage = 1
		self.optimizer = self._init_optimizer(stage=1)
		self.scheduler = self._init_scheduler(stage=1)
		self.best_lap = float("inf")
		self.best_ltotal = float("inf")

		if self.rank == 0 and self.wandb_run is not None:
			self.wandb_run.log({"train/stage": 1.0}, step=self._get_epoch_log_step(epoch))
		if self.distributed:
			dist.barrier()

	"""记录 epoch 级摘要日志，并同步到 wandb。"""
	def _log_epoch_summary(self, epoch: int, epoch_metrics: Dict[str, float], epoch_time_seconds: float) -> None:
		if self.rank != 0:
			return

		log_payload = {
			"train/loss_total": epoch_metrics["loss_total"],
			"train/loss_ap": epoch_metrics["loss_ap"],
			"train/loss_fm_pos_ap": epoch_metrics["loss_fm_pos_ap"],
			"train/loss_fm_rot_ap": epoch_metrics["loss_fm_rot_ap"],
			"train/loss_bce_grip_ap": epoch_metrics["loss_bce_grip_ap"],
			"train/loss_sep": epoch_metrics["loss_sep"],
			"train/loss_ag": epoch_metrics["loss_ag"],
			"train/loss_fm_pos_ag": epoch_metrics["loss_fm_pos_ag"],
			"train/loss_fm_rot_ag": epoch_metrics["loss_fm_rot_ag"],
			"train/loss_bce_grip_ag": epoch_metrics["loss_bce_grip_ag"],
			"train/loss_future": epoch_metrics["loss_future"],
			"train/grad_norm": epoch_metrics["grad_norm"],
			"train/lambda_future": epoch_metrics["lambda_future"],
			"train/stage": float(self.current_stage),
		}
		log_payload.update(self._get_lr_logs())
		if self.wandb_run is not None:
			self.wandb_run.log(log_payload, step=self._get_epoch_log_step(epoch))

		self.logger.info(
			f"epoch={epoch + 1}/{self.total_epochs} | stage={self.current_stage} | "
			f"loss_total={epoch_metrics['loss_total']:.4f} | loss_ap={epoch_metrics['loss_ap']:.4f} | "
			f"loss_ag={epoch_metrics['loss_ag']:.4f} | loss_future={epoch_metrics['loss_future']:.4f} | "
			f"lambda_future={epoch_metrics['lambda_future']:.4f} | "
			f"perp_g={epoch_metrics['perplexity_g']:.2f} | perp_d={epoch_metrics['perplexity_d']:.2f} | "
			f"time={epoch_time_seconds:.2f}s"
		)

	"""外层训练循环：stage 切换、epoch 训练、scheduler 更新、码本维护与保存。"""
	def train(self) -> None:
		for epoch in range(self.start_epoch, self.total_epochs):
			if self.current_stage == 0 and epoch >= self.stage0_epochs:
				self._setup_stage1(epoch)

			if isinstance(self.train_sampler, DistributedSampler):
				# DDP 下每个 epoch 都要重设 sampler 的随机种子，避免各卡重复取样。
				self.train_sampler.set_epoch(epoch)

			epoch_start_time = time.time()
			epoch_metrics = self._train_epoch(epoch)
			self.scheduler.step()

			if (epoch + 1) % self.save_every_epochs == 0 or (epoch + 1) == self.total_epochs:
				# 先根据当前 perplexity 决定是否替换死码，再把更新后的状态一起保存。
				replace_metrics = self._maybe_replace_dead_codebooks(epoch, epoch_metrics)
				epoch_metrics.update(replace_metrics)
				self._save_checkpoint(epoch, epoch_metrics)
				if self.rank == 0 and self.wandb_run is not None:
					self.wandb_run.log(
						{
							"codebook/perplexity_g": epoch_metrics["perplexity_g"],
							"codebook/perplexity_d": epoch_metrics["perplexity_d"],
							"codebook/replaced_codebooks_g": float(replace_metrics["replaced_codebooks_g"]),
							"codebook/replaced_codebooks_d": float(replace_metrics["replaced_codebooks_d"]),
							"codebook/replace_triggered": float(
								replace_metrics["replaced_codebooks_g"] > 0 or replace_metrics["replaced_codebooks_d"] > 0
							),
						},
						step=self._get_epoch_log_step(epoch),
					)

			self._log_epoch_summary(
				epoch=epoch,
				epoch_metrics=epoch_metrics,
				epoch_time_seconds=time.time() - epoch_start_time,
			)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the VQAP pretraining pipeline")
	parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to train.yaml")
	parser.add_argument("--model-config", type=str, default="config/model.yaml", help="Path to model.yaml")
	parser.add_argument("--global-config", type=str, default="config/global.yaml", help="Path to global.yaml")
	parser.add_argument("--exp-name", type=str, default=None, help="Override experiment name in train.yaml")
	parser.add_argument("--resume", action="store_true", help="Resume from latest.pth or --resume-path")
	parser.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint path for resuming")
	parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb even if train.yaml enables it")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	trainer = VQAPTrainer(args)
	try:
		trainer.train()
	finally:
		finish_wandb(rank=trainer.rank, run=trainer.wandb_run)
		if trainer.distributed and dist.is_initialized():
			dist.destroy_process_group()


if __name__ == "__main__":
	main()
