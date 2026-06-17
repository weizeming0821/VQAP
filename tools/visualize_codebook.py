"""码本可视化脚本。

加载训练好的 checkpoint，在数据集上跑一遍（只走 AtomAction 轨迹编码路径，不加载图像 / VASA / DINOv2），
统计两套码本（global / detail）的使用情况与语义对齐情况，输出以下图：

	1. 使用频次直方图       —— 看死码、利用率是否均匀
	2. 归一化困惑度 / 活跃码占比（打印 + 标题） —— 利用率标量
	3. 码字余弦相似度矩阵   —— 看码本冗余 / 塌缩
	4. t-SNE / PCA 叠加图   —— 编码特征 + 码字一起降维，看分布与聚簇
	5. 码字 × 动作类型热力图 —— 看码字是否抓到了动作语义

设计说明：
	- 码本分配只依赖 trajectory_data，与图像无关，因此本脚本只实例化 AtomAction_NSVQ，
	  并把 dataset.view_select 短路掉，省去图像读取与 DINOv2 加载。
	- 量化器在 eval 下要求显式传入索引，这里改用 compute_codebook_distances + argmin 手动算分配，
	  既绕开该限制又能在 eval（无 dropout）下复现训练时的硬最近邻分配。

用法示例：
	python scripts/visualize_codebook.py \
		--checkpoint checkpoints/vqap_pretrain/stage1/best_lap.pth \
		--output-dir viz/codebook/stage1 \
		--max-batches 0
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from data.dataset import AtomActionDataset
from data.utils import AtomActionDataset_collate_fn
from model.module.atomaction_nsvq import AtomAction_NSVQ
from model.module.nsvq import compute_perplexity, masked_average_pool


"""从完整 checkpoint 中抽取 atomaction_nsvq 子模块权重并加载。"""
def load_atomaction_model(checkpoint_path: Path, device: torch.device) -> Tuple[AtomAction_NSVQ, Dict[str, Any]]:
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	if "model" not in checkpoint:
		raise ValueError(
			f"checkpoint 内没有完整 model 权重（{checkpoint_path}）；"
			f"codebook.pth 仅含码本，无编码器，无法复现分配，请用 latest.pth / best_lap.pth。"
		)
	model_args = checkpoint["model_args"]

	prefix = "atomaction_nsvq."
	sub_state = {
		key[len(prefix):]: value
		for key, value in checkpoint["model"].items()
		if key.startswith(prefix)
	}
	if not sub_state:
		raise ValueError("checkpoint['model'] 中找不到 atomaction_nsvq.* 权重")

	model = AtomAction_NSVQ(model_args=model_args)
	missing, unexpected = model.load_state_dict(sub_state, strict=False)
	# codebooks_used 是 buffer，统计用，缺失/多余都不影响分配，仅提示。
	if missing:
		print(f"[warn] missing keys: {missing}")
	if unexpected:
		print(f"[warn] unexpected keys: {unexpected}")

	model.to(device).eval()
	return model, checkpoint


"""构建数据集与 DataLoader；短路 view_select 以跳过图像与 DINOv2。"""
def build_dataloader(global_args: Dict[str, Any], batch_size: int, num_workers: int) -> DataLoader:
	dataset_cfg = global_args["atomactiondataset"]
	dataset = AtomActionDataset(
		dataset_root=str(dataset_cfg["dataset_root"]),
		views=dataset_cfg.get("views"),
		top_k=int(dataset_cfg.get("top_k", dataset_cfg.get("tok_k", 1))),
		view_selector_kwargs=dict(dataset_cfg.get("view_selector", {})),
	)
	# 码本分配不需要图像，短路掉视角选择，避免读图与加载 DINOv2。
	dataset.view_select = lambda sample: []
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=AtomActionDataset_collate_fn,
	)


"""复现 AtomAction_NSVQ 的轨迹编码与码本硬分配（eval、无 dropout、不走 flow matching / 图像）。

返回：
	global_features: [B, D]   投影后未量化的全局特征
	global_indices:  [B]      全局码索引
	detail_features: [B, Q, D] 投影后未量化的细节特征
	detail_indices:  [B, Q]   细节码索引
"""
@torch.no_grad()
def encode_batch(
	model: AtomAction_NSVQ,
	batch: Dict[str, Any],
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	trajectory_data = {key: value.to(device) for key, value in batch["trajectory_data"].items()}
	trajectory_mask = batch["trajectory_mask"].to(device)

	# 与 AtomAction_NSVQ.forward 完全一致的四分支投影 + 拼接。
	ee_features = model.ee_projector(trajectory_data["gripper_pose"])
	body_inputs = torch.cat(
		(trajectory_data["joint_positions"], trajectory_data["joint_velocities"], trajectory_data["joint_forces"]),
		dim=-1,
	)
	body_features = model.body_projector(body_inputs)
	gripper_mech_inputs = torch.cat(
		(trajectory_data["gripper_joint_positions"], trajectory_data["gripper_touch_forces"]),
		dim=-1,
	)
	gripper_mech_features = model.gripper_mech_projector(gripper_mech_inputs)
	gripper_open_ids = trajectory_data["gripper_open"].squeeze(-1).round().clamp(0, 1).long()
	gripper_open_features = model.gripper_open_projector(model.gripper_open_embedding(gripper_open_ids))

	trajectory_features = torch.cat(
		(ee_features, body_features, gripper_mech_features, gripper_open_features),
		dim=-1,
	)
	channel_encoded = model.channel_encoder(trajectory_features, trajectory_mask)
	encoded = model.transformer_encoder(channel_encoded, trajectory_mask)

	# 全局码本：masked average pool -> 投影 -> 硬最近邻。
	global_module = model.global_codebook_module
	pooled = masked_average_pool(encoded, trajectory_mask)
	global_features = global_module.projection(pooled)
	global_indices = global_module.quantizer.compute_codebook_distances(global_features).argmin(dim=-1)

	# 细节码本：learnable query cross-attention -> 投影 -> 硬最近邻。
	detail_module = model.detail_codebook_module
	batch_size = encoded.shape[0]
	queries = detail_module.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)
	detail_query_features = detail_module.cross_attention(
		query=queries,
		key=encoded,
		value=encoded,
		trajectory_mask=trajectory_mask,
	)
	detail_features = detail_module.projection(detail_query_features)
	flat_detail = detail_features.reshape(batch_size * detail_module.num_queries, -1)
	detail_indices = detail_module.quantizer.compute_codebook_distances(flat_detail).argmin(dim=-1)
	detail_indices = detail_indices.view(batch_size, detail_module.num_queries)

	return global_features, global_indices, detail_features, detail_indices


"""遍历数据集，收集两套码本的特征、索引与动作标签。"""
@torch.no_grad()
def collect_statistics(
	model: AtomAction_NSVQ,
	dataloader: DataLoader,
	device: torch.device,
	max_batches: int,
) -> Dict[str, Any]:
	global_features_list: List[np.ndarray] = []
	global_indices_list: List[np.ndarray] = []
	detail_features_list: List[np.ndarray] = []
	detail_indices_list: List[np.ndarray] = []
	actions: List[str] = []

	for batch_index, batch in enumerate(dataloader):
		if max_batches > 0 and batch_index >= max_batches:
			break
		g_feat, g_idx, d_feat, d_idx = encode_batch(model, batch, device)
		global_features_list.append(g_feat.float().cpu().numpy())
		global_indices_list.append(g_idx.cpu().numpy())
		detail_features_list.append(d_feat.float().cpu().numpy())          # [B, Q, D]
		detail_indices_list.append(d_idx.cpu().numpy())                    # [B, Q]
		actions.extend(batch["Action"])
		print(f"\rprocessed batch {batch_index + 1}", end="", flush=True)
	print()

	detail_features = np.concatenate(detail_features_list, axis=0)         # [N, Q, D]
	detail_indices = np.concatenate(detail_indices_list, axis=0)          # [N, Q]
	return {
		"global_features": np.concatenate(global_features_list, axis=0),  # [N, D]
		"global_indices": np.concatenate(global_indices_list, axis=0),    # [N]
		"detail_features": detail_features.reshape(-1, detail_features.shape[-1]),  # [N*Q, D]
		"detail_indices": detail_indices.reshape(-1),                     # [N*Q]
		"detail_indices_per_traj": detail_indices,                        # [N, Q]
		"actions": np.array(actions),                                     # [N]
		"global_codebook": model.global_codebook_module.quantizer.codebooks.detach().float().cpu().numpy(),
		"detail_codebook": model.detail_codebook_module.quantizer.codebooks.detach().float().cpu().numpy(),
	}


"""画码字使用频次直方图，并在标题标注活跃码占比与归一化困惑度。"""
def plot_usage_histogram(indices: np.ndarray, codebook_size: int, name: str, output_path: Path) -> Dict[str, float]:
	counts = np.bincount(indices, minlength=codebook_size).astype(np.float64)
	active_ratio = float((counts > 0).mean())
	perplexity = float(compute_perplexity(torch.from_numpy(indices).long(), codebook_size).item())
	normalized_perplexity = perplexity / float(codebook_size)

	figure, axis = plt.subplots(figsize=(max(6.0, codebook_size * 0.06), 4.0))
	axis.bar(np.arange(codebook_size), counts, width=1.0, color="#4878CF")
	axis.set_xlabel("codeword id")
	axis.set_ylabel("assignment count")
	axis.set_title(
		f"{name} usage  |  active={active_ratio:.1%} ({int((counts > 0).sum())}/{codebook_size})  "
		f"|  perplexity={perplexity:.1f}  norm={normalized_perplexity:.2f}"
	)
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)
	return {"active_ratio": active_ratio, "perplexity": perplexity, "normalized_perplexity": normalized_perplexity}


"""画码字两两余弦相似度矩阵热力图。"""
def plot_codebook_similarity(codebook: np.ndarray, name: str, output_path: Path) -> None:
	normalized = codebook / (np.linalg.norm(codebook, axis=1, keepdims=True) + 1e-8)
	similarity = normalized @ normalized.T

	figure, axis = plt.subplots(figsize=(6.0, 5.0))
	image = axis.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0)
	axis.set_title(f"{name} codeword cosine similarity")
	axis.set_xlabel("codeword id")
	axis.set_ylabel("codeword id")
	figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)


"""把编码特征与码字一起降维，叠加可视化（PCA + t-SNE）。

	features:  [N, D] 编码特征
	indices:   [N]    每个特征分配到的码字 id（用于上色）
	codebook:  [K, D] 码字
	color_by_action: 若提供动作标签则按动作上色，否则按码字 id 上色
"""
def plot_embedding_overlay(
	features: np.ndarray,
	indices: np.ndarray,
	codebook: np.ndarray,
	name: str,
	output_path: Path,
	max_points: int,
	actions: np.ndarray = None,
) -> None:
	# 点太多时下采样，t-SNE 才跑得动。
	num_points = features.shape[0]
	if num_points > max_points:
		selected = np.random.RandomState(0).choice(num_points, size=max_points, replace=False)
		features = features[selected]
		indices = indices[selected]
		if actions is not None:
			actions = actions[selected]

	num_codes = codebook.shape[0]
	stacked = np.concatenate([features, codebook], axis=0)

	figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.5))
	for axis, (method_name, coords) in zip(axes, _reduce_dims(stacked)):
		feature_coords = coords[:-num_codes]
		code_coords = coords[-num_codes:]

		if actions is not None:
			unique_actions = sorted(set(actions.tolist()))
			color_map = {action: idx for idx, action in enumerate(unique_actions)}
			color_values = np.array([color_map[action] for action in actions])
			scatter = axis.scatter(
				feature_coords[:, 0], feature_coords[:, 1],
				c=color_values, cmap="tab20", s=10, alpha=0.6,
			)
			handles = [
				plt.Line2D([0], [0], marker="o", linestyle="", color=scatter.cmap(scatter.norm(color_map[a])), label=a)
				for a in unique_actions
			]
			axis.legend(handles=handles, fontsize=6, loc="best", ncol=2, framealpha=0.5)
		else:
			axis.scatter(
				feature_coords[:, 0], feature_coords[:, 1],
				c=indices, cmap="tab20", s=10, alpha=0.6,
			)

		axis.scatter(
			code_coords[:, 0], code_coords[:, 1],
			c="black", marker="X", s=80, edgecolors="white", linewidths=0.8, label="codewords",
		)
		axis.set_title(f"{name} {method_name}")
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)


"""对同一份数据分别做 PCA(2) 与 t-SNE(2)，返回 [(name, coords), ...]。"""
def _reduce_dims(stacked: np.ndarray) -> List[Tuple[str, np.ndarray]]:
	pca_coords = PCA(n_components=2, random_state=0).fit_transform(stacked)
	perplexity = min(30, max(5, stacked.shape[0] // 4))
	tsne_coords = TSNE(
		n_components=2,
		random_state=0,
		perplexity=perplexity,
		init="pca",
	).fit_transform(stacked)
	return [("PCA", pca_coords), ("t-SNE", tsne_coords)]


"""画码字 × 动作类型的行归一化热力图，看每个码字是否对应固定动作语义。"""
def plot_code_action_heatmap(
	indices: np.ndarray,
	actions: np.ndarray,
	codebook_size: int,
	name: str,
	output_path: Path,
) -> None:
	unique_actions = sorted(set(actions.tolist()))
	action_to_col = {action: col for col, action in enumerate(unique_actions)}

	matrix = np.zeros((codebook_size, len(unique_actions)), dtype=np.float64)
	for code_index, action in zip(indices, actions):
		matrix[code_index, action_to_col[action]] += 1.0

	# 只保留被用到的码字行，避免大量空行。
	used_rows = matrix.sum(axis=1) > 0
	matrix = matrix[used_rows]
	used_code_ids = np.arange(codebook_size)[used_rows]
	row_sums = matrix.sum(axis=1, keepdims=True)
	normalized = matrix / np.clip(row_sums, 1.0, None)

	figure, axis = plt.subplots(figsize=(max(6.0, len(unique_actions) * 0.5), max(4.0, used_rows.sum() * 0.22)))
	image = axis.imshow(normalized, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
	axis.set_xticks(np.arange(len(unique_actions)))
	axis.set_xticklabels(unique_actions, rotation=90, fontsize=7)
	axis.set_yticks(np.arange(len(used_code_ids)))
	axis.set_yticklabels(used_code_ids, fontsize=6)
	axis.set_xlabel("action type")
	axis.set_ylabel("codeword id")
	axis.set_title(f"{name} code -> action distribution (row-normalized)")
	figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize learned VQAP codebooks")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default="checkpoints/vqap_pretrain/stage1/best_lap.pth",
		help="完整 checkpoint 路径（需含 model 权重，如 latest.pth / best_lap.pth）",
	)
	parser.add_argument("--output-dir", type=str, default=None, help="图片输出目录，默认 viz/codebook/<checkpoint 所在目录名>")
	parser.add_argument("--global-config", type=str, default="config/global.yaml", help="数据集配置；默认读 config/global.yaml")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--max-batches", type=int, default=0, help="最多处理多少个 batch，0 表示全量")
	parser.add_argument("--max-points", type=int, default=4000, help="t-SNE 最多使用的点数（超出则下采样）")
	parser.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

	checkpoint_path = Path(args.checkpoint).expanduser()
	output_dir = Path(args.output_dir).expanduser() if args.output_dir else REPO_ROOT / "images" / checkpoint_path.parent.name
	output_dir.mkdir(parents=True, exist_ok=True)
	print(f"device={device} | checkpoint={checkpoint_path} | output_dir={output_dir}")

	model, checkpoint = load_atomaction_model(checkpoint_path, device)

	# 数据集配置优先用 checkpoint 内保存的 global_args，回退到配置文件。
	global_args = checkpoint.get("global_args")
	if global_args is None:
		with Path(args.global_config).expanduser().open("r", encoding="utf-8") as file:
			global_args = yaml.safe_load(file)

	dataloader = build_dataloader(global_args, batch_size=args.batch_size, num_workers=args.num_workers)
	stats = collect_statistics(model, dataloader, device, max_batches=args.max_batches)

	global_size = stats["global_codebook"].shape[0]
	detail_size = stats["detail_codebook"].shape[0]
	num_traj = stats["global_indices"].shape[0]
	print(f"collected {num_traj} trajectories | global K={global_size} | detail K={detail_size}")

	# 1. 使用频次直方图 + 利用率标量
	global_usage = plot_usage_histogram(stats["global_indices"], global_size, "global", output_dir / "usage_global.png")
	detail_usage = plot_usage_histogram(stats["detail_indices"], detail_size, "detail", output_dir / "usage_detail.png")
	print(f"[global] {global_usage}")
	print(f"[detail] {detail_usage}")

	# 2. 码字相似度矩阵
	plot_codebook_similarity(stats["global_codebook"], "global", output_dir / "similarity_global.png")
	plot_codebook_similarity(stats["detail_codebook"], "detail", output_dir / "similarity_detail.png")

	# 3. t-SNE / PCA 叠加图（global 按动作上色；detail 按码字 id 上色）
	plot_embedding_overlay(
		stats["global_features"], stats["global_indices"], stats["global_codebook"],
		"global", output_dir / "embedding_global.png", max_points=args.max_points, actions=stats["actions"],
	)
	plot_embedding_overlay(
		stats["detail_features"], stats["detail_indices"], stats["detail_codebook"],
		"detail", output_dir / "embedding_detail.png", max_points=args.max_points, actions=None,
	)

	# 4. 码字 × 动作类型热力图
	plot_code_action_heatmap(
		stats["global_indices"], stats["actions"], global_size, "global", output_dir / "code_action_global.png",
	)
	# 细节码本：每条轨迹 Q 个 slot，把动作标签按 slot 展开后统计。
	detail_actions = np.repeat(stats["actions"], stats["detail_indices_per_traj"].shape[1])
	plot_code_action_heatmap(
		stats["detail_indices"], detail_actions, detail_size, "detail", output_dir / "code_action_detail.png",
	)

	print(f"done. figures saved under {output_dir}")


if __name__ == "__main__":
	main()
