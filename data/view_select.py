from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

try:
    from .utils import build_dinov2_transform, get_configured_tensor_dtype
except ImportError:
    from utils import build_dinov2_transform, get_configured_tensor_dtype


SUPPORTED_DINOV2_MODELS = {
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
}

"""基于冻结 DINOv2 的视角信息量评估器。

输入：
    __init__:
        model_name: DINOv2 模型名称，必须属于 SUPPORTED_DINOV2_MODELS。
        repo: torch.hub 仓库地址，当前仅支持官方仓库 facebookresearch/dinov2。
        device: 推理设备，默认自动选择 CUDA 或 CPU。
        input_size: 输入图像尺寸，必须是 14 的倍数。

输出：
    select_best_view:
        返回按变化分数从高到低排序的列表，列表长度为 min(top_k, 可用视角数)。
        每个元素包含：
            best_view: str，视角名称。
            best_start_image: torch.Tensor，起始帧经过 transforms 处理后的图像张量，形状为 [3, H, W]。
                其中 H = W = input_size，默认情况下为 [3, 224, 224]。
            best_end_image: torch.Tensor，末帧经过 transforms 处理后的图像张量，形状为 [3, H, W]。
                其中 H = W = input_size，默认情况下为 [3, 224, 224]。
            best_score: torch.Tensor，0 维标量张量，dtype 由 config/utils.yaml 中的 tensor_dtype 控制。
                分数越大表示视觉变化越明显。
"""
class View_Selector:
    def __init__(
        self,
        model_name: str = "dinov2_vits14_reg",
        repo: str = "facebookresearch/dinov2",
        device: Optional[str] = None,
        input_size: int = 224,
    ) -> None:
        self.repo = repo
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.selection_model = None
        self.transform = None
        self.tensor_dtype = get_configured_tensor_dtype()
        self._validate_model_config()

    """在初始化阶段校验模型配置，当前仅接受官方 DINOv2 系列。"""
    def _validate_model_config(self) -> None:

        if self.repo != "facebookresearch/dinov2":
            raise ValueError(
                "View_Selector currently only supports the official DINOv2 hub repo: "
                "facebookresearch/dinov2"
            )
        
        # 目前仅支持 DINOv2 的 ViT 模型
        if self.model_name not in SUPPORTED_DINOV2_MODELS:
            supported_models = ", ".join(sorted(SUPPORTED_DINOV2_MODELS))
            raise ValueError(
                "Unsupported model_name for View_Selector. "
                f"Only DINOv2 models are supported. Got: {self.model_name}. "
                f"Supported models: {supported_models}"
            )

        # DINOv2 的 ViT 模型要求输入尺寸必须是 14 的倍数，且通常为 224
        if self.input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if self.input_size % 14 != 0:
            raise ValueError("input_size must be a multiple of 14 for DINOv2 models")

    """首次使用时加载官方权重，并永久冻结参数。"""
    def _ensure_model_loaded(self) -> None:
        if self.selection_model is not None and self.transform is not None:
            return

        try:
            self.selection_model = torch.hub.load(self.repo, self.model_name).to(self.device)
            self.selection_model.eval()
            for param in self.selection_model.parameters():
                param.requires_grad = False

            self.transform = build_dinov2_transform(self.input_size)
        except Exception as exc:
            self.selection_model = None
            self.transform = None
            raise RuntimeError(
                "Failed to load DINOv2 view selection model from torch.hub. "
                "Please check network access, local torch.hub cache, and model name settings. "
                f"repo={self.repo}, model={self.model_name}"
            ) from exc

    """校验图像路径是否存在，并返回规范化后的绝对路径。"""
    def _validate_image_path(self, img_path: str) -> Path:
        path = Path(img_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        return path

    """根据图像路径读取图片并执行 DINOv2 预处理。"""
    def _load_transformed_image(self, img_path: str) -> torch.Tensor:
        image_path = self._validate_image_path(img_path)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return self.transform(img).to(dtype=self.tensor_dtype)

    """校验多视角起止帧列表的长度和命名是否一致。"""
    def _validate_pairs(
        self,
        start_paths: Sequence[str],
        end_paths: Sequence[str],
        view_names: Optional[Sequence[str]] = None,
    ) -> None:
        if len(start_paths) == 0:
            raise ValueError("视角列表不能为空")
        if len(start_paths) != len(end_paths):
            raise ValueError("start_paths 与 end_paths 长度必须一致")
        if view_names is not None and len(view_names) != len(start_paths):
            raise ValueError("view_names 长度必须与图像路径数量一致")

    """提取图像的归一化全局语义特征，输出形状为 [1, D]。"""
    def get_feature(self, img_path: str) -> torch.Tensor:
        self._ensure_model_loaded()
        image_path = self._validate_image_path(img_path)

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).to(self.device)
            features = self.selection_model.forward_features(x)
            if isinstance(features, dict) and "x_norm_clstoken" in features:
                feat = features["x_norm_clstoken"]
            else:
                feat = self.selection_model(x)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat

    """计算起始帧与末帧的变化分数，分数越大表示视觉变化越明显。"""
    def compute_change_score(self, start_path: str, end_path: str) -> float:
        f1 = self.get_feature(start_path)
        f2 = self.get_feature(end_path)
        sim = torch.cosine_similarity(f1, f2, dim=-1).item()
        return float(1.0 - sim)

    """批量计算多个视角对应起止帧的变化分数。"""
    def compute_change_scores(
        self,
        start_paths: Sequence[str],
        end_paths: Sequence[str],
    ) -> List[float]:
        self._validate_pairs(start_paths, end_paths)

        scores: List[float] = []
        for start_path, end_path in zip(start_paths, end_paths):
            scores.append(self.compute_change_score(start_path, end_path))
        return scores

    """从多视角中选择变化分数最高的视角，并返回详细结果。"""
    def select_best_view(
        self,
        start_paths: Sequence[str],
        end_paths: Sequence[str],
        view_names: Optional[Sequence[str]] = None,
        top_k: int = 1,
    ) -> List[Dict[str, object]]:
        
        # 基础检查
        self._validate_pairs(start_paths, end_paths, view_names)
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # 加载模型并计算分数
        self._ensure_model_loaded()
        scores = self.compute_change_scores(start_paths, end_paths)

        # 打包结果
        resolved_view_names = list(view_names) if view_names is not None else [None] * len(scores)
        scored_views: List[Dict[str, object]] = []
        for index, (view_name, start_path, end_path, score) in enumerate(
            zip(resolved_view_names, start_paths, end_paths, scores)
        ):
            scored_views.append(
                {
                    "index": index,
                    "view_name": view_name,
                    "start_path": start_path,
                    "end_path": end_path,
                    "score": score,
                }
            )

        sorted_views = sorted(scored_views, key=lambda item: item["score"], reverse=True)
        selected_count = min(top_k, len(sorted_views))
        selected_views: List[Dict[str, object]] = []
        for selected_view in sorted_views[:selected_count]:
            selected_views.append(
                {
                    "best_view": selected_view["view_name"],
                    "best_start_image": self._load_transformed_image(selected_view["start_path"]),
                    "best_end_image": self._load_transformed_image(selected_view["end_path"]),
                    "best_score": torch.tensor(selected_view["score"], dtype=self.tensor_dtype)
                }
            )

        return selected_views


ViewSelector = View_Selector


# 简单测试
if __name__ == "__main__":
    selector = View_Selector()
    start_paths = ["Action_Primitive_Dataset_v0/approach/phase_0001/front_rgb/0.png", 
                   "Action_Primitive_Dataset_v0/approach/phase_0001/left_shoulder_rgb/0.png",
                   "Action_Primitive_Dataset_v0/approach/phase_0001/right_shoulder_rgb/0.png",
                   "Action_Primitive_Dataset_v0/approach/phase_0001/overhead_rgb/0.png",
                   "Action_Primitive_Dataset_v0/approach/phase_0001/wrist_rgb/0.png"]
    end_paths = ["Action_Primitive_Dataset_v0/approach/phase_0001/front_rgb/60.png", 
                 "Action_Primitive_Dataset_v0/approach/phase_0001/left_shoulder_rgb/60.png",
                 "Action_Primitive_Dataset_v0/approach/phase_0001/right_shoulder_rgb/60.png",
                 "Action_Primitive_Dataset_v0/approach/phase_0001/overhead_rgb/60.png",
                 "Action_Primitive_Dataset_v0/approach/phase_0001/wrist_rgb/60.png"]
    view_names = ["front", "left_shoulder", "right_shoulder", "overhead", "wrist"]

    try:
        result = selector.select_best_view(start_paths, end_paths, view_names)
        print("Best view selection result:")
        print(result)
    except Exception as exc:
        print(f"Error during view selection: {exc}")