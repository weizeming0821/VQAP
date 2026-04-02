from typing import Dict, List, Sequence

import torch
from PIL import Image

class View_Selector:
    """基于官方冻结 DINOv2 的视角信息量评估器。

    设计目标：
    1. 用于视角选择的 DINOv2 永久冻结，仅参与特征提取。
    2. 与后续微调用 DINOv2 隔离，避免共享同一个模型实例。
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        repo: str = "facebookresearch/dinov2",
        device: str = None,
    ):
        self.repo = repo
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 视角选择专用模型：始终加载官方权重并永久冻结。
        self.selection_model = torch.hub.load(self.repo, self.model_name).to(self.device)
        self.selection_model.eval()
        for param in self.selection_model.parameters():
            param.requires_grad = False

        # 与模型名称绑定的官方预处理。
        self.transform = torch.hub.load(self.repo, f"{self.model_name}_transform")

    def build_finetune_model(self) -> torch.nn.Module:
        """返回一个新的 DINOv2 实例用于微调。

        注意：该返回值与 self.selection_model 不是同一个对象，
        可以安全设置为 train 模式并打开梯度。
        """
        finetune_model = torch.hub.load(self.repo, self.model_name).to(self.device)
        finetune_model.train()
        for param in finetune_model.parameters():
            param.requires_grad = True
        return finetune_model

    def get_feature(self, img_path: str) -> torch.Tensor:
        """提取图像归一化特征，输出 shape 为 [1, D]。"""
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).to(self.device)
            feat = self.selection_model(x)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat

    def compute_change_score(self, start_path: str, end_path: str) -> float:
        """计算起始帧与末帧变化分数：score = 1 - cosine_similarity。

        分数越大表示视觉变化越明显，通常携带更多任务信息。
        """
        f1 = self.get_feature(start_path)
        f2 = self.get_feature(end_path)
        sim = torch.cosine_similarity(f1, f2, dim=-1).item()
        return float(1.0 - sim)

    def compute_change_scores(
        self,
        start_paths: Sequence[str],
        end_paths: Sequence[str],
    ) -> List[float]:
        """批量计算多个视角的变化分数。"""
        if len(start_paths) != len(end_paths):
            raise ValueError("start_paths 与 end_paths 长度必须一致")

        scores: List[float] = []
        for start_path, end_path in zip(start_paths, end_paths):
            scores.append(self.compute_change_score(start_path, end_path))
        return scores

    def select_best_view(
        self,
        start_paths: Sequence[str],
        end_paths: Sequence[str],
    ) -> Dict[str, object]:
        """从多视角中选择变化最大的视角。

        返回字段：
        - best_index: 最优视角下标
        - best_start_path / best_end_path: 最优视角的起始帧和末帧路径
        - best_score: 最优变化分数
        - all_scores: 全部视角分数
        """
        scores = self.compute_change_scores(start_paths, end_paths)
        if len(scores) == 0:
            raise ValueError("视角列表不能为空")

        best_index = max(range(len(scores)), key=lambda i: scores[i])
        return {
            "best_index": best_index,
            "best_start_path": start_paths[best_index],
            "best_end_path": end_paths[best_index],
            "best_score": scores[best_index],
            "all_scores": scores,
        }


ViewSelector = View_Selector

