import cv2
import torch
import numpy as np
from PIL import Image

# ----------------------
# 1. 加载 DINOv2 模型
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
dinov2.eval()

# 图像预处理（DINOv2 官方标准）
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    # DINOv2 不需要强行 resize 到很小，但保持长边统一更稳定
    transform = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_transform')
    return transform(img).unsqueeze(0).to(device)

# ----------------------
# 2. 计算两帧变化分数
# 分数越大 = 画面变化越大 = 视角越好
# ----------------------
def compute_difference_score(img_start_path, img_end_path):
    with torch.no_grad():
        feat_start = dinov2(preprocess(img_start_path))  # 提取初始帧特征
        feat_end = dinov2(preprocess(img_end_path))     # 提取结束帧特征

        # 归一化特征（DINOv2 特征必须归一化）
        feat_start = torch.nn.functional.normalize(feat_start, dim=-1)
        feat_end = torch.nn.functional.normalize(feat_end, dim=-1)

        # 余弦距离 = 1 - 余弦相似度
        cos_sim = (feat_start @ feat_end.T).item()
        score = 1 - cos_sim

    return score

# ----------------------
# 3. 输入多视角，自动选最优视角
# ----------------------
def select_best_view(view_dict):
    """
    view_dict = {
        "view_1": ("start1.jpg", "end1.jpg"),
        "view_2": ("start2.jpg", "end2.jpg"),
        ...
    }
    """
    scores = {}
    for view_name, (start, end) in view_dict.items():
        s = compute_difference_score(start, end)
        scores[view_name] = s
        print(f"{view_name} 变化分数: {s:.4f}")

    # 选分数最大的视角
    best_view = max(scores, key=scores.get)
    best_score = scores[best_view]
    return best_view, best_score, scores

# ----------------------
# 4. 使用示例
# ----------------------
if __name__ == "__main__":
    # 你只需要改这里！填入你的多视角始末帧
    multi_views = {
        "view_front": ("view1_start.png", "view1_end.png"),
        "view_side": ("view2_start.png", "view2_end.png"),
        "view_top": ("view3_start.png", "view3_end.png"),
        "view_back": ("view4_start.png", "view4_end.png"),  # 遮挡大 → 分数低
    }

    best_view, best_score, all_scores = select_best_view(multi_views)

    print("\n==== 结果 ====")
    print(f"最优视角: {best_view}")
    print(f"最大变化分数: {best_score:.4f}")