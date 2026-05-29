from typing import Any, Dict

import torch
import torch.nn as nn

from .encoder import ChannelEncoder, TransformerEncoder
from .nsvq import DetailCodebookModule, GlobalCodebookModule
from .utils import TrajectoryProjectionMLP


# 全局常量定义，各动作维度的输入维度
EE_INPUT_DIM = 9
BODY_INPUT_DIM = 21
GRIPPER_MECH_INPUT_DIM = 8
GRIPPER_OPEN_NUM_CLASSES = 2

"""AtomAction_NSVQ 

输入：
    __init__:
        model_args: Dict[str, Any]
    forward:
        trajectory_data: Dict[str, torch.Tensor]
        trajectory_mask: [B, T]，True 表示有效帧。

输出：
    forward:
        global_feature: [B, D]
        global_codeword: [B, D]
        global_codeindex: [B]
        global_perplexity: 标量
        detail_features: [B, N_detail, D]
        detail_codewords: [B, N_detail, D]
        detail_codeindices: [B, N_detail]
        detail_perplexity: 标量
"""
class AtomAction_NSVQ(nn.Module):

    def __init__(self, model_args: Dict[str, Any]):
        super(AtomAction_NSVQ, self).__init__()

        # 读取配置参数
        self.model_args = model_args["AtomAction_NSVQ"] if "AtomAction_NSVQ" in model_args else model_args

        ee_cfg = self.model_args["ee_branch"]
        body_cfg = self.model_args["body_branch"]
        gripper_mech_cfg = self.model_args["gripper_mech_branch"]
        gripper_open_cfg = self.model_args["gripper_open_branch"]
        channel_encoder_cfg = self.model_args["channel_encoder"]
        rope_cfg = self.model_args["rope"]
        encoder_cfg = self.model_args["transformer_encoder"]
        global_codebook_cfg = self.model_args["global_codebook"]
        detail_codebook_cfg = self.model_args["detail_codebook"]
        nsvq_cfg = self.model_args["nsvq"]

        # 初始化各分支的投影模块
        self.ee_projector = TrajectoryProjectionMLP(
            input_dim=EE_INPUT_DIM,
            hidden_dim=int(ee_cfg["hidden_dim"]),
            output_dim=int(ee_cfg["output_dim"]),
        )
        self.body_projector = TrajectoryProjectionMLP(
            input_dim=BODY_INPUT_DIM,
            hidden_dim=int(body_cfg["hidden_dim"]),
            output_dim=int(body_cfg["output_dim"]),
        )
        self.gripper_mech_projector = TrajectoryProjectionMLP(
            input_dim=GRIPPER_MECH_INPUT_DIM,
            hidden_dim=int(gripper_mech_cfg["hidden_dim"]),
            output_dim=int(gripper_mech_cfg["output_dim"]),
        )
        # 初始化二值夹爪开关的嵌入和投影模块
        gripper_open_output_dim = int(gripper_open_cfg["output_dim"])
        self.gripper_open_embedding = nn.Embedding(
            num_embeddings=GRIPPER_OPEN_NUM_CLASSES,
            embedding_dim=gripper_open_output_dim,
        )
        self.gripper_open_projector = nn.Linear(gripper_open_output_dim, gripper_open_output_dim)

        # Channel Encoder：[B, T, 512] -> [B, T, 512]
        self.channel_encoder = ChannelEncoder(
            hidden_dim=int(encoder_cfg["hidden_dim"]),
            bottleneck_dim=int(channel_encoder_cfg["bottleneck_dim"]),
        )

        # Transformer Encoder：[B, T, 512] -> [B, T, 512]
        self.transformer_encoder = TransformerEncoder(
            hidden_dim=int(encoder_cfg["hidden_dim"]),
            num_layers=int(encoder_cfg["num_layers"]),
            num_heads=int(encoder_cfg["num_heads"]),
            ffn_dim=int(encoder_cfg["ffn_dim"]),
            dropout=float(encoder_cfg["dropout"]),
            rope_theta=float(rope_cfg["theta"]),
            rope_max_seq_len=int(rope_cfg["max_seq_len"]),
            norm_type=str(encoder_cfg.get("norm_type", "layernorm")),
        )

        # 全局语义码本分支：[B, T, 512] + [B, T] -> [B, 256]
        self.global_codebook_module = GlobalCodebookModule(
            hidden_dim=int(encoder_cfg["hidden_dim"]),
            codebook_dim=int(global_codebook_cfg["codebook_dim"]),
            codebook_size=int(global_codebook_cfg["codebook_size"]),
            replace_every=int(nsvq_cfg["replace_every"]),
            discard_threshold=float(nsvq_cfg["discard_threshold"]),
            replace_noise_scale=float(nsvq_cfg["replace_noise_scale"]),
            eps=float(nsvq_cfg["eps"]),
        )

        # 细节语义码本分支：[B, T, 512] + [B, T] -> [B, N_detail, 256]
        self.detail_codebook_module = DetailCodebookModule(
            hidden_dim=int(encoder_cfg["hidden_dim"]),
            num_queries=int(detail_codebook_cfg["num_queries"]),
            num_heads=int(detail_codebook_cfg["num_heads"]),
            codebook_dim=int(detail_codebook_cfg["codebook_dim"]),
            codebook_size=int(detail_codebook_cfg["codebook_size"]),
            dropout=float(detail_codebook_cfg["dropout"]),
            rope_theta=float(rope_cfg["theta"]),
            rope_max_seq_len=int(rope_cfg["max_seq_len"]),
            replace_every=int(nsvq_cfg["replace_every"]),
            discard_threshold=float(nsvq_cfg["discard_threshold"]),
            replace_noise_scale=float(nsvq_cfg["replace_noise_scale"]),
            eps=float(nsvq_cfg["eps"]),
        )

    def forward(
        self,
        trajectory_data: Dict[str, torch.Tensor],
        trajectory_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # 读取各动作维度数据
        gripper_pose = trajectory_data["gripper_pose"]
        joint_positions = trajectory_data["joint_positions"]
        joint_velocities = trajectory_data["joint_velocities"]
        joint_forces = trajectory_data["joint_forces"]
        gripper_joint_positions = trajectory_data["gripper_joint_positions"]
        gripper_touch_forces = trajectory_data["gripper_touch_forces"]
        gripper_open = trajectory_data["gripper_open"]

        # 末端执行器分支：[B, T, 9] -> [B, T, 192]
        ee_features = self.ee_projector(gripper_pose)

        # 关节体态分支：([B, T, 7] * 3) -> [B, T, 21] -> [B, T, 192]
        body_inputs = torch.cat((joint_positions, joint_velocities, joint_forces), dim=-1)
        body_features = self.body_projector(body_inputs)

        # 夹爪机械量分支：[B, T, 2] + [B, T, 6] -> [B, T, 8] -> [B, T, 64]
        gripper_mech_inputs = torch.cat((gripper_joint_positions, gripper_touch_forces), dim=-1)
        gripper_mech_features = self.gripper_mech_projector(gripper_mech_inputs)

        # 夹爪开合分支：[B, T, 1] -> [B, T] -> [B, T, 64] -> [B, T, 64]
        gripper_open_ids = gripper_open.squeeze(-1).round().clamp(0, 1).long()
        gripper_open_features = self.gripper_open_projector(self.gripper_open_embedding(gripper_open_ids))

        # 四组特征拼接：[B, T, 192+192+64+64] -> [B, T, 512]
        trajectory_features = torch.cat(
            (ee_features, body_features, gripper_mech_features, gripper_open_features),
            dim=-1,
        )

        # Channel Encoder：[B, T, 512] -> [B, T, 512]
        channel_encoded_features = self.channel_encoder(trajectory_features, trajectory_mask)

        # Transformer Encoder：[B, T, 512] -> [B, T, 512]
        encoded_trajectory_features = self.transformer_encoder(channel_encoded_features, trajectory_mask)

        # 全局语义码本分支
        global_codebook_outputs = self.global_codebook_module(
            encoded_trajectory_features,
            trajectory_mask,
        )

        # 细节语义码本分支
        detail_codebook_outputs = self.detail_codebook_module(
            encoded_trajectory_features,
            trajectory_mask,
        )

        global_feature = global_codebook_outputs["global_feature"]          # [ B, D ]
        global_codeword = global_codebook_outputs["global_codeword"]        # [ B, D ]
        global_codeindex = global_codebook_outputs["global_codeindex"]      # [ B ]
        global_perplexity = global_codebook_outputs["global_perplexity"]    # 标量

        detail_features = detail_codebook_outputs["detail_features"]        # [B, N_detail, D]
        detail_codewords = detail_codebook_outputs["detail_codewords"]      # [B, N_detail, D]
        detail_codeindices = detail_codebook_outputs["detail_codeindexs"]   # [B, N_detail]
        detail_perplexity = detail_codebook_outputs["detail_perplexity"]    # 标量

        model_outputs = {
            "global_feature": global_feature,
            "global_codeword": global_codeword,
            "global_codeindex": global_codeindex,
            "global_perplexity": global_perplexity,
            "detail_features": detail_features,
            "detail_codewords": detail_codewords,
            "detail_codeindices": detail_codeindices,
            "detail_perplexity": detail_perplexity,
        }
        return model_outputs


    """计算模块参数量。"""
    def print_module_params(self):
        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        print(f"AtomAction_NSVQ total params: {total_params}")
        print(f"AtomAction_NSVQ trainable params: {trainable_params}")
        return {
            "total": total_params,
            "trainable": trainable_params,
        }


    """显式触发全局码本与细节码本的死码替换。"""
    @torch.no_grad()
    def replace_unused_codebooks(self):
        return {
            "replaced_codebooks_g": self.global_codebook_module.replace_unused_codebooks(),
            "replaced_codebooks_d": self.detail_codebook_module.replace_unused_codebooks(),
        }


    """按模块名冻结参数；指定 All 时冻结整个模块。"""
    def freeze_modules(self, freeze_module=None):
        # 不冻结任何模块
        if freeze_module is None:
            return  
        else:
            module_names = [freeze_module] if isinstance(freeze_module, str) else list(freeze_module)
            if "All" in module_names:
                modules = [self]
            else:
                resolved_modules = []
                for module_name in module_names:
                    module = getattr(self, module_name, None)
                    if module is None or not isinstance(module, nn.Module):
                        raise ValueError(f"Unknown module name: {module_name}")
                    resolved_modules.append(module)
                modules = resolved_modules

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
