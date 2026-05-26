from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from .encoder import ChannelEncoder, TransformerEncoder
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
            gripper_pose: [B, T, 9]
            joint_positions: [B, T, 7]
            joint_velocities: [B, T, 7]
            joint_forces: [B, T, 7]
            gripper_joint_positions: [B, T, 2]
            gripper_touch_forces: [B, T, 6]
            gripper_open: [B, T, 1]

输出：
    forward:
        encoded_trajectory_tokens: [B, T, 512]
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
        )

    def forward(self, trajectory_data: Dict[str, torch.Tensor], trajectory_mask: torch.Tensor) -> torch.Tensor:

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
        return encoded_trajectory_features


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
