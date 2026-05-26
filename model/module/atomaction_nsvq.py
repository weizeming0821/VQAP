from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

try:
    from .utils import *
except ImportError:
    from utils import *


"""AtomAction_NSVQ 前端动作编码模块。

输入：
    __init__:
        model_args: Dict[str, Any]，模型配置字典，可直接传入完整 model.yaml 或其中的 AtomAction_NSVQ 子配置。
    forward:
        trajectory_data: Dict[str, torch.Tensor]
            gripper_pose: [B, T, 9]
            joint_positions: [B, T, 7]
            joint_velocities: [B, T, 7]
            joint_forces: [B, T, 7]
            gripper_joint_positions: [B, T, 2]
            gripper_touch_forces: [B, T, 6]
            gripper_open: [B, T, 1]

输出：
    forward:
        trajectory_tokens: [B, T, 512]
"""
class AtomAction_NSVQ(nn.Module):
    EE_INPUT_DIM = 9
    BODY_INPUT_DIM = 21
    GRIPPER_MECH_INPUT_DIM = 8
    GRIPPER_OPEN_NUM_CLASSES = 2
    TARGET_TOKEN_DIM = 512

    """初始化四组动作特征投影层和 RoPE 模块。"""
    def __init__(self, model_args: Dict[str, Any]):
        super(AtomAction_NSVQ, self).__init__()

        self.model_args = self._resolve_model_args(model_args)

        ee_branch = self._get_config_section("ee_branch")
        body_branch = self._get_config_section("body_branch")
        gripper_mech_branch = self._get_config_section("gripper_mech_branch")
        gripper_open_branch = self._get_config_section("gripper_open_branch")
        rope_config = self._get_config_section("rope")

        self.ee_projector = TrajectoryProjectionMLP(
            input_dim=self.EE_INPUT_DIM,
            hidden_dim=int(ee_branch["hidden_dim"]),
            output_dim=int(ee_branch["output_dim"]),
        )
        self.body_projector = TrajectoryProjectionMLP(
            input_dim=self.BODY_INPUT_DIM,
            hidden_dim=int(body_branch["hidden_dim"]),
            output_dim=int(body_branch["output_dim"]),
        )
        self.gripper_mech_projector = TrajectoryProjectionMLP(
            input_dim=self.GRIPPER_MECH_INPUT_DIM,
            hidden_dim=int(gripper_mech_branch["hidden_dim"]),
            output_dim=int(gripper_mech_branch["output_dim"]),
        )

        gripper_open_output_dim = int(gripper_open_branch["output_dim"])
        self.gripper_open_embedding = nn.Embedding(
            num_embeddings=self.GRIPPER_OPEN_NUM_CLASSES,
            embedding_dim=gripper_open_output_dim,
        )
        self.gripper_open_projector = nn.Linear(gripper_open_output_dim, gripper_open_output_dim)

        self.token_dim = (
            int(ee_branch["output_dim"])
            + int(body_branch["output_dim"])
            + int(gripper_mech_branch["output_dim"])
            + gripper_open_output_dim
        )
        if self.token_dim != self.TARGET_TOKEN_DIM:
            raise ValueError(
                f"Projected token dim must be {self.TARGET_TOKEN_DIM}, got {self.token_dim}"
            )

        self.rope = RotaryPositionEncoding1D(
            feature_dim=self.token_dim,
            theta=float(rope_config["theta"]),
            max_seq_len=int(rope_config["max_seq_len"]),
        )

    """解析模型配置，兼容完整 model.yaml 和局部配置两种输入。"""
    @staticmethod
    def _resolve_model_args(model_args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(model_args, dict):
            raise ValueError("model_args must be a dictionary")
        if "AtomAction_NSVQ" in model_args:
            section = model_args["AtomAction_NSVQ"]
            if not isinstance(section, dict):
                raise ValueError("AtomAction_NSVQ config must be a mapping")
            return section
        return model_args

    """读取指定配置分支。"""
    def _get_config_section(self, section_name: str) -> Dict[str, Any]:
        section = self.model_args.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Missing config section: {section_name}")
        return section

    """检查轨迹字段是否存在且维度正确。"""
    @staticmethod
    def _require_field(
        trajectory_data: Dict[str, torch.Tensor],
        field_name: str,
        expected_last_dim: int,
    ) -> torch.Tensor:
        field = trajectory_data.get(field_name)
        if field is None or not torch.is_tensor(field):
            raise ValueError(f"trajectory_data[{field_name}] must be a tensor")
        if field.ndim != 3:
            raise ValueError(f"trajectory_data[{field_name}] must have shape [B, T, D]")
        if field.shape[-1] != expected_last_dim:
            raise ValueError(
                f"trajectory_data[{field_name}] last dim must be {expected_last_dim}, got {field.shape[-1]}"
            )
        return field

    """将轨迹字段映射为动作 token。

    输入：
        trajectory_data: Dict[str, torch.Tensor]
            gripper_pose: [B, T, 9]
            joint_positions: [B, T, 7]
            joint_velocities: [B, T, 7]
            joint_forces: [B, T, 7]
            gripper_joint_positions: [B, T, 2]
            gripper_touch_forces: [B, T, 6]
            gripper_open: [B, T, 1]

    输出：
        trajectory_tokens: [B, T, 512]
    """
    def forward(self, trajectory_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        gripper_pose = self._require_field(trajectory_data, "gripper_pose", self.EE_INPUT_DIM)
        joint_positions = self._require_field(trajectory_data, "joint_positions", 7)
        joint_velocities = self._require_field(trajectory_data, "joint_velocities", 7)
        joint_forces = self._require_field(trajectory_data, "joint_forces", 7)
        gripper_joint_positions = self._require_field(trajectory_data, "gripper_joint_positions", 2)
        gripper_touch_forces = self._require_field(trajectory_data, "gripper_touch_forces", 6)
        gripper_open = self._require_field(trajectory_data, "gripper_open", 1)

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
        trajectory_tokens = torch.cat(
            (ee_features, body_features, gripper_mech_features, gripper_open_features),
            dim=-1,
        )
        return trajectory_tokens

    """返回后续注意力层可直接使用的 RoPE cos/sin 缓存。"""
    def get_rope_cache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        return self.rope(seq_len=seq_len, device=device, dtype=dtype)

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

    """按模块名冻结参数；未指定时冻结整个模块。"""
    def freeze_modules(self, freeze_module=None):
        if freeze_module is None:
            modules: Iterable[nn.Module] = [self]
        else:
            module_names = [freeze_module] if isinstance(freeze_module, str) else list(freeze_module)
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
