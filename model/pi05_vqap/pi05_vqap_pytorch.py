"""VQAP 条件化的 pi0.5 PyTorch 独立模型实现。

本文件刻意复制/封装 pi0.5 PyTorch 路径，而不是直接修改 openpi 原始
PI0Pytorch 类，从而避免影响已有 RLBench 微调流程和初始 pi0.5 模型。
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

from .adapter import VQAPAdapter, VQAPAdapterOutput
from .codebook import FrozenVQAPCodebook
from .config import PI05VQAPConfig


# 根据目标设备选择安全 dtype，避免 CPU 上的 bfloat16 等类型引发兼容问题。
def get_safe_dtype(target_dtype, device_type):
    """返回当前设备上可安全使用的 dtype。"""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


# 为 flow-matching 时间步生成正余弦位置编码。
def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """根据标量时间步计算 sine-cosine 位置编码向量。"""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    # 按周期范围构造不同频率，使时间条件覆盖从细粒度到粗粒度的变化。
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


# 从 Beta 分布采样训练时间，保持与原 pi0.5 flow-matching 训练策略一致。
def sample_beta(alpha, beta, bsize, device):
    """在指定设备上按 batch 大小采样 Beta 分布。"""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


# 根据 padding mask 和自回归 block 标记构造二维注意力 mask。
def make_att_2d_masks(pad_masks, att_masks):
    """由 padding 与 block 标记生成模型 forward 所需的二维注意力 mask。"""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # att_masks 的累积和定义可见 block；padding mask 再屏蔽无效 token。
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# 独立的 pi0.5 + VQAP 条件模型，专供 VQAP 实验使用，不改 openpi 原模型类。
class PI05VQAPPytorch(nn.Module):
    """带冻结 VQAP 码本适配器的 pi0.5 动作模型。"""

    # 初始化 pi0.5 主干、冻结 VQAP 码本、码本适配器和条件注入层。
    def __init__(self, config: PI05VQAPConfig):
        super().__init__()
        if not getattr(config, "pi05", False):
            raise ValueError("PI05VQAPPytorch only supports pi0.5 configs.")
        self.config = config
        self.pi05 = True
        self.latest_vqap_metrics: dict[str, Tensor] = {}

        # 读取 pi0.5 的 PaliGemma 与 action expert 配置，保留原模型宽度设定。
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.prefix_width = int(paligemma_config.width)
        self.action_expert_width = int(action_expert_config.width)

        # 使用 pi0.5 的双塔结构：视觉/语言前缀塔 + 动作 expert 塔。
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, self.action_expert_width)
        self.action_out_proj = nn.Linear(self.action_expert_width, config.action_dim)
        self.time_mlp_in = nn.Linear(self.action_expert_width, self.action_expert_width)
        self.time_mlp_out = nn.Linear(self.action_expert_width, self.action_expert_width)

        # VQAP 码本保持冻结；adapter 只预测离散 code 的选择分布和软/硬 code 向量。
        self.vqap_codebook = FrozenVQAPCodebook(config.vqap_codebook_path)
        self.vqap_adapter = VQAPAdapter(
            self.vqap_codebook,
            prefix_width=self.prefix_width,
            hidden_dim=self.vqap_codebook.code_dim,
            tau=config.vqap_tau,
            load_loss_weight=config.vqap_load_loss_weight,
        )
        # global code 注入 action expert 的 adaRMS 条件；detail code 作为额外 prefix token。
        self.code_global_proj = nn.Linear(self.vqap_codebook.code_dim, self.action_expert_width)
        self.code_detail_proj = nn.Linear(self.vqap_codebook.code_dim, self.prefix_width)
        self.ada_cond_mlp = nn.Linear(self.action_expert_width * 2, self.action_expert_width)
        self._init_code_injection_layers()

        # 保留 openpi PyTorch 路径中的 matmul 精度与可选 compile 设置。
        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        self.gradient_checkpointing_enabled = False

        # 检查 transformers_replace 是否安装到位；这是 openpi PyTorch 模型的运行前置条件。
        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    # 将 VQAP 注入层置零，保证模型初始行为尽量等价于原 pi0.5。
    def _init_code_injection_layers(self) -> None:
        # 残差融合分支初始输出为 0，因此不会一开始破坏原 time condition。
        nn.init.zeros_(self.code_global_proj.weight)
        nn.init.zeros_(self.code_global_proj.bias)
        nn.init.zeros_(self.code_detail_proj.weight)
        nn.init.zeros_(self.code_detail_proj.bias)
        nn.init.zeros_(self.ada_cond_mlp.weight)
        nn.init.zeros_(self.ada_cond_mlp.bias)

    # 开启梯度检查点以降低显存占用，适合大 batch 或高分辨率图像训练。
    def gradient_checkpointing_enable(self):
        """开启梯度检查点。"""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05VQAPPytorch model")

    # 关闭梯度检查点，恢复常规 forward/backward。
    def gradient_checkpointing_disable(self):
        """关闭梯度检查点。"""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05VQAPPytorch model")

    # 查询当前是否启用梯度检查点。
    def is_gradient_checkpointing_enabled(self):
        """返回梯度检查点开关状态。"""
        return self.gradient_checkpointing_enabled

    # 对指定函数按需包一层 checkpoint，统一管理显存优化逻辑。
    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
                **kwargs,
            )
        return func(*args, **kwargs)

    # 将二维 bool attention mask 转成 transformer 使用的四维加性 mask。
    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    # 复用 openpi 的图像、语言和状态预处理，保持与 pi0.5 数据路径一致。
    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    # 采样 flow-matching 起点噪声。
    def sample_noise(self, shape, device):
        """生成标准高斯噪声作为 action flow 的噪声端点。"""
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    # 采样 flow-matching 时间步。
    def sample_time(self, bsize, device):
        """按 pi0.5 训练策略采样位于 (0, 1] 的时间步。"""
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    # 编码视觉图像与语言 token，形成 pi0.5 prefix 序列。
    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """将多视角图像和语言 token 编码为 pi0.5 prefix embedding。"""
        embs = []
        pad_masks = []
        att_masks = []

        # 逐路图像进入 vision tower，并把有效图像 mask 扩展到对应的 patch token。
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # 语言 token 进入 language embedding，并按 hidden size 做尺度校正。
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    # 追加 VQAP index token，用于从 prefix 表示中预测 global/detail 离散 code。
    def _append_index_tokens(self, prefix_embs: Tensor, pad_masks: Tensor, att_masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bsize = prefix_embs.shape[0]
        index_tokens = self.vqap_adapter.expanded_index_tokens(
            bsize,
            device=prefix_embs.device,
            dtype=prefix_embs.dtype,
        )
        index_pad = torch.ones(bsize, self.vqap_adapter.num_index_tokens, dtype=torch.bool, device=prefix_embs.device)
        # global token 开启 block 1；首个 detail token 开启 block 2；其余 detail 共享 block 2。
        index_att = torch.tensor([1, 1, *([0] * 8)], dtype=att_masks.dtype, device=prefix_embs.device)
        index_att = index_att[None, :].expand(bsize, -1)
        return (
            torch.cat([prefix_embs, index_tokens], dim=1),
            torch.cat([pad_masks, index_pad], dim=1),
            torch.cat([att_masks, index_att], dim=1),
        )

    # 将 VQAP detail code 投影成 prefix token，追加到 prefix 中供动作 expert 交叉关注。
    def _append_detail_code_tokens(
        self,
        prefix_embs: Tensor,
        pad_masks: Tensor,
        att_masks: Tensor,
        z_detail: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsize = prefix_embs.shape[0]
        # 每个 detail code 对应一个 token，维度从 code_dim 投影到 PaliGemma prefix width。
        detail_tokens = self.code_detail_proj(z_detail.float()).to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        detail_pad = torch.ones(bsize, self.vqap_codebook.num_detail_tokens, dtype=torch.bool, device=prefix_embs.device)
        detail_att = torch.tensor([1, *([0] * 8)], dtype=att_masks.dtype, device=prefix_embs.device)
        detail_att = detail_att[None, :].expand(bsize, -1)
        return (
            torch.cat([prefix_embs, detail_tokens], dim=1),
            torch.cat([pad_masks, detail_pad], dim=1),
            torch.cat([att_masks, detail_att], dim=1),
        )

    # 只运行 prefix 侧 transformer，用于产生 index token 表示或推理 cache。
    def _run_prefix_only(
        self,
        prefix_embs: Tensor,
        pad_masks: Tensor,
        att_masks: Tensor,
        *,
        use_cache: bool,
    ):
        # prefix 内部使用 block attention；推理时可缓存 prefix 的 KV，减少 denoise 重复计算。
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def prefix_func(prefix_embs, att_2d_masks_4d, position_ids):
            (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=use_cache,
            )
            return prefix_out, past_key_values

        if use_cache:
            return prefix_func(prefix_embs, att_2d_masks_4d, position_ids)
        return self._apply_checkpoint(prefix_func, prefix_embs, att_2d_masks_4d, position_ids)

    # 构造带 VQAP 条件的最终 prefix：原 prefix -> index token -> code 选择 -> detail code token。
    def _prepare_vqap_prefix(
        self,
        prefix_embs: Tensor,
        pad_masks: Tensor,
        att_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, VQAPAdapterOutput]:
        # 先追加 index token，并通过 prefix-only forward 得到 index token 的上下文表示。
        prefix_with_index, index_pad_masks, index_att_masks = self._append_index_tokens(prefix_embs, pad_masks, att_masks)
        prefix_out, _ = self._run_prefix_only(prefix_with_index, index_pad_masks, index_att_masks, use_cache=False)
        index_hidden = prefix_out[:, -self.vqap_adapter.num_index_tokens :]
        # adapter 根据 index hidden 预测 VQAP global/detail code，并返回辅助统计。
        adapter_output = self.vqap_adapter(
            index_hidden,
            tau=self.config.vqap_tau,
            code_dropout_p=self.config.vqap_code_dropout_p if self.training else 0.0,
            code_mode=self.config.vqap_code_mode,
        )
        # detail code 以 token 形式拼回 prefix；global code 稍后注入 action expert 的 adaRMS 条件。
        final_prefix, final_pad_masks, final_att_masks = self._append_detail_code_tokens(
            prefix_with_index,
            index_pad_masks,
            index_att_masks,
            adapter_output.z_detail,
        )
        return final_prefix, final_pad_masks, final_att_masks, adapter_output

    # 编码 noisy action 与时间步，形成 action expert 的 suffix 输入和 adaRMS 条件。
    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor, z_global: Tensor | None):
        """将 noisy action 与时间步编码为 pi0.5 action expert 的 suffix。"""
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # action 序列投影到 action expert hidden width。
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # 时间编码经过 MLP 后作为原 pi0.5 的 adaRMS 条件。
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_cond = self._apply_checkpoint(time_mlp_func, time_emb)
        # 将 VQAP global code 残差融合进时间条件，作为动作 expert 的全局意图条件。
        adarms_cond = self._fuse_global_code(time_cond, z_global)

        bsize, action_time_dim = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        att_masks = torch.tensor(
            [1] + ([0] * (self.config.action_horizon - 1)),
            dtype=action_emb.dtype,
            device=action_emb.device,
        )
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return action_emb, pad_masks, att_masks, adarms_cond

    # 将 VQAP global code 与 pi0.5 时间条件做残差融合。
    def _fuse_global_code(self, time_cond: Tensor, z_global: Tensor | None) -> Tensor:
        if z_global is None:
            return time_cond
        code_cond = self.code_global_proj(z_global.float())
        fused = torch.cat([time_cond.float(), code_cond], dim=-1)
        return time_cond + self.ada_cond_mlp(fused).to(dtype=time_cond.dtype)

    # 训练 forward：采样 flow-matching 轨迹点，并预测从 action 到噪声的速度场。
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """执行训练前向传播，返回逐元素 flow-matching MSE。"""
        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(observation, train=True)
        # 若调用方未传入 noise/time，则按 pi0.5 默认策略在线采样。
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 构造 flow-matching 中间点 x_t 与目标速度 u_t。
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # prefix 中加入 VQAP index/detail code；adapter_output 会保留辅助 loss 与 perplexity。
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs, prefix_pad_masks, prefix_att_masks, adapter_output = self._prepare_vqap_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t,
            time,
            adapter_output.z_global,
        )
        self._store_vqap_metrics(adapter_output)

        # 与原 pi0.5 PyTorch 路径保持 dtype 对齐，避免主干权重为 bf16 时出现混合类型问题。
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 拼接 prefix/suffix 后构造完整 attention mask 与 position id。
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # action expert 输出 suffix hidden，adaRMS 条件中已融合 VQAP global code。
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)

        # 将 suffix hidden 投影回 action 速度场，并与 flow target 计算 MSE。
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    # 推理采样：固定 observation 的 VQAP prefix/cache 后，逐步从噪声积分到动作。
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """在 VQAP code 条件下采样一个 action chunk。"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 推理时仍先从图像/语言 prefix 预测 VQAP code，再把 detail code 写入 prefix。
        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs, prefix_pad_masks, prefix_att_masks, adapter_output = self._prepare_vqap_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
        )
        self._store_vqap_metrics(adapter_output)

        # prefix 对每个 denoise step 不变，因此先计算 KV cache 复用。
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self._run_prefix_only(prefix_embs, prefix_pad_masks, prefix_att_masks, use_cache=True)

        # 从 t=1 的噪声端开始，用 Euler 积分反向走到 t=0 的动作端。
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                adapter_output.z_global,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    # 单步去噪：给定当前 x_t 和 prefix cache，预测当前时间步的速度场。
    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        z_global,
    ):
        """在指定 flow-matching 时间步执行一次去噪预测。"""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, z_global)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # suffix 需要能看见全部有效 prefix，同时 suffix 内部保持原 action block attention。
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        # suffix position id 从 prefix 有效长度之后继续累加，保证 cache 对齐。
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # 只跑 action expert 的 suffix；prefix KV 已由 sample_actions 提前缓存。
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    # 保存最近一次 VQAP adapter 的辅助指标，训练脚本可读取后写入日志。
    def _store_vqap_metrics(self, adapter_output: VQAPAdapterOutput) -> None:
        self.latest_vqap_metrics = {
            "loss_load": adapter_output.load_loss.detach(),
            "loss_load_weighted": (adapter_output.load_loss * self.config.vqap_load_loss_weight).detach(),
            "ppl_global": adapter_output.ppl_global.detach(),
            "ppl_detail": adapter_output.ppl_detail.detach(),
            "tau": torch.tensor(float(self.config.vqap_tau)),
            "code_dropout_p": torch.tensor(float(self.config.vqap_code_dropout_p)),
        }
