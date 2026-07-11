"""Standalone VQAP-conditioned pi0.5 PyTorch model.

This file deliberately copies the pi0.5 PyTorch path instead of editing
openpi's original PI0Pytorch class, so the existing RLBench fine-tuning
baseline remains untouched.
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


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Build block attention masks from padding and autoregressive block markers."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI05VQAPPytorch(nn.Module):
    """pi0.5 action model with a frozen VQAP codebook adapter."""

    def __init__(self, config: PI05VQAPConfig):
        super().__init__()
        if not getattr(config, "pi05", False):
            raise ValueError("PI05VQAPPytorch only supports pi0.5 configs.")
        self.config = config
        self.pi05 = True
        self.latest_vqap_metrics: dict[str, Tensor] = {}

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.prefix_width = int(paligemma_config.width)
        self.action_expert_width = int(action_expert_config.width)

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

        self.vqap_codebook = FrozenVQAPCodebook(config.vqap_codebook_path)
        self.vqap_adapter = VQAPAdapter(
            self.vqap_codebook,
            prefix_width=self.prefix_width,
            hidden_dim=self.vqap_codebook.code_dim,
            tau=config.vqap_tau,
            load_loss_weight=config.vqap_load_loss_weight,
        )
        self.code_global_proj = nn.Linear(self.vqap_codebook.code_dim, self.action_expert_width)
        self.code_detail_proj = nn.Linear(self.vqap_codebook.code_dim, self.prefix_width)
        self.ada_cond_mlp = nn.Linear(self.action_expert_width * 2, self.action_expert_width)
        self._init_code_injection_layers()

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def _init_code_injection_layers(self) -> None:
        # Residual fusion keeps the initial adaRMS condition identical to pi0.5.
        nn.init.zeros_(self.code_global_proj.weight)
        nn.init.zeros_(self.code_global_proj.bias)
        nn.init.zeros_(self.code_detail_proj.weight)
        nn.init.zeros_(self.code_detail_proj.bias)
        nn.init.zeros_(self.ada_cond_mlp.weight)
        nn.init.zeros_(self.ada_cond_mlp.bias)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05VQAPPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05VQAPPytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

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

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images and language tokens into the pi0.5 prefix."""
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

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

    def _append_index_tokens(self, prefix_embs: Tensor, pad_masks: Tensor, att_masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bsize = prefix_embs.shape[0]
        index_tokens = self.vqap_adapter.expanded_index_tokens(
            bsize,
            device=prefix_embs.device,
            dtype=prefix_embs.dtype,
        )
        index_pad = torch.ones(bsize, self.vqap_adapter.num_index_tokens, dtype=torch.bool, device=prefix_embs.device)
        # global starts block 1; first detail starts block 2; remaining details share block 2.
        index_att = torch.tensor([1, 1, *([0] * 8)], dtype=att_masks.dtype, device=prefix_embs.device)
        index_att = index_att[None, :].expand(bsize, -1)
        return (
            torch.cat([prefix_embs, index_tokens], dim=1),
            torch.cat([pad_masks, index_pad], dim=1),
            torch.cat([att_masks, index_att], dim=1),
        )

    def _append_detail_code_tokens(
        self,
        prefix_embs: Tensor,
        pad_masks: Tensor,
        att_masks: Tensor,
        z_detail: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsize = prefix_embs.shape[0]
        detail_tokens = self.code_detail_proj(z_detail.float()).to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        detail_pad = torch.ones(bsize, self.vqap_codebook.num_detail_tokens, dtype=torch.bool, device=prefix_embs.device)
        detail_att = torch.tensor([1, *([0] * 8)], dtype=att_masks.dtype, device=prefix_embs.device)
        detail_att = detail_att[None, :].expand(bsize, -1)
        return (
            torch.cat([prefix_embs, detail_tokens], dim=1),
            torch.cat([pad_masks, detail_pad], dim=1),
            torch.cat([att_masks, detail_att], dim=1),
        )

    def _run_prefix_only(
        self,
        prefix_embs: Tensor,
        pad_masks: Tensor,
        att_masks: Tensor,
        *,
        use_cache: bool,
    ):
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

    def _prepare_vqap_prefix(
        self,
        prefix_embs: Tensor,
        pad_masks: Tensor,
        att_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, VQAPAdapterOutput]:
        prefix_with_index, index_pad_masks, index_att_masks = self._append_index_tokens(prefix_embs, pad_masks, att_masks)
        prefix_out, _ = self._run_prefix_only(prefix_with_index, index_pad_masks, index_att_masks, use_cache=False)
        index_hidden = prefix_out[:, -self.vqap_adapter.num_index_tokens :]
        adapter_output = self.vqap_adapter(
            index_hidden,
            tau=self.config.vqap_tau,
            code_dropout_p=self.config.vqap_code_dropout_p if self.training else 0.0,
            code_mode=self.config.vqap_code_mode,
        )
        final_prefix, final_pad_masks, final_att_masks = self._append_detail_code_tokens(
            prefix_with_index,
            index_pad_masks,
            index_att_masks,
            adapter_output.z_detail,
        )
        return final_prefix, final_pad_masks, final_att_masks, adapter_output

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor, z_global: Tensor | None):
        """Embed noisy actions and timestep for the pi0.5 action expert."""
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_cond = self._apply_checkpoint(time_mlp_func, time_emb)
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

    def _fuse_global_code(self, time_cond: Tensor, z_global: Tensor | None) -> Tensor:
        if z_global is None:
            return time_cond
        code_cond = self.code_global_proj(z_global.float())
        fused = torch.cat([time_cond.float(), code_cond], dim=-1)
        return time_cond + self.ada_cond_mlp(fused).to(dtype=time_cond.dtype)

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Run a training forward pass and return the flow-matching MSE tensor."""
        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(observation, train=True)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

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

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

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

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Sample an action chunk with VQAP code conditioning."""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs, prefix_pad_masks, prefix_att_masks, adapter_output = self._prepare_vqap_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
        )
        self._store_vqap_metrics(adapter_output)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self._run_prefix_only(prefix_embs, prefix_pad_masks, prefix_att_masks, use_cache=True)

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

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        z_global,
    ):
        """Apply one denoising step at the requested flow-matching timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, z_global)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

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

    def _store_vqap_metrics(self, adapter_output: VQAPAdapterOutput) -> None:
        self.latest_vqap_metrics = {
            "loss_load": adapter_output.load_loss.detach(),
            "loss_load_weighted": (adapter_output.load_loss * self.config.vqap_load_loss_weight).detach(),
            "ppl_global": adapter_output.ppl_global.detach(),
            "ppl_detail": adapter_output.ppl_detail.detach(),
            "tau": torch.tensor(float(self.config.vqap_tau)),
            "code_dropout_p": torch.tensor(float(self.config.vqap_code_dropout_p)),
        }
