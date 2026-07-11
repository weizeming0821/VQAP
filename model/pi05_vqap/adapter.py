"""VQAP Adapter that predicts frozen codebook indices from pi0.5 prefix states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F  # noqa: N812

from .codebook import FrozenVQAPCodebook


@dataclass
class VQAPAdapterOutput:
    z_global: Tensor
    z_detail: Tensor
    logits_global: Tensor
    logits_detail: Tensor
    load_loss: Tensor
    ppl_global: Tensor
    ppl_detail: Tensor


class VQAPAdapter(nn.Module):
    """Predicts one global code and nine detail codes from special prefix tokens."""

    def __init__(
        self,
        codebook: FrozenVQAPCodebook,
        *,
        prefix_width: int = 2048,
        hidden_dim: int = 512,
        tau: float = 1.0,
        load_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}.")

        self.codebook = codebook
        self.prefix_width = int(prefix_width)
        self.hidden_dim = int(hidden_dim)
        self.tau = float(tau)
        self.load_loss_weight = float(load_loss_weight)
        self.num_index_tokens = 1 + self.codebook.num_detail_tokens

        self.index_token_embeddings = nn.Parameter(torch.empty(self.num_index_tokens, self.prefix_width))
        self.global_head = nn.Sequential(
            nn.Linear(self.prefix_width, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.codebook.global_size),
        )
        self.detail_head = nn.Sequential(
            nn.Linear(self.prefix_width, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.codebook.detail_size),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.index_token_embeddings, mean=0.0, std=0.02)

    def expanded_index_tokens(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        tokens = self.index_token_embeddings.to(device=device, dtype=dtype)
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        index_hidden_states: Tensor,
        *,
        tau: float | None = None,
        code_dropout_p: float = 0.0,
        code_mode: Literal["always_on", "always_off"] = "always_on",
    ) -> VQAPAdapterOutput:
        if index_hidden_states.ndim != 3:
            raise ValueError(f"index_hidden_states must be [B, 10, D], got {tuple(index_hidden_states.shape)}.")
        if index_hidden_states.shape[1] != self.num_index_tokens:
            raise ValueError(
                f"Expected {self.num_index_tokens} index tokens, got {index_hidden_states.shape[1]}."
            )
        if not 0.0 <= code_dropout_p <= 1.0:
            raise ValueError(f"code_dropout_p must be in [0, 1], got {code_dropout_p}.")

        tau = float(self.tau if tau is None else tau)
        hidden = index_hidden_states.float()
        logits_global = self.global_head(hidden[:, 0])
        logits_detail = self.detail_head(hidden[:, 1:])

        probs_global = F.softmax(logits_global / tau, dim=-1)
        probs_detail = F.softmax(logits_detail / tau, dim=-1)
        load_loss = self._load_balance_loss(probs_global, probs_detail)
        ppl_global = self._perplexity(probs_global.mean(dim=0))
        ppl_detail = self._perplexity(probs_detail.reshape(-1, probs_detail.shape[-1]).mean(dim=0))

        if code_mode == "always_off":
            z_global = torch.zeros(
                logits_global.shape[0],
                self.codebook.code_dim,
                device=logits_global.device,
                dtype=logits_global.dtype,
            )
            z_detail = torch.zeros(
                logits_detail.shape[0],
                self.codebook.num_detail_tokens,
                self.codebook.code_dim,
                device=logits_detail.device,
                dtype=logits_detail.dtype,
            )
            return VQAPAdapterOutput(
                z_global=z_global,
                z_detail=z_detail,
                logits_global=logits_global,
                logits_detail=logits_detail,
                load_loss=load_loss,
                ppl_global=ppl_global,
                ppl_detail=ppl_detail,
            )
        if code_mode != "always_on":
            raise ValueError(f"Unsupported code_mode={code_mode!r}; stage 1 supports always_on/always_off only.")

        if self.training:
            weights_global = self._straight_through_gumbel(logits_global, tau)
            weights_detail = self._straight_through_gumbel(logits_detail, tau)
        else:
            weights_global = F.one_hot(logits_global.argmax(dim=-1), num_classes=self.codebook.global_size).float()
            weights_detail = F.one_hot(logits_detail.argmax(dim=-1), num_classes=self.codebook.detail_size).float()

        z_global = weights_global @ self.codebook.global_codebook.to(device=weights_global.device, dtype=weights_global.dtype)
        z_detail = weights_detail @ self.codebook.detail_codebook.to(device=weights_detail.device, dtype=weights_detail.dtype)
        z_global, z_detail = self._apply_code_dropout(z_global, z_detail, code_dropout_p)

        return VQAPAdapterOutput(
            z_global=z_global,
            z_detail=z_detail,
            logits_global=logits_global,
            logits_detail=logits_detail,
            load_loss=load_loss,
            ppl_global=ppl_global,
            ppl_detail=ppl_detail,
        )

    @staticmethod
    def _straight_through_gumbel(logits: Tensor, tau: float) -> Tensor:
        soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=soft.shape[-1]).to(dtype=soft.dtype)
        return hard + soft - soft.detach()

    @staticmethod
    def _perplexity(probabilities: Tensor) -> Tensor:
        probabilities = probabilities.clamp_min(1e-8)
        return torch.exp(-(probabilities * probabilities.log()).sum())

    @staticmethod
    def _load_balance_loss(probs_global: Tensor, probs_detail: Tensor) -> Tensor:
        avg_global = probs_global.mean(dim=0).clamp_min(1e-8)
        avg_detail = probs_detail.reshape(-1, probs_detail.shape[-1]).mean(dim=0).clamp_min(1e-8)
        global_loss = (avg_global * (avg_global * avg_global.numel()).log()).sum()
        detail_loss = (avg_detail * (avg_detail * avg_detail.numel()).log()).sum()
        return global_loss + detail_loss

    def _apply_code_dropout(self, z_global: Tensor, z_detail: Tensor, p: float) -> tuple[Tensor, Tensor]:
        if (not self.training) or p <= 0.0:
            return z_global, z_detail
        keep = (torch.rand(z_global.shape[0], device=z_global.device) >= p).to(dtype=z_global.dtype)
        z_global = z_global * keep[:, None]
        z_detail = z_detail * keep[:, None, None]
        return z_global, z_detail
