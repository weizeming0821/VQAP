"""Lightweight config for the standalone VQAP + pi0.5 PyTorch model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_codebook_path() -> str:
    return str(repo_root() / "checkpoints" / "vqap_pretrain" / "stage1" / "codebook.pth")


@dataclass
class PI05VQAPConfig:
    """Config fields needed by the copied pi0.5 PyTorch model plus VQAP knobs."""

    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int | None = None
    pi05: bool = True
    discrete_state_input: bool = False
    pytorch_compile_mode: str | None = None

    vqap_codebook_path: str = default_codebook_path()
    vqap_tau: float = 1.0
    vqap_code_dropout_p: float = 0.0
    vqap_load_loss_weight: float = 0.01
    vqap_code_mode: Literal["always_on", "always_off"] = "always_on"

    def __post_init__(self) -> None:
        if not self.pi05:
            raise ValueError("PI05VQAPConfig is only for pi0.5; pi05 must stay True.")
        if self.max_token_len is None:
            self.max_token_len = 200
        if self.action_dim != 32:
            raise ValueError(f"pi0.5 RLBench expects action_dim=32, got {self.action_dim}.")
        if self.action_horizon != 50:
            raise ValueError(f"pi0.5 RLBench expects action_horizon=50, got {self.action_horizon}.")
        if self.vqap_tau <= 0:
            raise ValueError(f"vqap_tau must be positive, got {self.vqap_tau}.")
        if not 0.0 <= self.vqap_code_dropout_p <= 1.0:
            raise ValueError(f"vqap_code_dropout_p must be in [0, 1], got {self.vqap_code_dropout_p}.")
