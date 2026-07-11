"""Frozen VQAP codebook loading for the pi0.5 integration path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class FrozenVQAPCodebook(nn.Module):
    """Loads VQAP Stage 1 codebooks and exposes frozen lookup tensors."""

    EXPECTED_GLOBAL_SHAPE = (36, 512)
    EXPECTED_DETAIL_SHAPE = (192, 512)
    EXPECTED_NUM_DETAIL = 9

    def __init__(self, checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        super().__init__()
        checkpoint_path = self._resolve_path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in VQAP codebook checkpoint: {checkpoint_path}")

        global_codebook = self._extract_codebook(payload, "global")
        detail_codebook = self._extract_codebook(payload, "detail")
        self._validate_shape("global_codebook", global_codebook, self.EXPECTED_GLOBAL_SHAPE)
        self._validate_shape("detail_codebook", detail_codebook, self.EXPECTED_DETAIL_SHAPE)

        detail_queries = payload.get("detail_codebook_module", {}).get("learnable_queries")
        if detail_queries is not None:
            self._validate_shape("detail_codebook_module.learnable_queries", detail_queries, (self.EXPECTED_NUM_DETAIL, 512))

        self.checkpoint_path = str(checkpoint_path)
        self.epoch = int(payload.get("epoch", -1))
        self.global_step = int(payload.get("global_step", -1))
        self.stage = int(payload.get("stage", -1))
        self.perplexity_g = float(payload.get("perplexity_g", float("nan")))
        self.perplexity_d = float(payload.get("perplexity_d", float("nan")))

        self.register_buffer("global_codebook", global_codebook.detach().float().contiguous())
        self.register_buffer("detail_codebook", detail_codebook.detach().float().contiguous())

    @staticmethod
    def _resolve_path(checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        repo_root = Path(__file__).resolve().parents[2]
        return (repo_root / path).resolve()

    @staticmethod
    def _extract_codebook(payload: dict[str, Any], name: str) -> torch.Tensor:
        direct_key = f"{name}_codebook"
        module_key = f"{name}_codebook_module"
        if isinstance(payload.get(direct_key), dict) and "codebooks" in payload[direct_key]:
            return payload[direct_key]["codebooks"]
        if isinstance(payload.get(module_key), dict) and "quantizer.codebooks" in payload[module_key]:
            return payload[module_key]["quantizer.codebooks"]
        raise KeyError(f"Could not find {name} codebook in checkpoint payload.")

    @staticmethod
    def _validate_shape(name: str, tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}.")

    @property
    def global_size(self) -> int:
        return int(self.global_codebook.shape[0])

    @property
    def detail_size(self) -> int:
        return int(self.detail_codebook.shape[0])

    @property
    def code_dim(self) -> int:
        return int(self.global_codebook.shape[1])

    @property
    def num_detail_tokens(self) -> int:
        return self.EXPECTED_NUM_DETAIL
