"""VQAP-conditioned pi0.5 model components."""

from .adapter import VQAPAdapter, VQAPAdapterOutput
from .codebook import FrozenVQAPCodebook
from .config import PI05VQAPConfig

__all__ = [
    "FrozenVQAPCodebook",
    "PI05VQAPConfig",
    "PI05VQAPPytorch",
    "VQAPAdapter",
    "VQAPAdapterOutput",
]


def __getattr__(name: str):
    if name == "PI05VQAPPytorch":
        from .pi05_vqap_pytorch import PI05VQAPPytorch

        return PI05VQAPPytorch
    raise AttributeError(name)
