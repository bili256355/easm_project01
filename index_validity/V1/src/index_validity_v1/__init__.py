"""index_validity_v1: smooth5 index shape and physical representativeness diagnostics."""

from .pipeline import run_index_validity_smooth5_v1
from .settings import IndexValiditySettings

__all__ = ["run_index_validity_smooth5_v1", "IndexValiditySettings"]
