"""savitr — fast Surya OCR on Apple Silicon, applied to Indian electoral rolls.

Layers (see the README "What's in the box"):
  * generic MLX runtime: :class:`savitr.mlx_ocr.MLXSuryaOCR`;
  * electoral-roll parsing and pipelines: :mod:`savitr.rolls`.
Training/distillation is repo-only (top-level ``training/``), not shipped in the wheel.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("savitr")

if TYPE_CHECKING:
    from .mlx_ocr import PROMPT, MLXSuryaOCR

# Keep the pure-Python parsing API importable without MLX.
from .rolls.parse import (
    TERSE_PROMPT,
    dedupe_voters,
    parse_terse,
    parse_voters,
    resolve_terse_model,
    to_terse,
)


def __getattr__(name: str):
    """Lazily expose the MLX engine without loading MLX for parser users."""
    if name in ("MLXSuryaOCR", "PROMPT"):
        from . import mlx_ocr

        return getattr(mlx_ocr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PROMPT",
    "TERSE_PROMPT",
    "MLXSuryaOCR",
    "__version__",
    "dedupe_voters",
    "parse_terse",
    "parse_voters",
    "resolve_terse_model",
    "to_terse",
]
