"""savitr — fast Surya OCR on Apple Silicon, applied to Indian electoral rolls.

Layers (see the README "What's in the box"):
  * generic MLX runtime for Surya — :class:`savitr.mlx_ocr.MLXSuryaOCR`, :mod:`savitr.mlx_backend`;
  * the electoral-roll application — :mod:`savitr.rolls` (parsing + canonical-CSV pipeline) and the
    distilled terse model.
Training/distillation is repo-only (top-level ``training/``), not shipped in the wheel.
"""

from typing import TYPE_CHECKING

__version__ = "0.2.0"

if TYPE_CHECKING:  # names are real at runtime via __getattr__; this is just for type-checkers
    from savitr.mlx_ocr import PROMPT, MLXSuryaOCR

# The roll parsing/rendering helpers are pure Python and import anywhere. The MLX engine
# (MLXSuryaOCR/PROMPT) is loaded lazily via __getattr__ so `import savitr` — and the pure-Python
# `parse_terse` — work on non-Apple-Silicon machines that don't have mlx installed.
from savitr.rolls.parse import (  # noqa: F401
    TERSE_PROMPT,
    dedupe_voters,
    parse_terse,
    parse_voters,
    resolve_terse_model,
    to_terse,
)


def __getattr__(name: str):
    """Lazily expose the MLX engine so the pure-Python API imports without mlx (PEP 562)."""
    if name in ("MLXSuryaOCR", "PROMPT"):
        from savitr import mlx_ocr

        return getattr(mlx_ocr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MLXSuryaOCR",
    "PROMPT",
    "TERSE_PROMPT",
    "parse_voters",
    "dedupe_voters",
    "parse_terse",
    "to_terse",
    "resolve_terse_model",
    "__version__",
]
