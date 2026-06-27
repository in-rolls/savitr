"""savitr — fast Surya OCR on Apple Silicon, applied to Indian electoral rolls.

Layers (see the README "What's in the box"):
  * generic MLX runtime for Surya — :class:`savitr.mlx_ocr.MLXSuryaOCR`, :mod:`savitr.mlx_backend`;
  * the electoral-roll application — :mod:`savitr.rolls` (parsing + canonical-CSV pipeline) and the
    distilled terse model.
Training/distillation is repo-only (top-level ``training/``), not shipped in the wheel.
"""

__version__ = "0.1.0"

from savitr.mlx_ocr import PROMPT, MLXSuryaOCR  # noqa: F401
from savitr.rolls.parse import (  # noqa: F401
    TERSE_PROMPT,
    dedupe_voters,
    parse_terse,
    parse_voters,
    resolve_terse_model,
    to_terse,
)

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
