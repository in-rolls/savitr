"""Generic MLX runtime for Surya OCR — load the model once, OCR page images to text.

This roll-agnostic engine turns a page image into the model's raw text. The
electoral-roll parsing of that text lives in :mod:`savitr.rolls`.

It is reusable for any Surya OCR task, not just rolls.
"""

import os
from pathlib import Path

PROMPT = "OCR this image to HTML."

#: Upstream Surya weights. Not published in MLX form — converted locally, once.
BASE_REPO = "datalab-to/surya-ocr-2"

#: Locations searched for a converted base model, in order.
BASE_PATHS = ("models/surya-mlx-4bit", "~/.cache/savitr/surya-mlx-4bit")


def convert_hint(looked_in: list[str]) -> str:
    """Build the one message every entry point gives when there is no base model."""
    return (
        "no MLX Surya model found (looked in: " + ", ".join(looked_in) + ").\n"
        "Base Surya is not published in MLX form; convert it once:\n\n"
        f"    python -m mlx_vlm convert --hf-path {BASE_REPO} \\\n"
        "        --mlx-path models/surya-mlx-4bit -q --q-bits 4\n\n"
        'Or pass a path of your own: MLXSuryaOCR("/path/to/model").\n'
        "For electoral rolls use the published distilled model, which downloads "
        "itself: MLXSuryaOCR(resolve_terse_model())."
    )


def base_model_path(path: str | None = None) -> str:
    """Return a local MLX Surya directory, or say exactly how to make one.

    Savitr publishes the terse roll model, while base Surya must be converted
    locally. This function prevents a missing local path from being mistaken
    for a Hub repository ID.

    The library raises ``FileNotFoundError``; CLI entry points turn it into a
    ``SystemExit`` with the same guidance.
    """
    looked_in: list[str] = []
    for candidate in (path, os.environ.get("SAVITR_BASE_PATH"), *BASE_PATHS):
        if not candidate:
            continue
        expanded = Path(candidate).expanduser()
        if expanded.is_dir():
            return str(expanded)
        looked_in.append(candidate)
    raise FileNotFoundError(convert_hint(looked_in))


class MLXSuryaOCR:
    """Load an MLX-converted Surya model once; OCR page images to text.

    With no ``mlx_path``, this looks for a converted base Surya model. Pass
    ``savitr.rolls.resolve_terse_model()`` for the electoral-roll model.
    """

    def __init__(
        self, mlx_path: str | None = None, max_tokens: int = 8192, prompt: str = PROMPT
    ) -> None:
        # mlx-vlm is installed only on Apple Silicon; Linux still type-checks
        # the pure-Python package and exercises its parser API.
        from mlx_vlm import generate, load  # type: ignore[import-not-found]
        from mlx_vlm.prompt_utils import (  # type: ignore[import-not-found]
            apply_chat_template,
        )
        from mlx_vlm.utils import load_config  # type: ignore[import-not-found]

        mlx_path = base_model_path(mlx_path)
        self._generate = generate
        self.model, self.processor = load(mlx_path)
        config = load_config(mlx_path)
        self.prompt = apply_chat_template(self.processor, config, prompt, num_images=1)
        self.max_tokens = max_tokens

    def ocr_image(self, png_path: str) -> tuple[str, int]:
        """OCR one page image; return ``(text, generation_token_count)``."""
        # mlx_vlm annotates `generate` more narrowly than it accepts: `processor`
        # is declared ProcessorLike | PreTrainedTokenizer while load() hands
        # back a ProcessorMixin, and `prompt` is declared str while the chat
        # form is a list of message dicts. Both work at runtime.
        res = self._generate(
            self.model,
            self.processor,  # pyright: ignore[reportArgumentType]
            self.prompt,  # pyright: ignore[reportArgumentType]
            image=png_path,
            max_tokens=self.max_tokens,
            verbose=False,
        )
        return getattr(res, "text", None) or str(res), getattr(
            res, "generation_tokens", 0
        )
