"""Generic MLX runtime for Surya OCR — load the model once, OCR page images to text.

This is the engine layer (roll-agnostic): it turns a page image into the model's raw text — HTML
for the base ``surya-ocr-2``, terse rows for the distilled roll model — given whatever ``prompt``
you pass. The electoral-roll parsing of that text lives in :mod:`savitr.rolls`.

Runs ~175-180 tok/s on Apple Silicon (~3.6x the llama.cpp f16 pipeline). Reusable for any Surya
OCR task, not just rolls.
"""

PROMPT = "OCR this image to HTML."


class MLXSuryaOCR:
    """Load an MLX-converted Surya model once; OCR page images to text."""

    def __init__(
        self, mlx_path: str = "models/surya-mlx-4bit", max_tokens: int = 8192, prompt: str = PROMPT
    ) -> None:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self._generate = generate
        self.model, self.processor = load(mlx_path)
        config = load_config(mlx_path)
        self.prompt = apply_chat_template(self.processor, config, prompt, num_images=1)
        self.max_tokens = max_tokens

    def ocr_image(self, png_path: str) -> tuple[str, int]:
        """OCR one page image; return ``(text, generation_token_count)``."""
        res = self._generate(
            self.model,
            self.processor,
            self.prompt,
            image=png_path,
            max_tokens=self.max_tokens,
            verbose=False,
        )
        return getattr(res, "text", None) or str(res), getattr(res, "generation_tokens", 0)
