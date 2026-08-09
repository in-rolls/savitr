"""Generic MLX runtime for Surya OCR — load the model once, OCR page images to text.

This is the engine layer (roll-agnostic): it turns a page image into the model's raw text — HTML
for the base ``surya-ocr-2``, terse rows for the distilled roll model — given whatever ``prompt``
you pass. The electoral-roll parsing of that text lives in :mod:`savitr.rolls`.

Runs ~175-180 tok/s on Apple Silicon (~3.6x the llama.cpp f16 pipeline). Reusable for any Surya
OCR task, not just rolls.
"""

import os

PROMPT = "OCR this image to HTML."

#: Upstream Surya weights. Not published in MLX form — converted locally, once.
BASE_REPO = "datalab-to/surya-ocr-2"

#: Where a converted base model is looked for, in order. ``models/`` first because that is the
#: layout the repo's own bench/ and training/ scripts assume; the cache directory second because it
#: is the only one that works from an arbitrary working directory.
BASE_PATHS = ("models/surya-mlx-4bit", "~/.cache/savitr/surya-mlx-4bit")


def convert_hint(looked_in: list[str]) -> str:
    """Build the one message every entry point gives when there is no base model."""
    return (
        "no MLX Surya model found (looked in: " + ", ".join(looked_in) + ").\n"
        "Base Surya is not published in MLX form; convert it once:\n\n"
        f"    python -m mlx_vlm convert --hf-path {BASE_REPO} \\\n"
        "        --mlx-path models/surya-mlx-4bit -q --q-bits 4\n\n"
        'Or pass a path of your own: MLXSuryaOCR("/path/to/model").\n'
        "For electoral rolls use the distilled model instead, which is published and downloads "
        "itself: MLXSuryaOCR(resolve_terse_model())."
    )


def base_model_path(path: str | None = None) -> str:
    """Return a local MLX Surya directory, or say exactly how to make one.

    The terse roll model has had :func:`savitr.rolls.resolve_terse_model` since it shipped, because
    savitr publishes it. Base Surya is upstream's and has to be converted, so there is nothing to
    download — but until now there was nothing to *say* either. The default ``mlx_path`` was the
    bare relative string ``models/surya-mlx-4bit``; when that directory did not exist
    ``mlx_vlm.load`` read it as a Hub repo id and raised ``RepositoryNotFoundError`` for
    ``models/models/surya-mlx-4bit`` — a repo that has never existed, naming nothing anyone can act
    on. ``models/`` is gitignored, so that path only ever resolved on a maintainer's machine, and
    only from the repo root.

    Raises ``FileNotFoundError`` rather than exiting: this is a library constructor, and taking the
    host process down is not its call. The CLI entry points turn it into a ``SystemExit``.
    """
    looked_in: list[str] = []
    for candidate in (path, os.environ.get("SAVITR_BASE_PATH"), *BASE_PATHS):
        if not candidate:
            continue
        expanded = os.path.expanduser(candidate)
        if os.path.isdir(expanded):
            return expanded
        looked_in.append(candidate)
    raise FileNotFoundError(convert_hint(looked_in))


class MLXSuryaOCR:
    """Load an MLX-converted Surya model once; OCR page images to text.

    With no ``mlx_path`` this looks for a converted base Surya — see :func:`base_model_path`. Pass
    ``savitr.rolls.resolve_terse_model()`` for the distilled roll model instead: it is much faster
    on rolls and useless off them, having been distilled to emit voter rows whatever the page holds.
    """

    def __init__(
        self, mlx_path: str | None = None, max_tokens: int = 8192, prompt: str = PROMPT
    ) -> None:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        mlx_path = base_model_path(mlx_path)
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
