# savitr — fast Surya OCR on Apple Silicon, for Indian electoral rolls

[![PyPI](https://img.shields.io/pypi/v/savitr.svg)](https://pypi.org/project/savitr/)
[![CI](https://github.com/in-rolls/savitr/actions/workflows/ci.yml/badge.svg)](https://github.com/in-rolls/savitr/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/savitr)](https://pepy.tech/project/savitr)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-gojiberries%2Fsavitr-yellow)](https://huggingface.co/gojiberries/savitr)
[![Docs](https://img.shields.io/badge/docs-in--rolls.github.io%2Fsavitr-blue)](https://in-rolls.github.io/savitr/)

savitr runs [Surya OCR](https://github.com/datalab-to/surya)
(`datalab-to/surya-ocr-2`) on Apple Silicon via **MLX**. It also ships an
electoral-roll-specific model that emits one compact line per voter, plus a
pipeline that turns scanned roll PDFs into the canonical voter CSV. It runs
locally without a cloud GPU.

## Install

**Requirements:** an **Apple-Silicon Mac** (M-series — the OCR runs on MLX) and **poppler** (used to
read PDFs):

```bash
brew install poppler             # macOS  (Debian: sudo apt-get install poppler-utils)
```

Then:

```bash
pip install savitr               # MLX runtime + terse roll model (auto-downloaded from HF)
pip install "savitr[train]"      # + the distillation toolchain (transformers/peft/torch)
```

Latest from git: `pip install "git+https://github.com/in-rolls/savitr"`

The terse model is fetched from [`gojiberries/savitr`](https://huggingface.co/gojiberries/savitr) on
first use (~800 MB, one time). The pure-Python parsing API (`parse_terse`) imports on any platform;
only the OCR itself needs Apple Silicon.

## Quickstart

The distilled terse model is the default and auto-downloads on first run — a bundled sample roll lets
you try it immediately:

```bash
# OCR the bundled sample roll -> per-page voter summary (works from any install)
savitr ocr "$(savitr sample)"

# ... or write the voter records straight to CSV
savitr ocr "$(savitr sample)" -o voters.csv

# whole rolls -> canonical voter CSV (a directory of *_ENG.pdf, or a single -f file)
savitr parse-rolls -d english/ -o voters.csv
```

```python
from savitr import MLXSuryaOCR, parse_terse, resolve_terse_model
from savitr.rolls.parse import TERSE_PROMPT

eng = MLXSuryaOCR(
    resolve_terse_model(), prompt=TERSE_PROMPT
)  # downloads the model if not local
text, _ = eng.ocr_image("page.png")
voters = parse_terse(text)  # [{'id': 'KMY...', 'elector_name': ..., 'age': ..., ...}]
```

## Two models, and which one you want

**Electoral rolls → the terse model.** Published at
[`gojiberries/savitr`](https://huggingface.co/gojiberries/savitr) and downloaded on first use, so
everything above works with no setup. It was distilled to emit voter rows and will emit them
whatever the page holds — it is not a general OCR.

**Anything else → base Surya.** Upstream publishes `datalab-to/surya-ocr-2` but not an MLX build of
it, so convert it once (~1.3 GB fetched, ~500 MB written):

```bash
python -m mlx_vlm convert --hf-path datalab-to/surya-ocr-2 \
    --mlx-path models/surya-mlx-4bit -q --q-bits 4
```

```python
from savitr import MLXSuryaOCR

eng = MLXSuryaOCR()  # finds models/surya-mlx-4bit, or $SAVITR_BASE_PATH
eng = MLXSuryaOCR("/some/other/model")  # or say where
text, _ = eng.ocr_image("page.png")  # HTML: <table><tr><td>…
```

With no converted model, every entry point — the constructor, `savitr ocr --html`, and
`--cover-model` — says the same thing and repeats the command above. `savitr ocr --html` needs it;
`--cover-model` is optional metadata and carries on without it.

## terse-Surya (`gojiberries/savitr`)

Surya self-distilled to emit pipe-delimited voter rows. The
[model card](https://in-rolls.github.io/savitr/model_card.html) is the single
source for checkpoint provenance, evaluation definitions, results, and
limitations.

## What's in the box

Three layers — two you install and use, one for reproducing the model:

```
src/savitr/            # the pip package (use it)
  mlx_ocr.py           # GENERIC MLX Surya engine (MLXSuryaOCR) — run any Surya OCR fast
  rolls/               # ELECTORAL-ROLL app: parse · fields · schema · pipeline · ocr
  cli.py, __init__.py
training/              # repo-only: build_corpus · train · eval · merge · kaggle_*  (reproduce the model)
```

- **Use it — electoral rolls (the product):** `savitr parse-rolls` / `savitr ocr` run the distilled
  terse model (the default) on roll PDFs → voter records / canonical CSV (`savitr.rolls`).
- **Use it — generic fast Surya:** `savitr.MLXSuryaOCR` runs any compatible
  Surya model on Apple Silicon.
- **Reproduce it — training/distillation:** lives in top-level `training/`, **not shipped in the
  wheel** (install the `[train]` extra to run it). We ship code to *use* the model, not to train it.

## How it was built / what was tried

See [the findings](https://in-rolls.github.io/savitr/findings.html) for the
measured baseline (decode, not cold-start, dominates; ~110 s/page on llama.cpp),
the MLX win, the tested negatives (compact prompt, guided JSON, parallelism),
and the distillation method and numbers.

## Develop

```bash
uv sync --all-groups --all-extras
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
