# savitr — fast Surya OCR on Apple Silicon, for Indian electoral rolls

[![PyPI](https://img.shields.io/pypi/v/savitr.svg)](https://pypi.org/project/savitr/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-gojiberries%2Fsavitr-yellow)](https://huggingface.co/gojiberries/savitr)
[![Docs](https://img.shields.io/badge/docs-in--rolls.github.io%2Fsavitr-blue)](https://in-rolls.github.io/savitr/)

savitr makes [Surya OCR](https://github.com/datalab-to/surya) (`datalab-to/surya-ocr-2`, a 650M
Qwen3.5-VL-style model) fast on Apple Silicon via **MLX** (~3.6× over llama.cpp), and ships a
distilled, **electoral-roll-specific** model — *terse-Surya* — that emits one compact line per
voter instead of verbose HTML (~5× fewer decode tokens at Surya's accuracy), plus a pipeline that
turns scanned roll PDFs into the canonical voter CSV. Runs locally; no cloud GPU.

## Install

**Requirements:** an **Apple-Silicon Mac** (M-series — the OCR runs on MLX) and **poppler** (used to
read PDFs):

```bash
brew install poppler             # macOS  (Debian: sudo apt-get install poppler-utils)
```

Then:

```bash
pip install savitr               # MLX runtime + terse roll model (auto-downloaded from HF)
pip install "savitr[backend]"    # + the generic MLX Backend for Surya's own pipeline
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
eng = MLXSuryaOCR(resolve_terse_model(), prompt=TERSE_PROMPT)   # downloads the model if not local
text, _ = eng.ocr_image("page.png")
voters = parse_terse(text)        # [{'id': 'KMY...', 'elector_name': ..., 'age': ..., ...}]
```

## terse-Surya (`gojiberries/savitr`)

Surya self-distilled to emit pipe-delimited voter rows
(`epic|name|relation(F/H/M)|relative|house|age|sex`). Trained for **$0 on a free Kaggle T4** by
labeling roll pages with Surya itself (teacher) and LoRA-fine-tuning to the terse format.

**Speed (M4 / 16 GB)**

| Pipeline | s/page |
|---|---|
| Original Surya (llama.cpp f16) | ~178 |
| MLX 4-bit, HTML | ~38 |
| **terse-Surya MLX 8-bit** | **~17.5** |

**Out-of-sample fidelity to the Surya teacher** (held-out constituencies — 37 pages, 1,076 voters)

| Field | | Field | |
|---|---|---|---|
| voter recall | 99.3% | relative name | 96.2% |
| EPIC | 97.2% | relation (F/H/M) | 97.9% |
| name | 96.2% | house | 98.8% |
| age | 97.5% | sex | 98.2% |

Per-voter record similarity **98.7%**, whole-page **92.9%** (1 − normalized edit distance).

> **v0.2** — 450 training pages, constituency-holdout. Absolute accuracy ≈ these × the teacher's own
> ~93–95%.

## What's in the box

Three layers — two you install and use, one for reproducing the model:

```
savitr/                # the pip package (use it)
  mlx_ocr.py           # GENERIC MLX Surya engine (MLXSuryaOCR) — run any Surya OCR fast
  mlx_backend.py       # GENERIC Surya Backend (mlx) — also offered upstream (PR.md)
  rolls/               # ELECTORAL-ROLL app: parse · fields · schema · pipeline · ocr
  cli.py, __init__.py
training/              # repo-only: build_corpus · train · eval · merge · kaggle_*  (reproduce the model)
```

- **Use it — electoral rolls (the product):** `savitr parse-rolls` / `savitr ocr` run the distilled
  terse model (the default) on roll PDFs → voter records / canonical CSV (`savitr.rolls`).
- **Use it — generic fast Surya:** `savitr.MLXSuryaOCR` + the `mlx` `Backend` run *any* Surya OCR
  ~3.6× faster on Apple Silicon (also offered upstream to Surya — see [`PR.md`](PR.md)).
- **Reproduce it — training/distillation:** lives in top-level `training/`, **not shipped in the
  wheel** (install the `[train]` extra to run it). We ship code to *use* the model, not to train it.

## How it was built / what was tried

See [FINDINGS.md](FINDINGS.md) for the measured baseline (decode, not cold-start, dominates;
~110 s/page on llama.cpp), the MLX win, the tested negatives (compact prompt, guided JSON,
parallelism), and the distillation method + numbers.

## Develop

```bash
pip install -e ".[backend,train,dev]"
ruff format . && ruff check .
```
