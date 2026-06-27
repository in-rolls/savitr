# savitr

Fast [Surya OCR](https://github.com/datalab-to/surya) on Apple Silicon, applied to Indian electoral
rolls. savitr runs Surya ~3.6× faster via **MLX**, and ships a distilled, electoral-roll-specific
model — *terse-Surya* — that emits one compact line per voter (~5× fewer decode tokens at Surya's
accuracy), plus a pipeline that turns scanned roll PDFs into the canonical voter CSV.

## Install

```bash
pip install savitr               # MLX runtime + terse roll model (auto-downloaded from HF)
pip install "savitr[backend]"    # + the generic MLX Backend for Surya's own pipeline
pip install "savitr[train]"      # + the distillation toolchain
```

## Quickstart

```bash
savitr ocr roll.pdf --terse                       # voter records from a PDF
savitr parse-rolls -d english/ -o voters.csv --terse
```

```python
from savitr import MLXSuryaOCR, parse_terse

eng = MLXSuryaOCR("models/surya-terse-8bit", prompt=...)
voters = parse_terse(eng.ocr_image("page.png")[0])
```

```{toctree}
:hidden:
:caption: Documentation

API reference <api>
Model card <model_card>
Findings <findings>
```
