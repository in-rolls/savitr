---
license: other
license_name: surya-ocr-2-license
license_link: https://huggingface.co/datalab-to/surya-ocr-2
base_model: datalab-to/surya-ocr-2
tags:
  - ocr
  - electoral-roll
  - india
  - surya
  - mlx
  - vlm
pipeline_tag: image-to-text
library_name: mlx-vlm
---

# savitr — terse electoral-roll OCR (distilled Surya)

`gojiberries/savitr` is [`datalab-to/surya-ocr-2`](https://huggingface.co/datalab-to/surya-ocr-2)
(650M Qwen3.5-VL-style OCR) **self-distilled** to read Indian electoral-roll pages and emit one
compact, pipe-delimited line per voter instead of verbose HTML:

```
epic|name|relation(F/H/M)|relative_name|house|age|sex
```

That is ~5× fewer decode tokens than the HTML output, so it runs ~2.7× faster end-to-end at the
teacher's accuracy. Converted to **MLX** 8-bit for Apple Silicon.

## Usage

```bash
pip install savitr
savitr ocr roll.pdf --terse          # auto-downloads this model
```

```python
from huggingface_hub import snapshot_download
from savitr import MLXSuryaOCR, parse_terse
from savitr.rolls.parse import TERSE_PROMPT
path = snapshot_download("gojiberries/savitr")
eng = MLXSuryaOCR(path, prompt=TERSE_PROMPT)
voters = parse_terse(eng.ocr_image("page.png")[0])
```

## How it was trained

Teacher = full Surya (`surya-ocr-2`) OCRs roll pages to HTML; a parser cleans them into terse
targets; the model is **LoRA-fine-tuned** on (page image → terse rows) — 450 pages drawn from
constituencies held out of the eval, 1 epoch, for $0 on a free Kaggle T4. The terse format is the only
behavioral change — reading ability is inherited from Surya.

## Evaluation (out-of-sample, vs the Surya teacher)

Held-out constituencies never seen in training (37 pages, 1,076 teacher voters):

| Field | Fidelity | | Field | Fidelity |
|---|---|---|---|---|
| voter recall | 99.3% | | relative name | 96.2% |
| EPIC | 97.2% | | relation code (F/H/M) | 97.9% |
| name | 96.2% | | house | 98.8% |
| age | 97.5% | | sex | 98.2% |

Per-voter record similarity **98.7%**, whole-page similarity **92.9%** (1 − normalized edit distance).
Fidelity = agreement with the teacher's output; absolute accuracy ≈ these × Surya's own ~93–95%.

## Limitations

**v0.2 (450 training pages, AC-holdout).** All fields are teacher-grade out-of-sample — EPIC, house,
age, sex, the relation code, names, and recall all 96–99% (the relation code, the weak spot of the
earlier 77-page model, is now 98%). Trained on **Manipur 2025 English** rolls; other states/scripts are
out of distribution. Pair with savitr's value-anchored `parse_terse`, which stays column-aligned even
when the model drops a field.

## License & attribution

Derived from `datalab-to/surya-ocr-2`; its license governs use of these weights. savitr's *code*
is MIT. Electoral rolls are public records published by the Election Commission of India.
