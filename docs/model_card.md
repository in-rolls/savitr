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

`in-rolls/savitr` is [`datalab-to/surya-ocr-2`](https://huggingface.co/datalab-to/surya-ocr-2)
(650M Qwen3.5-VL-style OCR) **self-distilled** to read Indian electoral-roll pages and emit one
compact, pipe-delimited line per voter instead of verbose HTML:

```
epic|name|relation(F/H/M)|relative_name|house|age|sex
```

That is ~5× fewer decode tokens than the HTML output, so it runs ~2.7× faster end-to-end at the
teacher's accuracy. Converted to **MLX** for Apple Silicon (8-bit recommended; 4-bit also included).

## Usage

```bash
pip install savitr
savitr ocr roll.pdf --terse          # auto-downloads this model
```

```python
from huggingface_hub import snapshot_download
from savitr import MLXSuryaOCR, parse_terse
path = snapshot_download("in-rolls/savitr")
eng = MLXSuryaOCR(f"{path}/surya-terse-8bit", terse=True)
voters = parse_terse(eng.ocr_image("page.png")[0])
```

## How it was trained

Teacher = full Surya (`surya-ocr-2`) OCRs roll pages to HTML; a parser cleans them into terse
targets; the model is **LoRA-fine-tuned** on (page image → terse rows). Trained for $0 on a free
Kaggle T4. The terse format is the only behavioral change — reading ability is inherited from Surya.

## Evaluation (out-of-sample, vs the Surya teacher)

Held-out constituencies never seen in training:

| Field | Fidelity | | Field | Fidelity |
|---|---|---|---|---|
| EPIC | 97% | | relative name | 88% |
| house | 99% | | name | 82% |
| sex | 98% | | relation code | 79% |
| age | 96% | | voter recall | 81% |

Fidelity = agreement with the teacher's output; absolute accuracy ≈ these × Surya's own ~93–95%.

## Limitations

**v0.1, early model (77 training pages).** Structured fields (EPIC/house/age/sex) are teacher-grade
(96–99%); names, the relation code, and recall (~80%) are weaker and improve with more data — a
larger-corpus revision is in progress. Trained on **Manipur 2025 English** rolls; other
states/scripts are out of distribution. Pair with savitr's value-anchored `parse_terse`, which
stays column-aligned even when the model drops a field.

## License & attribution

Derived from `datalab-to/surya-ocr-2`; its license governs use of these weights. savitr's *code*
is MIT. Electoral rolls are public records published by the Election Commission of India.
