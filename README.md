# savitr — making Surya OCR fast on Apple Silicon

Acceleration work for [Surya OCR](https://github.com/datalab-to/surya) (`surya-ocr-2`, a
650M-param Qwen3-VL-style OCR model) on Apple Silicon (M-series / MPS), driven by a concrete
workload: parsing scanned Indian electoral-roll PDFs into structured voter records, at
state-to-national scale, locally (no cloud GPU).

Surya is accurate on this data but slow on a Mac. This repo measures *why*, then optimizes —
quantization, parallelism, image-token/decode reduction, an MLX backend, and a narrow
distilled model — scoring every speed variant against unmodified f16 Surya output so we know
the accuracy cost of each win.

## Measured baseline (M4 / 16 GB, llama.cpp backend)

| Fact | Value |
|---|---|
| Warm voter page | **~110 s/page** (not the ~9 s headline; that's lighter pages) |
| Vision encode | ~5 s/page — **decode dominates** (~100 s) |
| Why decode is slow | page injects a large image-token context (Qwen-VL ≥1024 img tokens) → ~11–48 tok/s |
| Q4_K_M quant (llama.cpp) | 1.41× decode (48→68 tok/s) |
| **MLX 4-bit backend** | **~3.6× decode, ~5.2× full-PDF end-to-end (33.9 s/page), <2 GB, ~97–100% field acc** ✅ |

**Headline: the MLX backend is the win** — converting `datalab-to/surya-ocr-2` to MLX 4-bit
runs at ~175–180 tok/s (vs llama.cpp's 48–68), dropping a full PDF from ~178 s/page to
~34 s/page. Tested negatives: compact prompt, guided JSON schema, parallelism. See
[FINDINGS.md](FINDINGS.md). Scale: ~24k pages for Manipur (1,756 PDFs × ~14 pp) ≈ ~11 days/M4
on MLX (vs ~49 on the f16 baseline).

## Layout

```
savitr/
  mlx_backend.py       # RELEASABLE: an 'mlx' Backend for Surya (SuryaInferenceManager(method="mlx"))
  parse_manipur_mlx.py # PRODUCTION: PDF -> canonical voter CSV (reuses repo fields.py)
  mlx_ocr.py           # MLX engine — load once, OCR a PDF, layout-robust voter parser
bench/
  bench_surya.py       # warm-path timing (llama.cpp): load / cold / warm s-per-page + structure
  compare_quality.py   # run the real Surya pipeline per model variant; score vs f16 gold
  prompt_tokens.py     # bbox vs compact prompt token comparison (negative result)
  guided_schema.py     # guided JSON token+accuracy test (negative result)
  bench_mlx.py         # MLX decode tok/s on a real page
  mlx_quality.py       # MLX full-page speed + field accuracy vs f16 gold
models/                # quantized GGUFs + MLX model (gitignored)
```

## Environments
- `.venv` (py3.14) — `surya-ocr` + llama.cpp backend (baseline/benchmarks)
- `.venv-mlx` (uv, py3.12) — `mlx-vlm` for the MLX engine. Run MLX scripts with this one.

## Key knobs (Surya settings / env)

- `SURYA_GGUF_LOCAL_MODEL_PATH`, `SURYA_GGUF_LOCAL_MMPROJ_PATH` — run a custom GGUF (e.g. Q4)
- `SURYA_INFERENCE_PARALLEL` — llama-server slots (default 8)
- `SURYA_INFERENCE_KEEP_ALIVE` — keep the server warm across invocations
- `SURYA_MAX_TOKENS_FULL_PAGE` — decode ceiling per page

## Usage

```bash
# PRODUCTION: fast OCR a whole PDF with the MLX engine
.venv-mlx/bin/python savitr/mlx_ocr.py PATH/TO/roll.pdf

# MLX decode speed + accuracy vs f16 gold
.venv-mlx/bin/python bench/bench_mlx.py PATH/TO/roll.pdf --page 3
.venv-mlx/bin/python bench/mlx_quality.py PATH/TO/roll.pdf --pages 3-4

# timing baseline on a real PDF (llama.cpp)
python bench/bench_surya.py PATH/TO/roll.pdf --pages 3-8

# accuracy cost of a fast variant vs f16 gold
python bench/compare_quality.py run roll.pdf --pages 3-5 --out gold.json
python bench/compare_quality.py run roll.pdf --pages 3-5 \
    --model models/surya-2-Q4_K_M.gguf --mmproj <mmproj.gguf> --out q4.json
python bench/compare_quality.py cmp gold.json q4.json
```
