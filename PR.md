# PR: Add an MLX inference backend (Apple Silicon, ~3.6× faster than llama.cpp)

## Motivation
Surya serves the VLM via `vllm` (NVIDIA) or `llama.cpp` (CPU/Apple Silicon). On Apple
Silicon, MLX is substantially faster than llama.cpp for this 650M `qwen3_5` model. Measured
on an M4/16 GB (`surya-ocr-2`, real document pages):

| backend | decode tok/s | per-page (full voter page) |
|---|---|---|
| llama.cpp f16 | 48 | 177.8 s |
| llama.cpp Q4 | 68 | — |
| **MLX 4-bit** | **~178** | **~31–34 s (~3.6–5.2×)** |

…at <2 GB RAM, ~97–100% field fidelity vs the f16 output. mlx-vlm already supports the
`qwen3_5` architecture and ships an OpenAI-compatible server, so the backend reuses Surya's
existing `chat_completions_batch` path with no pipeline changes.

## Validation
`SuryaInferenceManager(method="mlx")` + `RecognitionPredictor(...)(images, full_page=True)`
runs the full pipeline (layout → blocks → parse) end-to-end on the MLX server. Proven on real
pages: 8 layout blocks parsed, voter `<table>` correct.

## Changes (3 small touch-points + 1 new file)

**1. New file `surya/inference/backends/mlx.py`** — the body of `savitr/mlx_backend.py`
in this repo (a `Backend` subclass that spawns the mlx-vlm OpenAI server, attaches an
`OpenAI` client, and delegates to `chat_completions_batch`). Mirrors `LlamaCppBackend`.

**2. `surya/inference/__init__.py` — `_build_backend`:**
```python
    if method == "mlx":
        from surya.inference.backends.mlx import MlxBackend
        return MlxBackend()
```

**3. `_autodetect_backend()`** (optional) — prefer MLX on Apple Silicon when available:
```python
    if torch.backends.mps.is_available() and importlib.util.find_spec("mlx_vlm"):
        return "mlx"
    return "llamacpp"
```

**4. `surya/settings.py`** — add:
```python
    SURYA_MLX_MODEL_PATH: Optional[str] = None   # converted MLX model dir
    SURYA_MLX_PYTHON: Optional[str] = None        # python env that has mlx-vlm
    SURYA_MLX_PORT: Optional[int] = None
```
(This repo's module reads them from `os.environ` so it works without a fork.)

## Setup (user side)
```bash
pip install mlx-vlm
python -m mlx_vlm convert --hf-path datalab-to/surya-ocr-2 \
    --mlx-path surya-mlx-4bit -q --q-bits 4        # one-time, ~500 MB
export SURYA_MLX_MODEL_PATH=surya-mlx-4bit
surya_ocr doc.pdf                                   # now runs on MLX
```

## Notes
- The mlx-vlm server runs as its own subprocess, so its deps never conflict with Surya's.
- mlx-vlm's server also exposes vision-feature caching and `--draft-model` (speculative
  decoding) — future follow-ups for more speedup.
- Concurrency: MLX/Metal is memory-bandwidth bound; high `SURYA_INFERENCE_PARALLEL` on dense
  pages thrashes. Default kept low (2).
