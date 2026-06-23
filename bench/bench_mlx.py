#!/usr/bin/env python3
"""Benchmark the MLX-converted Surya model: decode tok/s on a real page.

Compares against the llama.cpp numbers (f16 ~48 tok/s cool, Q4 ~68) to test whether the
MLX backend delivers the expected ~1.9× on Apple Silicon. Run in the isolated .venv-mlx.

Usage:
  .venv-mlx/bin/python bench/bench_mlx.py PDF --page 3 --mlx-path models/surya-mlx-4bit
"""

import argparse
import time

from pdf2image import convert_from_path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

PROMPT = "OCR this image to HTML."


def stat(res, *names):
    for n in names:
        if hasattr(res, n):
            return getattr(res, n)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--mlx-path", default="models/surya-mlx-4bit")
    ap.add_argument("--max-tokens", type=int, default=1200)
    args = ap.parse_args()

    png = "/tmp/mlx_page.png"
    convert_from_path(args.pdf, dpi=args.dpi, first_page=args.page,
                      last_page=args.page)[0].convert("RGB").save(png)

    t = time.time()
    model, processor = load(args.mlx_path)
    config = load_config(args.mlx_path)
    print(f"load: {time.time()-t:.1f}s")

    formatted = apply_chat_template(processor, config, PROMPT, num_images=1)

    # warm (Metal compile + graph build)
    t = time.time()
    generate(model, processor, formatted, image=png, max_tokens=16, verbose=False)
    print(f"warm (16 tok): {time.time()-t:.1f}s")

    # timed full generation
    t = time.time()
    res = generate(model, processor, formatted, image=png, max_tokens=args.max_tokens,
                   verbose=False)
    wall = time.time() - t

    gen_tok = stat(res, "generation_tokens", "generation_tokens_count")
    gen_tps = stat(res, "generation_tps", "tokens_per_second")
    prompt_tok = stat(res, "prompt_tokens")
    prompt_tps = stat(res, "prompt_tps")
    peak = stat(res, "peak_memory")
    text = stat(res, "text") or str(res)

    print(f"\n=== MLX result ({args.mlx_path}) ===")
    print(f"prompt tokens: {prompt_tok}  (prefill {prompt_tps} tok/s)")
    print(f"generated tokens: {gen_tok}  in {wall:.1f}s")
    print(f"DECODE: {gen_tps} tok/s" + (f"  (peak mem {peak} GB)" if peak else ""))
    print(f"\noutput head:\n{text[:500]}")


if __name__ == "__main__":
    main()
