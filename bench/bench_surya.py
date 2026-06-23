#!/usr/bin/env python3
"""Benchmark Surya OCR on real Manipur roll pages, the warm in-process way.

Establishes the true baseline the speedup work is measured against, and dumps the
actual full-page output structure so we can see the block granularity (per-line vs
per-region) that fields.py has to consume.

It separates the three costs the per-PDF CLI conflates:
  - load: build SuryaInferenceManager + RecognitionPredictor (spawn llama.cpp, load model)
  - cold: first page (Metal shader compile + server warmup)
  - warm: steady-state s/page, measured by OCR-ing a batch of N pages in one call

Usage:
  python bench_surya.py PDF [--pages 1-6] [--dpi 192] [--dump structure.json]
"""

import argparse
import json
import time

from pdf2image import convert_from_path


def parse_page_range(spec, n):
    """'1-6' / '0,3,5' -> 0-based page indices, clamped to [0, n)."""
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a) - 1, int(b)))  # 1-based inclusive
        else:
            out.append(int(part) - 1)
    return [i for i in out if 0 <= i < n]


def render(pdf_path, dpi, indices):
    """Render only the requested pages to PIL images."""
    imgs = []
    for i in indices:
        page = convert_from_path(pdf_path, dpi=dpi, first_page=i + 1, last_page=i + 1)[0]
        imgs.append(page)
    return imgs


def block_summary(page_result):
    """Compact view of one PageOCRResult: per-block label, bbox, text length."""
    blocks = []
    for b in page_result.blocks:
        html = (b.html or "").strip()
        text = html
        # crude tag strip just for length/preview
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        blocks.append({
            "label": b.label,
            "bbox": [round(x, 1) for x in b.bbox],
            "reading_order": b.reading_order,
            "text_len": len(text),
            "preview": text[:120],
            "html": b.html or "",
        })
    return {"image_bbox": page_result.image_bbox, "n_blocks": len(blocks), "blocks": blocks}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", help="a real roll PDF to benchmark")
    ap.add_argument("--pages", default="1-6", help="1-based page range, e.g. 1-6 or 1,3,5")
    ap.add_argument("--dpi", type=int, default=192, help="render DPI (Surya highres default)")
    ap.add_argument("--dump", default="bench_structure.json",
                    help="write the full-page block structure of the pages here")
    args = ap.parse_args()

    from pdf2image.pdf2image import pdfinfo_from_path
    npages = int(pdfinfo_from_path(args.pdf)["Pages"])
    indices = parse_page_range(args.pages, npages)
    print(f"PDF: {args.pdf}  ({npages} pages)  benchmarking pages {[i+1 for i in indices]}")

    t = time.time()
    images = render(args.pdf, args.dpi, indices)
    print(f"render: {time.time()-t:.1f}s for {len(images)} pages @ {args.dpi}dpi "
          f"({images[0].width}x{images[0].height})")

    # ---- load: construct manager + predictor (spawns server, loads model) ----
    t = time.time()
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor
    manager = SuryaInferenceManager()
    rec = RecognitionPredictor(manager)
    print(f"load (import + manager + predictor build): {time.time()-t:.1f}s")

    # ---- cold: first page (server warmup + Metal shader compile) ----
    t = time.time()
    first = rec([images[0]], full_page=True)
    cold = time.time() - t
    print(f"cold (1st page, includes server start + Metal compile): {cold:.1f}s")

    # ---- warm: remaining pages in one batched call ----
    rest = images[1:]
    warm_per_page = None
    if rest:
        t = time.time()
        warm_results = rec(rest, full_page=True)
        warm_total = time.time() - t
        warm_per_page = warm_total / len(rest)
        print(f"warm: {warm_total:.1f}s for {len(rest)} pages "
              f"= {warm_per_page:.1f}s/page (batched, steady-state)")
    else:
        warm_results = []

    # ---- dump structure for inspection (block granularity question) ----
    all_results = [first[0]] + list(warm_results)
    dump = {
        "pdf": args.pdf,
        "dpi": args.dpi,
        "image_size": [images[0].width, images[0].height],
        "timing": {"cold_s": round(cold, 1),
                   "warm_s_per_page": round(warm_per_page, 1) if warm_per_page else None},
        "pages": [block_summary(r) for r in all_results],
    }
    with open(args.dump, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"wrote block structure -> {args.dump}")

    # quick console peek at page 1 blocks
    s = dump["pages"][0]
    print(f"\npage 1: {s['n_blocks']} blocks")
    for b in s["blocks"][:12]:
        print(f"  [{b['label']:<10}] bbox={b['bbox']} len={b['text_len']:<4} {b['preview']!r}")

    try:
        manager.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
