#!/usr/bin/env python3
"""Render a diverse sample of roll pages to PNGs for the Kaggle training dataset.

Rendering only (no OCR) — fast (~0.2 s/page). The Surya teacher labels these on the GPU (where
it's ~100× faster than the Mac). Output `pages/` becomes a Kaggle Dataset; the training kernel
OCRs + fine-tunes from it.

  .venv-mlx/bin/python savitr/distill/prep_pages.py --n 1000 --out pages
"""

import argparse
import glob
import os
import random
import sys
import time

DEFAULT_PDF_DIR = "/Users/soodoku/Documents/GitHub/electoral_rolls/manipur/2025/pdfs/english"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    ap.add_argument("--out", default="pages")
    ap.add_argument("--n", type=int, default=1000, help="pages to render")
    ap.add_argument("--pages-per-pdf", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path

    pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*_ENG.pdf")))
    random.Random(args.seed).shuffle(pdfs)
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(pdfs)} PDFs; rendering {args.n} diverse pages @ {args.dpi}dpi -> {args.out}/")

    done = 0
    t0 = time.time()
    for pdf in pdfs:
        if done >= args.n:
            break
        try:
            npages = int(pdfinfo_from_path(pdf)["Pages"])
        except Exception:
            continue
        interior = list(range(3, npages))  # skip cover(1) + maps(2)
        random.Random(args.seed + hash(pdf) % 1000).shuffle(interior)
        stem = os.path.splitext(os.path.basename(pdf))[0]
        for pg in interior[: args.pages_per_pdf]:
            if done >= args.n:
                break
            out_png = os.path.join(args.out, f"{stem}_p{pg}.png")
            if os.path.exists(out_png):
                done += 1
                continue
            try:
                convert_from_path(pdf, dpi=args.dpi, first_page=pg, last_page=pg)[0].convert(
                    "RGB"
                ).save(out_png)
            except Exception as e:
                print(f"  skip {stem} p{pg}: {e}")
                continue
            done += 1
            if done % 50 == 0:
                print(f"  rendered {done}/{args.n} ({(time.time()-t0)/60:.1f} min)")

    sz = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 1e6
    print(f"\nDONE: {done} pages -> {args.out}/ ({sz:.0f} MB, {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    sys.exit(main())
