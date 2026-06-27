#!/usr/bin/env python3
"""Build the self-distillation corpus: teacher-OCR roll pages -> clean terse targets.

Samples interior voter pages diversely across many PDFs, OCRs each with the current MLX Surya
(teacher), parses to clean voter records, quality-filters, and writes an mlx-vlm training
dataset (image PNG + terse target). We train on the PARSER's cleaned output (not raw HTML):
parse_voters/dedupe_voters already de-loop and fix layout-B, so kept targets are clean.

Output (LOCAL ONLY — real-voter PII, gitignored): corpus/images/*.png + corpus/dataset.jsonl
Each line: {"image": "<abs png>", "messages":[{user: TERSE_PROMPT}, {assistant: terse rows}]}

Usage (validate small first, then full):
  .venv-mlx/bin/python savitr/distill/build_corpus.py --target 5     --out corpus_smoke
  .venv-mlx/bin/python savitr/distill/build_corpus.py --target 400   --out corpus
"""

import argparse
import json
import os
import random
import sys
import time

from savitr.mlx_ocr import MLXSuryaOCR
from savitr.rolls.parse import TERSE_PROMPT, dedupe_voters, parse_voters, to_terse

DEFAULT_PDF_DIR = "/Users/soodoku/Documents/GitHub/electoral_rolls/manipur/2025/pdfs/english"


def is_clean(voters):
    """Keep only pages the teacher OCR'd cleanly (so the student learns good behavior)."""
    n = len(voters)
    if not (20 <= n <= 35):
        return False
    epics = [v["id"] for v in voters if v.get("id")]
    if len(epics) < 0.9 * n:  # ≥90% have an EPIC
        return False
    if len(set(epics)) != len(epics):  # no duplicate EPICs (mis-assoc/loop)
        return False
    good = sum(1 for v in voters if v.get("age") and v.get("sex") and v.get("elector_name"))
    return good >= 0.9 * n  # fields mostly complete


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    ap.add_argument("--out", default="corpus", help="output dir (gitignored)")
    ap.add_argument("--target", type=int, default=450, help="target clean pages")
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=0.12,
        help="fraction of ACs reserved ENTIRELY for the OOS test set (true generalization)",
    )
    ap.add_argument("--pages-per-pdf", type=int, default=2, help="interior pages to try per PDF")
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--mlx-path", default="models/surya-mlx-4bit")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import glob
    import re as _re

    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path

    pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*_ENG.pdf")))
    random.Random(args.seed).shuffle(pdfs)  # diverse, deterministic order

    def _ac(path):  # AC id from filename, e.g. AC07_part012_... -> AC07
        m = _re.match(r"(AC\d+)", os.path.basename(path))
        return m.group(1) if m else "AC00"

    acs = sorted({_ac(p) for p in pdfs})  # whole-AC OOS holdout (deterministic by seed)
    n_hold = max(1, round(len(acs) * args.holdout_frac))
    test_acs = set(random.Random(args.seed + 1).sample(acs, n_hold))
    print(f"{len(pdfs)} PDFs, {len(acs)} ACs; targeting {args.target} clean pages -> {args.out}/")
    print(f"OOS held-out ACs ({n_hold}): {sorted(test_acs)}")

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    train_path = os.path.join(args.out, "dataset_train.jsonl")
    test_path = os.path.join(args.out, "dataset_test.jsonl")

    # resume: append, skipping pages already in the corpus
    done = set()
    n_test = 0
    for p, is_test in ((train_path, False), (test_path, True)):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["image"])
                    n_test += is_test
                except Exception:
                    pass
    kept = len(done)
    if kept:
        print(f"resuming: {kept} existing pages ({n_test} test)")
    train_f = open(train_path, "a", encoding="utf-8")
    test_f = open(test_path, "a", encoding="utf-8")

    eng = MLXSuryaOCR(args.mlx_path)
    tried = 0
    t0 = time.time()
    try:
        for pdf in pdfs:
            if kept >= args.target:
                break
            try:
                npages = int(pdfinfo_from_path(pdf)["Pages"])
            except Exception:
                continue
            # interior voter pages: skip cover(1) + maps(2) and the last page
            interior = list(range(3, npages))
            random.Random(args.seed + hash(pdf) % 1000).shuffle(interior)
            for pg in interior[: args.pages_per_pdf]:
                if kept >= args.target:
                    break
                stem = os.path.splitext(os.path.basename(pdf))[0]
                png = os.path.abspath(os.path.join(img_dir, f"{stem}_p{pg}.png"))
                if png in done:
                    continue
                tried += 1
                try:
                    convert_from_path(pdf, dpi=args.dpi, first_page=pg, last_page=pg)[0].convert(
                        "RGB"
                    ).save(png)
                    html, _ = eng.ocr_image(png)
                    voters = dedupe_voters(parse_voters(html))
                except Exception as e:
                    print(f"  skip {stem} p{pg}: {e}")
                    continue
                if not is_clean(voters):
                    os.remove(png)
                    continue
                rec = (
                    json.dumps(
                        {
                            "image": png,
                            "messages": [
                                {"role": "user", "content": TERSE_PROMPT},
                                {"role": "assistant", "content": to_terse(voters)},
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # whole-AC OOS split: pages from held-out ACs -> test, never trained on
                if _ac(pdf) in test_acs:
                    test_f.write(rec)
                    n_test += 1
                else:
                    train_f.write(rec)
                train_f.flush()
                test_f.flush()
                kept += 1
                if kept % 10 == 0 or kept <= 5:
                    rate = kept / max(tried, 1)
                    print(
                        f"  kept {kept}/{args.target} (tried {tried}, {rate:.0%} clean, "
                        f"test {n_test}, {(time.time()-t0)/60:.1f} min)"
                    )
    finally:
        train_f.close()
        test_f.close()

    print(
        f"\nDONE: {kept} clean pages from {tried} tried "
        f"(train {kept-n_test} / test {n_test}) -> {args.out}/ ({(time.time()-t0)/60:.1f} min)"
    )


if __name__ == "__main__":
    sys.exit(main())
