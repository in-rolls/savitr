#!/usr/bin/env python3
"""Parse Manipur 2025 rolls into the canonical voter CSV — using the fast MLX OCR engine.

Drop-in alternative to the repo's RapidOCR-based `parse_manipur_2025.py`, but ~5× faster and
higher fidelity: it OCRs each page with MLX Surya (~34 s/page), parses voters layout-robustly,
and reuses the existing `fields.py` (cover-page metadata + serial dedup) and column schema so
the output is byte-compatible with the rest of the pipeline.

Run in the MLX env:
  .venv-mlx/bin/python savitr/parse_manipur_mlx.py -f AC01_part001_final_ENG.pdf -o out.csv
  .venv-mlx/bin/python savitr/parse_manipur_mlx.py -d english/ -o out.csv [--limit N] [--resume]
"""

import argparse
import csv
import glob
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_ocr import MLXSuryaOCR, parse_voters, dedupe_voters  # noqa: E402

# Reuse the repo's field logic + schema (pure-stdlib; no engine import side effects).
MANIPUR_DIR = os.environ.get(
    "MANIPUR_DIR", "/Users/soodoku/Documents/GitHub/parse_unsearchable_rolls/scripts/manipur")
sys.path.insert(0, MANIPUR_DIR)
import fields  # noqa: E402
from parse_manipur_2025 import COLUMNS, ac_part_from_filename, STATE, YEAR  # noqa: E402

log = logging.getLogger("parse_manipur_mlx")

_BREAK = re.compile(r"</(tr|p|td|th|div)>|<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(html):
    """HTML -> newline-separated text (so fields.py's cover regexes see line structure)."""
    t = _BREAK.sub("\n", html)
    t = _TAG.sub(" ", t)
    return "\n".join(ln.strip() for ln in t.splitlines() if ln.strip())


def _synthetic_page(text):
    """A minimal page dict fields.page_text / parse_cover_page can consume."""
    return {"lines": [{"text": ln, "cx": 0, "cy": i * 10}
                      for i, ln in enumerate(text.splitlines())]}


def parse_pdf_mlx(eng, pdf_path, dpi):
    """OCR + parse one PDF with MLX; return (rows, recon) in the canonical schema."""
    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path

    fname = os.path.basename(pdf_path)
    ac_no, part_no = ac_part_from_filename(fname)
    npages = int(pdfinfo_from_path(pdf_path)["Pages"])

    page_texts = []
    voters = []
    for i in range(npages):
        png = f"/tmp/mlx_{os.getpid()}_p{i+1}.png"
        convert_from_path(pdf_path, dpi=dpi, first_page=i + 1, last_page=i + 1)[0] \
            .convert("RGB").save(png)
        html, _ = eng.ocr_image(png)
        os.remove(png)
        page_texts.append(html_to_text(html))
        voters.extend(parse_voters(html))

    # cover page = first page with the elector summary / "ELECTORAL ROLL" header
    meta = {}
    for text in page_texts[:2]:
        flat = re.sub(r"[^A-Z]", "", text.upper())
        if "ELECTORALROLL" in flat or "NUMBEROFELECTORS" in flat:
            meta = fields.parse_cover_page(_synthetic_page(text))
            break

    voters = dedupe_voters(voters)

    header = {
        "ac_name": meta.get("ac_name", "") or (ac_no if ac_no else ""),
        "parl_constituency": meta.get("parl_constituency", ""),
        "part_no": meta.get("part_no") or part_no,
        "year": meta.get("year") or YEAR,
        "state": STATE,
        "filename": fname,
        "main_town": meta.get("main_town", ""),
        "police_station": meta.get("police_station", ""),
        "mandal": "",
        "revenue_division": "",
        "district": meta.get("district", ""),
        "pin_code": meta.get("pin_code", ""),
        "polling_station_name": meta.get("polling_station_name", ""),
        "polling_station_address": meta.get("polling_station_address", ""),
        "net_electors_male": meta.get("net_electors_male", ""),
        "net_electors_female": meta.get("net_electors_female", ""),
        "net_electors_third_gender": meta.get("net_electors_third_gender", ""),
        "net_electors_total": meta.get("net_electors_total", ""),
    }

    rows = []
    for v in voters:
        row = {c: "" for c in COLUMNS}
        row.update(header)
        row.update(v)
        rows.append(row)

    recon = {"filename": fname, "voters_extracted": len(voters),
             "net_electors_total": header["net_electors_total"]}
    return rows, recon


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-f", "--file", help="a single PDF to parse")
    g.add_argument("-d", "--dir", help="a directory of *_ENG.pdf files")
    ap.add_argument("-o", "--out", required=True, help="output CSV path")
    ap.add_argument("--mlx-path", default="models/surya-mlx-4bit")
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip PDFs already present in the output CSV")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    pdfs = [args.file] if args.file else sorted(glob.glob(os.path.join(args.dir, "*_ENG.pdf")))
    if args.limit:
        pdfs = pdfs[:args.limit]

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="", encoding="utf-8") as fh:
            done = {r["filename"] for r in csv.DictReader(fh)}

    log.info("loading MLX model %s ...", args.mlx_path)
    eng = MLXSuryaOCR(args.mlx_path)

    write_header = not (args.resume and os.path.exists(args.out))
    total_voters = 0
    with open(args.out, "a" if args.resume else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for i, pdf in enumerate(pdfs, 1):
            if os.path.basename(pdf) in done:
                log.info("[%d/%d] skip (done) %s", i, len(pdfs), os.path.basename(pdf))
                continue
            t = time.time()
            try:
                rows, recon = parse_pdf_mlx(eng, pdf, args.dpi)
            except Exception as exc:  # noqa: BLE001 - one bad PDF must not kill the run
                log.exception("FAILED %s: %s", os.path.basename(pdf), exc)
                continue
            writer.writerows(rows)
            fh.flush()
            total_voters += recon["voters_extracted"]
            net = recon["net_electors_total"]
            flag = ""
            if net and net.isdigit():
                diff = recon["voters_extracted"] - int(net)
                if abs(diff) > max(3, 0.02 * int(net)):
                    flag = "  <-- COUNT MISMATCH (net=%s)" % net
            log.info("[%d/%d] %s: %d voters (net=%s) %.0fs%s", i, len(pdfs),
                     os.path.basename(pdf), recon["voters_extracted"], net or "?",
                     time.time() - t, flag)

    log.info("done: %d voters from %d PDFs -> %s", total_voters, len(pdfs), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
