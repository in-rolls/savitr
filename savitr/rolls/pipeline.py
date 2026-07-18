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
from typing import TYPE_CHECKING

from savitr.rolls.parse import (
    TERSE_PROMPT,
    dedupe_voters,
    parse_terse,
    parse_voters,
    resolve_terse_model,
)
from savitr.rolls.pdfio import page_count, render_page, require_poppler

if TYPE_CHECKING:
    from savitr.mlx_ocr import MLXSuryaOCR

# Roll schema + cover-field logic. Default to savitr's vendored copy (self-contained); set
# MANIPUR_DIR to point at parse_unsearchable_rolls/scripts/manipur to use that repo's copy instead.
MANIPUR_DIR = os.environ.get("MANIPUR_DIR")
if MANIPUR_DIR:
    sys.path.insert(0, MANIPUR_DIR)
    import fields  # noqa: E402
    from parse_manipur_2025 import COLUMNS, STATE, YEAR, ac_part_from_filename  # noqa: E402
else:
    from savitr.rolls import fields
    from savitr.rolls.schema import COLUMNS, STATE, YEAR, ac_part_from_filename

log = logging.getLogger("parse_manipur_mlx")

_BREAK = re.compile(r"</(tr|p|td|th|div)>|<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(html: str) -> str:
    """HTML -> newline-separated text (so fields.py's cover regexes see line structure)."""
    t = _BREAK.sub("\n", html)
    t = _TAG.sub(" ", t)
    return "\n".join(ln.strip() for ln in t.splitlines() if ln.strip())


def _synthetic_page(text: str) -> dict:
    """Build a minimal page dict that fields.page_text / parse_cover_page can consume."""
    return {
        "lines": [{"text": ln, "cx": 0, "cy": i * 10} for i, ln in enumerate(text.splitlines())]
    }


def parse_pdf_mlx(
    eng: "MLXSuryaOCR",
    pdf_path: str,
    dpi: int,
    terse: bool = True,
    cover_eng: "MLXSuryaOCR | None" = None,
) -> tuple[list[dict], dict]:
    """OCR + parse one PDF with MLX; return (rows, recon) in the canonical schema.

    terse mode (default): the terse model ``eng`` reads every page for voters (covers yield none),
    and — only if ``cover_eng`` is provided — the HTML model reads the first two pages for cover
    metadata. Without a cover model, voter extraction still works; cover metadata is just skipped
    and the AC/part number is taken from the filename.

    HTML mode (``terse=False``): the HTML model ``eng`` reads every page for voters + metadata.
    """
    fname = os.path.basename(pdf_path)
    ac_no, part_no = ac_part_from_filename(fname)
    npages = page_count(pdf_path)

    page_texts = []
    voters = []
    for i in range(npages):
        png = render_page(pdf_path, i + 1, dpi, f"/tmp/mlx_{os.getpid()}_p{i + 1}.png")
        if terse:  # terse model reads voters from every page (covers simply yield none)
            voters.extend(parse_terse(eng.ocr_image(png)[0]))
            if cover_eng is not None and i < 2:  # optional cover metadata
                page_texts.append(html_to_text(cover_eng.ocr_image(png)[0]))
        else:  # HTML mode: one model does voters + metadata
            html, _ = eng.ocr_image(png)
            page_texts.append(html_to_text(html))
            voters.extend(parse_voters(html))
        os.remove(png)

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

    recon = {
        "filename": fname,
        "voters_extracted": len(voters),
        "net_electors_total": header["net_electors_total"],
    }
    return rows, recon


def main() -> int:
    """Run the `savitr parse-rolls` command: roll PDFs -> canonical voter CSV."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-f", "--file", help="a single PDF to parse")
    g.add_argument("-d", "--dir", help="a directory of *_ENG.pdf files")
    ap.add_argument("-o", "--out", required=True, help="output CSV path")
    ap.add_argument(
        "--terse",
        action="store_true",
        help="(default) use the distilled terse-Surya voter model (~2.7x faster, Surya accuracy)",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="use the full Surya HTML model instead (needs a local --mlx-path Surya MLX model)",
    )
    ap.add_argument(
        "--mlx-path", default=None, help="override the voter model dir (default: the terse model)"
    )
    ap.add_argument(
        "--cover-model",
        default="models/surya-mlx-4bit",
        help="local Surya MLX model for cover-page metadata (optional; skipped if absent)",
    )
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--resume", action="store_true", help="skip PDFs already present in the output CSV"
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    pdfs = [args.file] if args.file else sorted(glob.glob(os.path.join(args.dir, "*_ENG.pdf")))
    if args.limit:
        pdfs = pdfs[: args.limit]

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="", encoding="utf-8") as fh:
            done = {r["filename"] for r in csv.DictReader(fh)}

    require_poppler()  # fail fast with a friendly hint before loading a model
    from savitr.mlx_ocr import MLXSuryaOCR
    from savitr.rolls.ocr import html_model_or_exit

    terse = not args.html
    cover_eng = None
    if terse:
        voter_path = args.mlx_path or resolve_terse_model()
        eng = MLXSuryaOCR(voter_path, max_tokens=2048, prompt=TERSE_PROMPT)
        if os.path.isdir(args.cover_model):
            log.info("terse voter model %s + cover model %s", voter_path, args.cover_model)
            cover_eng = MLXSuryaOCR(args.cover_model, max_tokens=8192)
        else:
            log.info(
                "terse voter model %s; no cover model at %s -> cover metadata skipped "
                "(AC/part from filename; pass --cover-model for full metadata)",
                voter_path,
                args.cover_model,
            )
    else:
        html_path = html_model_or_exit(args.mlx_path or "models/surya-mlx-4bit")
        log.info("loading Surya HTML model %s ...", html_path)
        eng = MLXSuryaOCR(html_path)

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
                rows, recon = parse_pdf_mlx(eng, pdf, args.dpi, terse=terse, cover_eng=cover_eng)
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
                    flag = f"  <-- COUNT MISMATCH (net={net})"
            log.info(
                "[%d/%d] %s: %d voters (net=%s) %.0fs%s",
                i,
                len(pdfs),
                os.path.basename(pdf),
                recon["voters_extracted"],
                net or "?",
                time.time() - t,
                flag,
            )

    log.info("done: %d voters from %d PDFs -> %s", total_voters, len(pdfs), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
