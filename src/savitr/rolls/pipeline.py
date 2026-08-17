"""Parse Manipur 2025 rolls into the canonical voter CSV.

The pipeline OCRs each page with MLX Surya, parses voter records, and applies
the in-rolls column schema.

Use ``savitr parse-rolls --help`` for command-line options.
"""

import argparse
import csv
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from savitr.rolls import fields
from savitr.rolls.parse import (
    TERSE_PROMPT,
    dedupe_voters,
    parse_terse,
    parse_voters,
    resolve_terse_model,
)
from savitr.rolls.pdfio import page_count, render_page, require_poppler
from savitr.rolls.schema import COLUMNS, STATE, YEAR, ac_part_from_filename

if TYPE_CHECKING:
    from savitr.mlx_ocr import MLXSuryaOCR

log = logging.getLogger("parse_manipur_mlx")

_BREAK = re.compile(r"</(tr|p|td|th|div)>|<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(html: str) -> str:
    """Convert HTML to newline-separated text for cover-field parsing."""
    t = _BREAK.sub("\n", html)
    t = _TAG.sub(" ", t)
    return "\n".join(ln.strip() for ln in t.splitlines() if ln.strip())


def _synthetic_page(text: str) -> dict:
    """Build the minimal page object required by the cover parser."""
    return {
        "lines": [
            {"text": ln, "cx": 0, "cy": i * 10}
            for i, ln in enumerate(text.splitlines())
        ]
    }


def parse_pdf_mlx(
    eng: "MLXSuryaOCR",
    pdf_path: str,
    dpi: int,
    terse: bool = True,
    cover_eng: "MLXSuryaOCR | None" = None,
) -> tuple[list[dict], dict]:
    """OCR + parse one PDF with MLX; return (rows, recon) in the canonical schema.

    In terse mode, ``eng`` reads voters from every page. If ``cover_eng`` is
    provided, it reads the first two pages for metadata.

    In HTML mode, ``eng`` reads both voters and metadata.
    """
    fname = Path(pdf_path).name
    ac_no, part_no = ac_part_from_filename(fname)
    npages = page_count(pdf_path)

    page_texts = []
    voters = []
    for i in range(npages):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
            png = temporary.name
        try:
            render_page(pdf_path, i + 1, dpi, png)
            if terse:
                voters.extend(parse_terse(eng.ocr_image(png)[0]))
                if cover_eng is not None and i < 2:
                    page_texts.append(html_to_text(cover_eng.ocr_image(png)[0]))
            else:
                html, _ = eng.ocr_image(png)
                page_texts.append(html_to_text(html))
                voters.extend(parse_voters(html))
        finally:
            Path(png).unlink(missing_ok=True)

    # cover page = first page with the elector summary / "ELECTORAL ROLL" header
    meta = {}
    for text in page_texts[:2]:
        flat = re.sub(r"[^A-Z]", "", text.upper())
        if "ELECTORALROLL" in flat or "NUMBEROFELECTORS" in flat:
            meta = fields.parse_cover_page(_synthetic_page(text))
            break

    voters = dedupe_voters(voters)

    header = {
        "ac_name": meta.get("ac_name", "") or (ac_no or ""),
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
        row = dict.fromkeys(COLUMNS, "")
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
        "--html",
        action="store_true",
        help="use base Surya HTML output (requires a local MLX model)",
    )
    ap.add_argument(
        "--mlx-path",
        default=None,
        help="override the voter model dir (default: the terse model)",
    )
    ap.add_argument(
        "--cover-model",
        default="models/surya-mlx-4bit",
        help="optional base Surya model for cover-page metadata",
    )
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip PDFs already present in the output CSV",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    pdfs = (
        [args.file]
        if args.file
        else [str(path) for path in sorted(Path(args.dir).glob("*_ENG.pdf"))]
    )
    if args.limit:
        pdfs = pdfs[: args.limit]

    out_path = Path(args.out)
    done = set()
    if args.resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as fh:
            done = {r["filename"] for r in csv.DictReader(fh)}

    require_poppler()  # fail fast with a friendly hint before loading a model
    from savitr.mlx_ocr import MLXSuryaOCR, base_model_path
    from savitr.rolls.ocr import html_model_or_exit

    terse = not args.html
    cover_eng = None
    if terse:
        voter_path = args.mlx_path or resolve_terse_model()
        eng = MLXSuryaOCR(voter_path, max_tokens=2048, prompt=TERSE_PROMPT)
        try:
            cover_path = base_model_path(args.cover_model)
            log.info("terse voter model %s + cover model %s", voter_path, cover_path)
            cover_eng = MLXSuryaOCR(cover_path, max_tokens=8192)
        except FileNotFoundError as missing:
            log.info(
                "terse model %s; cover metadata skipped (filename fallback).\n%s",
                voter_path,
                missing,
            )
    else:
        html_path = html_model_or_exit(args.mlx_path)
        log.info("loading Surya HTML model %s ...", html_path)
        eng = MLXSuryaOCR(html_path)

    write_header = not (args.resume and out_path.exists())
    total_voters = 0
    with out_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for i, pdf in enumerate(pdfs, 1):
            pdf_name = Path(pdf).name
            if pdf_name in done:
                log.info("[%d/%d] skip (done) %s", i, len(pdfs), pdf_name)
                continue
            t = time.time()
            try:
                rows, recon = parse_pdf_mlx(
                    eng, pdf, args.dpi, terse=terse, cover_eng=cover_eng
                )
            except Exception as exc:
                log.exception("FAILED %s: %s", pdf_name, exc)
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
                pdf_name,
                recon["voters_extracted"],
                net or "?",
                time.time() - t,
                flag,
            )

    log.info("done: %d voters from %d PDFs -> %s", total_voters, len(pdfs), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
