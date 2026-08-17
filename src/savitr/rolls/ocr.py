"""Implement ``savitr ocr`` for electoral-roll PDFs."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from savitr.rolls.parse import (
    TERSE_COLS,
    TERSE_PROMPT,
    dedupe_voters,
    parse_terse,
    parse_voters,
    resolve_terse_model,
)
from savitr.rolls.pdfio import page_count, render_page, require_poppler


def html_model_or_exit(path: str | None = None) -> str:
    """Resolve base Surya for HTML mode or exit with conversion guidance.

    The CLI uses ``SystemExit`` so users see the library's guidance without a
    traceback.
    """
    from savitr.mlx_ocr import base_model_path

    try:
        return base_model_path(path)
    except FileNotFoundError as missing:
        raise SystemExit(
            f"--html needs a converted base Surya model.\n\n{missing}\n\n"
            "Or drop --html to use the distilled terse model, which downloads itself."
        ) from missing


def main() -> int:
    """Run the `savitr ocr` command: OCR a roll PDF's pages to voter records."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf")
    ap.add_argument(
        "--pages", default=None, help="1-based range e.g. 3-14 (default: all)"
    )
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="write voter records to this CSV (default: print a summary)",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="use base Surya HTML output (requires a local MLX model)",
    )
    ap.add_argument(
        "--mlx-path",
        default=None,
        help="override the model dir (default: the terse model)",
    )
    args = ap.parse_args()
    require_poppler()  # fail fast before importing mlx or downloading the model

    from savitr.mlx_ocr import PROMPT, MLXSuryaOCR

    terse = not args.html
    parse: Callable[[str], list[dict]]
    if terse:
        mlx_path = args.mlx_path or resolve_terse_model()
        parse, prompt = parse_terse, TERSE_PROMPT
    else:
        mlx_path = html_model_or_exit(args.mlx_path)
        parse, prompt = parse_voters, PROMPT

    npages = page_count(args.pdf)  # validates poppler, then reads the page count
    if args.pages:
        idx: list[int] = []
        for part in args.pages.split(","):
            if "-" in part:
                a, b = part.split("-")
                idx += range(int(a) - 1, int(b))
            else:
                idx.append(int(part) - 1)
        idx = [i for i in idx if 0 <= i < npages]
    else:
        idx = list(range(npages))

    print(f"loading MLX model {mlx_path} (terse={terse}) ...")  # noqa: T201
    eng = MLXSuryaOCR(mlx_path, max_tokens=(2048 if terse else 8192), prompt=prompt)

    all_voters: list[dict] = []
    total_voters = 0
    t0 = time.time()
    for i in idx:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
            png = temporary.name
        t = time.time()
        try:
            render_page(args.pdf, i + 1, args.dpi, png)
            text, gtok = eng.ocr_image(png)
        finally:
            Path(png).unlink(missing_ok=True)
        voters = parse(text)
        all_voters.extend(voters)
        n = len(voters)
        total_voters += n
        eg = (
            "  e.g. "
            f"{voters[0]['number']}|{voters[0]['id']}|"
            f"{voters[0]['elector_name']}"
            if voters
            else ""
        )
        duration = time.time() - t
        print(  # noqa: T201
            f"  page {i + 1:>3}: {gtok:>5} tok, {duration:5.1f}s, {n:>2} voters{eg}"
        )

    elapsed = time.time() - t0
    seconds_per_page = elapsed / max(len(idx), 1)
    print(  # noqa: T201
        f"\n{len(idx)} pages in {elapsed:.1f}s = {seconds_per_page:.1f}s/page, "
        f"{total_voters} voters total"
    )
    if args.out:
        rows = dedupe_voters(all_voters)
        with Path(args.out).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=TERSE_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} voter records -> {args.out}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
