"""`savitr ocr` — OCR a roll PDF's pages to voter records (distilled terse model by default)."""

import argparse
import csv
import os
import sys
import time
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


def html_model_or_exit(path: str) -> str:
    """Return a local Surya MLX model dir for HTML mode, or exit (no HTML model is published)."""
    if os.path.isdir(path):
        return path
    raise SystemExit(
        f"--html needs a local Surya MLX model at {path!r}, which isn't published on the Hub.\n"
        "Drop --html to use the distilled terse model (auto-downloaded, recommended), or pass\n"
        "--mlx-path to your own MLX-converted Surya (see training/ to build one)."
    )


def main() -> int:
    """Run the `savitr ocr` command: OCR a roll PDF's pages to voter records."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf")
    ap.add_argument("--pages", default=None, help="1-based range e.g. 3-14 (default: all)")
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="write voter records to this CSV (default: print a summary)",
    )
    ap.add_argument(
        "--terse",
        action="store_true",
        help="(default) use the distilled terse-Surya model — ~5x fewer tokens, Surya accuracy",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="use the full Surya HTML model instead (needs a local --mlx-path Surya MLX model)",
    )
    ap.add_argument(
        "--mlx-path", default=None, help="override the model dir (default: the terse model)"
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
        mlx_path = html_model_or_exit(args.mlx_path or "models/surya-mlx-4bit")
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

    print(f"loading MLX model {mlx_path} (terse={terse}) ...")
    eng = MLXSuryaOCR(mlx_path, max_tokens=(2048 if terse else 8192), prompt=prompt)

    all_voters: list[dict] = []
    total_voters = 0
    t0 = time.time()
    for i in idx:
        png = f"/tmp/mlx_pdf_p{i + 1}.png"
        render_page(args.pdf, i + 1, args.dpi, png)
        t = time.time()
        text, gtok = eng.ocr_image(png)
        voters = parse(text)
        all_voters.extend(voters)
        n = len(voters)
        total_voters += n
        eg = (
            f"  e.g. {voters[0]['number']}|{voters[0]['id']}|{voters[0]['elector_name']}"
            if voters
            else ""
        )
        print(f"  page {i + 1:>3}: {gtok:>5} tok, {time.time() - t:5.1f}s, {n:>2} voters{eg}")

    elapsed = time.time() - t0
    print(
        f"\n{len(idx)} pages in {elapsed:.1f}s = {elapsed / max(len(idx), 1):.1f}s/page, "
        f"{total_voters} voters total"
    )
    if args.out:
        rows = dedupe_voters(all_voters) if terse else all_voters
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=TERSE_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} voter records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
