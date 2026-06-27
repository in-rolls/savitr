"""`savitr ocr` — OCR a roll PDF's pages to voter records (HTML model or distilled terse model)."""

import argparse
import sys
import time

from savitr.mlx_ocr import PROMPT, MLXSuryaOCR
from savitr.rolls.parse import TERSE_PROMPT, parse_terse, parse_voters, resolve_terse_model


def main() -> int:
    """Run the `savitr ocr` command: OCR a roll PDF's pages to voter records."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf")
    ap.add_argument("--pages", default=None, help="1-based range e.g. 3-14 (default: all)")
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument(
        "--terse",
        action="store_true",
        help="use the distilled terse-Surya model (~5x fewer tokens, Surya accuracy)",
    )
    ap.add_argument(
        "--mlx-path", default=None, help="default: terse model with --terse, else surya-mlx-4bit"
    )
    args = ap.parse_args()

    mlx_path = args.mlx_path or (resolve_terse_model() if args.terse else "models/surya-mlx-4bit")
    parse = parse_terse if args.terse else parse_voters
    prompt = TERSE_PROMPT if args.terse else PROMPT

    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path

    npages = int(pdfinfo_from_path(args.pdf)["Pages"])
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

    print(f"loading MLX model {mlx_path} (terse={args.terse}) ...")
    eng = MLXSuryaOCR(mlx_path, max_tokens=(2048 if args.terse else 8192), prompt=prompt)

    total_voters = 0
    t0 = time.time()
    for i in idx:
        png = f"/tmp/mlx_pdf_p{i + 1}.png"
        convert_from_path(args.pdf, dpi=args.dpi, first_page=i + 1, last_page=i + 1)[0].convert(
            "RGB"
        ).save(png)
        t = time.time()
        text, gtok = eng.ocr_image(png)
        voters = parse(text)
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
        f"\n{len(idx)} pages in {elapsed:.1f}s = {elapsed / len(idx):.1f}s/page, "
        f"{total_voters} voters total"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
