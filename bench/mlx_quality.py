#!/usr/bin/env python3
"""MLX full-page: real per-page time + voter field accuracy vs the f16 gold fixture.

Confirms the MLX win end-to-end (not just burst tok/s) and checks the accuracy cost vs gold.
Run in .venv-mlx.

Usage:
  .venv-mlx/bin/python bench/mlx_quality.py PDF --pages 3-4 \
      --mlx-path models/surya-mlx-4bit --gold bench/fixtures/gold_f16_AC01p001_pp3-5.json
"""

import argparse
import json
import os
import sys
import time

from pdf2image import convert_from_path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_quality import extract_voters  # noqa: E402

PROMPT = "OCR this image to HTML."
FIELDS = ["name", "rel_name", "house", "age", "sex"]


def parse_pages(spec, n):
    out = []
    for part in (spec or f"1-{n}").split(","):
        if "-" in part:
            a, b = part.split("-"); out += range(int(a) - 1, int(b))
        else:
            out.append(int(part) - 1)
    return [i for i in out if 0 <= i < n]


def gold_voters(gold_json, page):
    for p in gold_json["pages"]:
        if p["page"] == page:
            return extract_voters("\n".join(b.get("html", "") for b in p["blocks"]))
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="3-4")
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--mlx-path", default="models/surya-mlx-4bit")
    ap.add_argument("--gold", default="bench/fixtures/gold_f16_AC01p001_pp3-5.json")
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    from pdf2image.pdf2image import pdfinfo_from_path
    npages = int(pdfinfo_from_path(args.pdf)["Pages"])
    idx = parse_pages(args.pages, npages)
    gold = json.load(open(args.gold))

    model, processor = load(args.mlx_path)
    config = load_config(args.mlx_path)
    formatted = apply_chat_template(processor, config, PROMPT, num_images=1)

    # warm
    p0 = "/tmp/mlx_warm.png"
    convert_from_path(args.pdf, dpi=args.dpi, first_page=idx[0] + 1,
                      last_page=idx[0] + 1)[0].convert("RGB").save(p0)
    generate(model, processor, formatted, image=p0, max_tokens=16, verbose=False)

    print(f"\n{'page':>4} {'gen_tok':>8} {'wall_s':>7} {'tok/s':>7} {'gold_v':>6} {'mlx_v':>6}  field-match(mlx vs gold)")
    agg = {f: [0, 0] for f in FIELDS}
    for i in idx:
        png = f"/tmp/mlx_p{i+1}.png"
        convert_from_path(args.pdf, dpi=args.dpi, first_page=i + 1,
                          last_page=i + 1)[0].convert("RGB").save(png)
        t = time.time()
        res = generate(model, processor, formatted, image=png, max_tokens=args.max_tokens,
                       verbose=False)
        wall = time.time() - t
        text = getattr(res, "text", None) or str(res)
        gtok = getattr(res, "generation_tokens", None) or 0
        tps = getattr(res, "generation_tps", None) or (gtok / wall if wall else 0)

        gv = gold_voters(gold, i + 1)
        mv = extract_voters(text)
        per = {f: [0, 0] for f in FIELDS}
        for k in set(gv) & set(mv):
            for f in FIELDS:
                a, b = gv[k][f].lower().strip(), mv[k][f].lower().strip()
                if a:
                    per[f][1] += 1; agg[f][1] += 1
                    hit = (a == b) or (f in ("name", "rel_name") and (a in b or b in a))
                    per[f][0] += hit; agg[f][0] += hit
        fm = " ".join(f"{f[:4]}={per[f][0]}/{per[f][1]}" for f in FIELDS)
        print(f"{i+1:>4} {gtok:>8} {wall:>7.1f} {tps:>7.1f} {len(gv):>6} {len(mv):>6}  {fm}")

    print(f"\n  field-match MLX-4bit vs f16 gold (over fields gold has):")
    for f in FIELDS:
        m, n = agg[f]
        print(f"    {f:<9} {m}/{n}  {100*m/n if n else 0:.1f}%")


if __name__ == "__main__":
    main()
