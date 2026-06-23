#!/usr/bin/env python3
"""Speed vs. accuracy: measure how far a *faster* Surya variant drifts from the f16 gold.

The "gold standard" is the unmodified f16 Surya pipeline output. Each speed optimization
(Q4 quant, lower DPI, MLX, distilled model, ...) is scored by how much its output diverges
from that gold — no hand-labeling needed.

Two modes:

  # 1) run the real Surya pipeline once with a given model, dump per-page text + timing
  python compare_quality.py run  PDF --pages 3-8 --dpi 192 \
      --model /tmp/surya-2-Q4_K_M.gguf --mmproj <mmproj.gguf> --out q4.json
  #   (omit --model to use the default f16 model from the HF cache)

  # 2) compare a variant against the gold dump
  python compare_quality.py cmp  gold.json q4.json

`run` must set SURYA_GGUF_LOCAL_MODEL_PATH *before* importing surya, so it runs as its own
process (the llama.cpp server loads the model at spawn). A small bash driver calls it twice.
"""

import argparse
import difflib
import json
import os
import re
import sys
import time


# ---- voter-field extraction from Surya table HTML (observed format) ----------
# "<b>N</b> ... Name : X  Father's/Husband's Name: Y  House Number : Z  Age : A Gender : G"
SERIAL_RE = re.compile(r"<b>\s*#?\s*(\d{1,4})\s*</b>")
NAME_RE = re.compile(r"Name\s*:\s*([^<\n]+?)(?:<|Father|Husband|Mother|$)", re.I)
REL_RE = re.compile(r"(Father|Husband|Mother)'?s?\s*Name\s*:\s*([^<\n]+?)(?:<|House|Age|$)", re.I)
HOUSE_RE = re.compile(r"House\s*Number\s*:\s*([^<\n]+?)(?:<|Age|$)", re.I)
AGE_RE = re.compile(r"Age\s*:\s*(\d{1,3})", re.I)
GENDER_RE = re.compile(r"Gender\s*:\s*(Male|Female|Third|Other)", re.I)
EPIC_RE = re.compile(r"\b([A-Z]{2,3}\d{6,8})\b")
TAG_RE = re.compile(r"<[^>]+>")


def strip_tags(html):
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", html)).strip()


def page_html_to_text(blocks):
    return "\n".join(b.get("html", "") for b in blocks)


def extract_voters(html):
    """Split the page HTML into per-voter cells and pull fields. Keyed by serial."""
    # cells are <td>...</td>; each real voter cell has a Name :
    cells = re.split(r"</td>", html)
    voters = {}
    for cell in cells:
        if "Name" not in cell:
            continue
        sm = SERIAL_RE.search(cell)
        nm = NAME_RE.search(cell)
        if not nm:
            continue
        rel = REL_RE.search(cell)
        v = {
            "name": strip_tags(nm.group(1)).strip(" :"),
            "rel_name": strip_tags(rel.group(2)).strip(" :") if rel else "",
            "house": (HOUSE_RE.search(cell).group(1).strip() if HOUSE_RE.search(cell) else ""),
            "age": (AGE_RE.search(cell).group(1) if AGE_RE.search(cell) else ""),
            "sex": (GENDER_RE.search(cell).group(1) if GENDER_RE.search(cell) else ""),
            "epic": (EPIC_RE.search(cell).group(1) if EPIC_RE.search(cell) else ""),
        }
        key = sm.group(1) if sm else v["name"]
        voters[key] = v
    return voters


# ---- mode: run -------------------------------------------------------------
def mode_run(args):
    if args.model:
        os.environ["SURYA_GGUF_LOCAL_MODEL_PATH"] = args.model
    if args.mmproj:
        os.environ["SURYA_GGUF_LOCAL_MMPROJ_PATH"] = args.mmproj
    os.environ.setdefault("DISABLE_TQDM", "true")

    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path

    npages = int(pdfinfo_from_path(args.pdf)["Pages"])
    idx = parse_pages(args.pages, npages)
    images = [convert_from_path(args.pdf, dpi=args.dpi, first_page=i + 1, last_page=i + 1)[0]
              for i in idx]

    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor
    manager = SuryaInferenceManager()
    rec = RecognitionPredictor(manager)

    # warm once on the first page (server start + Metal compile), not timed
    _ = rec([images[0]], full_page=True)

    t = time.time()
    results = rec(images, full_page=True)
    elapsed = time.time() - t

    pages = []
    for i, r in zip(idx, results):
        blocks = [{"label": b.label, "bbox": [round(x, 1) for x in b.bbox], "html": b.html or ""}
                  for b in r.blocks]
        pages.append({"page": i + 1, "blocks": blocks})

    out = {"pdf": args.pdf, "dpi": args.dpi, "model": args.model or "f16-default",
           "seconds_total": round(elapsed, 1),
           "seconds_per_page": round(elapsed / len(images), 1),
           "pages": pages}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"{out['model']}: {out['seconds_per_page']}s/page over {len(images)} pages -> {args.out}")
    try:
        manager.stop()
    except Exception:
        pass


# ---- mode: cmp -------------------------------------------------------------
def mode_cmp(args):
    gold = json.load(open(args.gold))
    var = json.load(open(args.variant))
    gpages = {p["page"]: p for p in gold["pages"]}
    vpages = {p["page"]: p for p in var["pages"]}

    FIELDS = ["name", "rel_name", "house", "age", "sex"]
    tot = {f: [0, 0] for f in FIELDS}  # [matches, compared]
    text_ratios = []
    voter_counts = []

    for pg in sorted(set(gpages) & set(vpages)):
        gh = page_html_to_text(gpages[pg]["blocks"])
        vh = page_html_to_text(vpages[pg]["blocks"])
        text_ratios.append(difflib.SequenceMatcher(None, strip_tags(gh), strip_tags(vh)).ratio())

        gv = extract_voters(gh)
        vv = extract_voters(vh)
        voter_counts.append((len(gv), len(vv)))
        for k in set(gv) & set(vv):
            for f in FIELDS:
                a, b = gv[k][f].lower().strip(), vv[k][f].lower().strip()
                if a:  # only score fields the gold actually has
                    tot[f][1] += 1
                    tot[f][0] += (a == b)

    print(f"\n  GOLD  {gold['model']:<18} {gold['seconds_per_page']}s/page")
    print(f"  VAR   {var['model']:<18} {var['seconds_per_page']}s/page "
          f"({gold['seconds_per_page']/max(var['seconds_per_page'],0.1):.2f}x speed)")
    print(f"\n  text similarity (gold vs variant): "
          f"{sum(text_ratios)/len(text_ratios)*100:.1f}% avg over {len(text_ratios)} pages")
    gvtot = sum(g for g, _ in voter_counts); vvtot = sum(v for _, v in voter_counts)
    print(f"  voters parsed: gold={gvtot}  variant={vvtot}")
    print(f"\n  per-field exact match (variant vs gold), over fields gold has:")
    for f in FIELDS:
        m, c = tot[f]
        print(f"    {f:<10} {m}/{c}  {100*m/c if c else 0:.1f}%")


def parse_pages(spec, n):
    out = []
    for part in (spec or f"1-{n}").split(","):
        if "-" in part:
            a, b = part.split("-"); out += range(int(a) - 1, int(b))
        else:
            out.append(int(part) - 1)
    return [i for i in out if 0 <= i < n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("pdf")
    r.add_argument("--pages", default=None)
    r.add_argument("--dpi", type=int, default=192)
    r.add_argument("--model", default=None, help="GGUF path; omit for f16 default")
    r.add_argument("--mmproj", default=None)
    r.add_argument("--out", required=True)
    r.set_defaults(func=mode_run)
    c = sub.add_parser("cmp")
    c.add_argument("gold")
    c.add_argument("variant")
    c.set_defaults(func=mode_cmp)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
