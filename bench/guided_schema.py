#!/usr/bin/env python3
"""Token lever: terse guided JSON per voter vs the gold labeled-HTML output.

The gold pipeline spends ~6-8k tokens/page emitting labeled HTML
(`<td><p><b>N</b></p><p>Name : … <br/>Father's Name: … `) for 30 voters. The labels/tags
repeat 30×/page and are pure overhead. This forces the model (via Surya `guided_json`,
i.e. OpenAI `response_format: json_schema`, which llama.cpp honors) to emit a compact JSON
array instead, and measures, per page:

  - tokens generated: gold bbox HTML vs guided JSON (thermal-independent speed proxy)
  - field match of the JSON voters vs the gold HTML voters

Runs sequentially (concurrent dense pages thrash the M4 / time out).

Usage:
  python bench/guided_schema.py PDF --pages 3-4 [--dpi 192] [--model … --mmproj …]
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_quality import extract_voters  # noqa: E402

FIELDS = ["name", "rel", "house", "age", "sex"]

# compact per-voter schema; short keys = fewer tokens
VOTER_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "name": {"type": "string"},
            "rel": {"type": "string"},
            "house": {"type": "string"},
            "age": {"type": "integer"},
            "sex": {"type": "string"},
        },
        "required": ["n", "name", "rel", "house", "age", "sex"],
    },
}
GUIDED_PROMPT = (
    "Read this electoral-roll page. Return a JSON array with one object per voter, in serial "
    "order. Keys: n=serial number, name=elector name, rel=father's or husband's name, "
    "house=house number, age=age, sex=Male/Female/Third."
)


def parse_pages(spec, n):
    out = []
    for part in (spec or f"1-{n}").split(","):
        if "-" in part:
            a, b = part.split("-"); out += range(int(a) - 1, int(b))
        else:
            out.append(int(part) - 1)
    return [i for i in out if 0 <= i < n]


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def voters_from_json(raw):
    """Parse the guided JSON array -> {serial: {name, rel, house, age, sex}}."""
    try:
        arr = json.loads(raw)
    except Exception:
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return {}
        try:
            arr = json.loads(m.group(0))
        except Exception:
            return {}
    out = {}
    for v in arr if isinstance(arr, list) else []:
        if not isinstance(v, dict):
            continue
        key = str(v.get("n", "")).strip()
        out[key] = {"name": str(v.get("name", "")), "rel": str(v.get("rel", "")),
                    "house": str(v.get("house", "")), "age": str(v.get("age", "")),
                    "sex": str(v.get("sex", ""))}
    return out


def voters_from_gold(html):
    """gold extract_voters -> normalize keys to the FIELDS used here."""
    gv = extract_voters(html)
    return {k: {"name": v["name"], "rel": v["rel_name"], "house": v["house"],
                "age": v["age"], "sex": v["sex"]} for k, v in gv.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="3-4")
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--model", default=None)
    ap.add_argument("--mmproj", default=None)
    ap.add_argument("--max-tokens", type=int, default=12288)
    args = ap.parse_args()

    if args.model:
        os.environ["SURYA_GGUF_LOCAL_MODEL_PATH"] = args.model
    if args.mmproj:
        os.environ["SURYA_GGUF_LOCAL_MMPROJ_PATH"] = args.mmproj
    os.environ.setdefault("DISABLE_TQDM", "true")

    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path
    from surya.inference import SuryaInferenceManager
    from surya.inference.schema import BatchInputItem, PROMPT_TYPE_HIGH_ACCURACY_BBOX

    npages = int(pdfinfo_from_path(args.pdf)["Pages"])
    idx = parse_pages(args.pages, npages)
    images = [convert_from_path(args.pdf, dpi=args.dpi, first_page=i + 1, last_page=i + 1)[0]
              .convert("RGB") for i in idx]

    manager = SuryaInferenceManager()
    manager.generate([BatchInputItem(image=images[0],
                                     prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX,
                                     prompt="OCR this image to HTML.", max_tokens=128,
                                     metadata={})])  # warm

    items = []
    for i, img in zip(idx, images):
        items.append(("bbox", i + 1, BatchInputItem(
            image=img, prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX, prompt=None,
            max_tokens=args.max_tokens, metadata={})))
        items.append(("guided", i + 1, BatchInputItem(
            image=img, prompt_type=PROMPT_TYPE_HIGH_ACCURACY_BBOX, prompt=GUIDED_PROMPT,
            guided_json=VOTER_SCHEMA, max_tokens=args.max_tokens, metadata={})))

    t = time.time()
    out = {}
    for variant, pg, item in items:
        out[(variant, pg)] = manager.generate([item])[0]
    elapsed = time.time() - t

    print(f"\nmodel: {args.model or 'f16-default'}  pages: {[i+1 for i in idx]}  "
          f"wall: {elapsed:.1f}s\n")
    print(f"{'page':>4} {'bbox_tok':>9} {'json_tok':>9} {'reduction':>10}  "
          f"{'gold_v':>6} {'json_v':>6}  field-match(json vs gold)")
    tb = tj = 0
    agg = {f: [0, 0] for f in FIELDS}
    for pg in [i + 1 for i in idx]:
        b = out[("bbox", pg)]; g = out[("guided", pg)]
        tb += b.token_count; tj += g.token_count
        gold = voters_from_gold(b.raw)
        jv = voters_from_json(g.raw)
        for k in set(gold) & set(jv):
            for f in FIELDS:
                a = norm(gold[k][f]); c = norm(jv[k][f])
                # sex: compare first letter (Male/M)
                if f == "sex":
                    a, c = a[:1], c[:1]
                if a:
                    agg[f][1] += 1
                    agg[f][0] += (a == c or (f in ("name", "rel") and (a in c or c in a)))
        fm = " ".join(f"{f[:4]}={sum(1 for k in set(gold)&set(jv) if norm(gold[k][f]) and (norm(gold[k][f])==norm(jv[k][f]) or (f in ('name','rel') and (norm(gold[k][f]) in norm(jv[k][f]) or norm(jv[k][f]) in norm(gold[k][f])))))}" for f in FIELDS)
        red = b.token_count / max(g.token_count, 1)
        print(f"{pg:>4} {b.token_count:>9} {g.token_count:>9} {red:>9.2f}x  "
              f"{len(gold):>6} {len(jv):>6}  {fm}")

    print(f"\n  TOTAL tokens: bbox={tb}  json={tj}  -> {tb/max(tj,1):.2f}x fewer with guided JSON")
    print(f"  field-match json vs gold (over fields gold has):")
    for f in FIELDS:
        m, n = agg[f]
        print(f"    {f:<6} {m}/{n}  {100*m/n if n else 0:.1f}%")

    try:
        manager.stop()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
