#!/usr/bin/env python3
"""Out-of-sample accuracy of terse-Surya vs the Surya teacher, on the held-out test pages.

The teacher's terse output (test_split.json targets) is the gold we distilled toward; this measures
how faithfully terse-Surya reproduces it on pages it never trained on. Match voters by EPIC, then by
fuzzy name; score every field on matched voters + voter recall."""

import glob
import json
import os
import re
import sys

from savitr.mlx_ocr import MLXSuryaOCR
from savitr.rolls.parse import TERSE_PROMPT, parse_terse

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/surya-terse-8bit"
TEST = "/tmp/kfin/surya-terse-lora/test_split.json"
FIELDS = ["id", "elector_name", "father_or_husband_name", "relationship", "house_no", "age", "sex"]


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def fuzzy(a, b):
    a, b = norm(a), norm(b)
    return a == b or (len(a) > 3 and (a in b or b in a))


def main():
    test = json.load(open(TEST))
    eng = MLXSuryaOCR(MODEL, max_tokens=2048, prompt=TERSE_PROMPT)
    agg = {f: [0, 0] for f in FIELDS}
    gold_total = matched_total = pages = 0
    for ex in test:
        loc = glob.glob(f"pages/{os.path.basename(ex['image'])}")
        if not loc:
            continue
        pages += 1
        gold = parse_terse(ex["target"])
        pred = parse_terse(eng.ocr_image(loc[0])[0])
        gold_total += len(gold)
        pmap = {v["id"]: v for v in pred if v.get("id")}
        used = set()
        for gv in gold:
            pv = pmap.get(gv.get("id")) if gv.get("id") else None
            if not pv or id(pv) in used:  # fall back to fuzzy name match
                pv = next(
                    (
                        p
                        for p in pred
                        if id(p) not in used and fuzzy(gv["elector_name"], p["elector_name"])
                    ),
                    None,
                )
            if not pv:
                continue
            used.add(id(pv))
            matched_total += 1
            for f in FIELDS:
                if norm(gv.get(f, "")):
                    agg[f][1] += 1
                    agg[f][0] += int(fuzzy(gv.get(f, ""), pv.get(f, "")))
    print(f"\n=== OOS fidelity to Surya teacher: {MODEL} ===")
    print(
        f"{pages} held-out pages | {gold_total} teacher voters | "
        f"{matched_total} matched = {100*matched_total/max(gold_total,1):.1f}% recall"
    )
    for f in FIELDS:
        h, c = agg[f]
        print(f"  {f:<26} {h}/{c} = {100*h/c if c else 0:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
