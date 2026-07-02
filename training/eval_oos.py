#!/usr/bin/env python3
"""Out-of-sample accuracy of terse-Surya vs the Surya teacher, on the held-out test pages.

The teacher's terse output (corpus/dataset_test.jsonl, gold in ``messages[-1]``) is the gold we
distilled toward; this measures how faithfully terse-Surya reproduces it on pages from ACs it never
trained on. Match voters by EPIC, then by fuzzy name; score every field on matched voters + voter
recall, plus an edit-distance similarity (difflib ratio) per matched record and per whole page.

Usage:  .venv-mlx/bin/python training/eval_oos.py [models/surya-terse-8bit-v2]
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher

from savitr.mlx_ocr import MLXSuryaOCR
from savitr.rolls.parse import TERSE_PROMPT, parse_terse

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/surya-terse-8bit"
TEST = os.environ.get("OOS_TEST", "corpus/dataset_test.jsonl")
IMG_DIR = os.environ.get("OOS_IMAGES", "corpus/images")
FIELDS = ["id", "elector_name", "father_or_husband_name", "relationship", "house_no", "age", "sex"]


def norm(s):
    """Lowercase + collapse whitespace for forgiving comparison."""
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def fuzzy(a, b):
    """True if two field values match exactly or one contains the other (len>3)."""
    a, b = norm(a), norm(b)
    return a == b or (len(a) > 3 and (a in b or b in a))


def gold_of(ex):
    """Pull the teacher terse text from a chat-format example's assistant turn."""
    c = ex["messages"][-1]["content"]
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c)


def rec_str(v):
    """Canonical per-voter string for edit-distance scoring."""
    return "|".join(norm(v.get(f, "")) for f in FIELDS)


def main():
    """Run the OOS eval for ``MODEL`` and print field accuracy + edit-distance similarity."""
    exs = [json.loads(line) for line in open(TEST, encoding="utf-8")]
    eng = MLXSuryaOCR(MODEL, max_tokens=2048, prompt=TERSE_PROMPT)
    agg = {f: [0, 0] for f in FIELDS}
    gold_total = matched_total = pages = 0
    sims, page_sims = [], []
    for ex in exs:
        img = os.path.join(IMG_DIR, os.path.basename(ex["image"]))
        if not os.path.exists(img):
            continue
        pages += 1
        gtext = gold_of(ex)
        ptext = eng.ocr_image(img)[0]
        gold, pred = parse_terse(gtext), parse_terse(ptext)
        page_sims.append(SequenceMatcher(None, gtext, ptext).ratio())
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
            sims.append(SequenceMatcher(None, rec_str(gv), rec_str(pv)).ratio())
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
    print(
        f"  {'record similarity':<26} avg {100*sum(sims)/max(len(sims),1):5.1f}% "
        f"(1 - norm. edit distance, {len(sims)} matched voters)"
    )
    print(f"  {'whole-page similarity':<26} avg {100*sum(page_sims)/max(len(page_sims),1):5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
