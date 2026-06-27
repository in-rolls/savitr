#!/usr/bin/env python3
"""Evaluate the distilled student on the gold TEST split (the teacher's held-out labels).

For each test page: run the student, parse its terse output, and compare to the gold terse
target (already in dataset_test.jsonl — no re-OCR). Reports tokens/page, s/page, voter recall,
and per-field accuracy vs the teacher. Run in .venv-torch.

  .venv-torch/bin/python savitr/distill/eval_distilled.py \
      --adapter models/smolvlm-savitr-lora --test corpus/dataset_test.jsonl
"""

import argparse
import json
import os
import re
import sys

from infer_torch import DistilledOCR
from savitr.rolls.parse import parse_terse

FIELDS = ["id", "elector_name", "father_or_husband_name", "house_no", "age", "sex"]


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def by_serial(voters):
    return {v["number"]: v for v in voters if v.get("number")}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--test", default="corpus/dataset_test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    ocr = DistilledOCR(args.adapter, args.base)

    tot_tok = tot_s = 0.0
    gold_n = matched_n = pred_n = 0
    field_hits = {f: [0, 0] for f in FIELDS}
    print(f"evaluating {len(rows)} gold test pages ...\n")
    print(f"{'page':>4} {'gold':>4} {'pred':>4} {'match':>5} {'tok':>5} {'s':>6}")
    for i, r in enumerate(rows):
        gold = by_serial(parse_terse(r["messages"][1]["content"]))
        try:
            pv, ntok, dt = ocr.ocr_image(r["image"])
        except Exception as e:
            print(f"{i:>4}  infer error: {e}")
            continue
        pred = by_serial(pv)
        tot_tok += ntok
        tot_s += dt
        gold_n += len(gold)
        pred_n += len(pred)
        m = set(gold) & set(pred)
        matched_n += len(m)
        for k in m:
            for f in FIELDS:
                a = norm(gold[k][f])
                if a:
                    field_hits[f][1] += 1
                    b = norm(pred[k][f])
                    hit = bool(
                        a == b
                        or (
                            f in ("elector_name", "father_or_husband_name")
                            and b
                            and (a in b or b in a)
                        )
                    )
                    field_hits[f][0] += hit
        print(f"{i:>4} {len(gold):>4} {len(pred):>4} {len(m):>5} {ntok:>5} {dt:>6.1f}")

    n = len(rows)
    print(f"\n  pages: {n} | avg {tot_tok/n:.0f} tokens/page | {tot_s/n:.1f} s/page")
    print(
        f"  voter recall (matched/gold serials): {matched_n}/{gold_n} "
        f"= {100*matched_n/max(gold_n,1):.1f}%  (pred total {pred_n})"
    )
    print(f"  per-field accuracy (on matched voters, vs teacher gold):")
    for f in FIELDS:
        h, c = field_hits[f]
        print(f"    {f:<24} {h}/{c}  {100*h/c if c else 0:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
