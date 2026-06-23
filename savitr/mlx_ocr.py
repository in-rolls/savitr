#!/usr/bin/env python3
"""Fast Surya OCR via the MLX backend — the production engine.

Loads the MLX-converted Surya model once and OCRs a PDF's voter pages to structured records
at ~175-180 tok/s on Apple Silicon (~3.6× the llama.cpp f16 pipeline). The voter parser is
**layout-robust**: it anchors on the repeated field labels (`Name :`, `Father's Name:`, …)
rather than table-cell structure, because the model emits more than one valid `<table>`
layout for the same page.

Run in the MLX env:
  .venv-mlx/bin/python savitr/mlx_ocr.py PDF [--pages 3-14] [--mlx-path models/surya-mlx-4bit]
"""

import argparse
import re
import sys
import time

PROMPT = "OCR this image to HTML."

# Layout-robust voter extraction: anchor on "Name :", read fields forward, attach the
# nearest preceding serial + EPIC. Works regardless of <td>/<tr> nesting.
TAG = re.compile(r"<[^>]+>")
EPIC = re.compile(r"\b([A-Z]{2,3}\d{6,8})\b")
SERIAL = re.compile(r"(\d{1,4})")
NAME = re.compile(r"Name\s*:\s*(.*?)(?:Father|Husband|Mother|House|Age|Gender|$)", re.I | re.S)
REL = re.compile(r"(Father|Husband|Mother)'?s?\s*Name\s*:\s*(.*?)(?:House|Age|Gender|$)", re.I | re.S)
HOUSE = re.compile(r"House\s*Number\s*:\s*(.*?)(?:Age|Gender|$)", re.I | re.S)
AGE = re.compile(r"Age\s*:\s*(\d{1,3})", re.I)
GENDER = re.compile(r"Gender\s*:\s*(Male|Female|Third|Other)", re.I)
REL_CODE = {"father": "F", "husband": "H", "mother": "M"}
SEX_CODE = {"male": "M", "female": "F", "third": "T", "other": "T"}


def _clean(s):
    return re.sub(r"\s+", " ", TAG.sub(" ", s)).strip(" :")


def parse_voters(html):
    """Split page HTML into per-voter records by the 'Name :' anchor (layout-agnostic)."""
    # tokenize into (kind, value) markers in reading order: serial bolds, epics, name-anchors
    # Anchor on the ELECTOR "Name :" only. Exclude header/relation occurrences that also
    # contain "Name :": "Father's/Husband's/Mother's Name:", "… No and Name :" (section /
    # constituency / polling-station headers), "Name and Reservation …".
    def _is_header(m):
        pre = html[max(0, m.start() - 16):m.start()]
        post = html[m.end():m.end() + 24]
        return bool(re.search(r"(Father|Husband|Mother)'?s?\s*$", pre, re.I)
                    or re.search(r"\band\s*$", pre, re.I)
                    or re.match(r"\s*and\s+Reservation", post, re.I))

    # 1) ordered voter blocks — name/relation/house/age/sex are co-located with the name in
    #    both table layouts the model emits. Capture a per-block LEADING serial too
    #    (layout A renders it as a plain "31<br/>Name :" right before the name).
    name_iters = [m for m in re.finditer(r"Name\s*:", html, re.I) if not _is_header(m)]
    voters, starts = [], []
    for i, m in enumerate(name_iters):
        blob = html[m.start():(name_iters[i + 1].start() if i + 1 < len(name_iters) else len(html))]
        nm = NAME.search(blob)
        name = _clean(nm.group(1)) if nm else ""
        am, gm = AGE.search(blob), GENDER.search(blob)
        # a real voter record has a letter-name AND an age or gender (headers have neither)
        if not name or name[0].isdigit() or not (am or gm):
            continue
        rel, hm = REL.search(blob), HOUSE.search(blob)
        pre = html[max(0, m.start() - 28):m.start()]
        lead = re.search(r"(#)?\s*(\d{1,4})\s*(?:<br\s*/?>)\s*(?:<[^>]*>\s*)*$", pre)
        starts.append(m.start())
        voters.append({
            "elector_name": name,
            "father_or_husband_name": _clean(rel.group(2)) if rel else "",
            "relationship": REL_CODE.get(rel.group(1).lower(), "") if rel else "",
            "house_no": _clean(hm.group(1)) if hm else "",
            "age": am.group(1) if am else "",
            "sex": SEX_CODE.get(gm.group(1).lower(), "") if gm else "",
            "number": lead.group(2) if lead else "",
            "original_or_amendment": "amendment" if (lead and lead.group(1)) else "original",
        })

    # 1b) de-loop: the model sometimes repeats voters (a decode loop). Cut at the first time a
    #     (name, age) identity recurs — and truncate the HTML at the SAME point so the
    #     EPIC/serial index-alignment below reads only the clean first copy.
    seen, cut = set(), len(voters)
    for j, v in enumerate(voters):
        key = (re.sub(r"\s+", " ", v["elector_name"].lower()).strip(), v["age"])
        if key in seen:
            cut = j
            break
        seen.add(key)
    clean = html[:starts[cut]] if cut < len(voters) else html
    voters = voters[:cut]

    # 2) EPICs (and layout-B <b> serials) in document order, aligned to voters BY INDEX —
    #    correct within each row-group. Only fill what the per-block leading serial missed.
    epics = EPIC.findall(clean)
    bold = re.findall(r"<b>\s*(#)?\s*(\d{1,4})\s*</b>", clean)
    for k, v in enumerate(voters):
        v["id"] = epics[k] if k < len(epics) else ""
        if not v["number"] and k < len(bold):
            v["number"] = bold[k][1]
            v["original_or_amendment"] = "amendment" if bold[k][0] else "original"
    return voters


def dedupe_voters(voters):
    """Collapse the model's duplicated rows, keeping the fullest record per voter.

    Keys by EPIC id when present (it is the unique voter key — a repeated EPIC means the
    model mis-associated or looped, so those rows must collapse), else by identity
    (name+relation+age). Robust to the messy duplication the VLM emits on looped pages.
    """
    def score(v):
        return sum(1 for k in ("id", "number", "father_or_husband_name", "house_no",
                               "age", "sex") if v.get(k))

    best = {}
    for v in voters:
        key = ("epic", v["id"]) if v.get("id") else (
            "id", re.sub(r"\s+", " ", v["elector_name"].lower()).strip(),
            re.sub(r"\s+", " ", v.get("father_or_husband_name", "").lower()).strip(),
            v.get("age", ""))
        if key not in best or score(v) > score(best[key]):
            best[key] = v
    out = list(best.values())
    return sorted(out, key=lambda v: int(v["number"]) if v.get("number", "").isdigit() else 1e9)


class MLXSuryaOCR:
    """Load the MLX Surya model once; OCR pages to HTML."""

    def __init__(self, mlx_path="models/surya-mlx-4bit", max_tokens=8192):
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        self._generate = generate
        self.model, self.processor = load(mlx_path)
        config = load_config(mlx_path)
        self.prompt = apply_chat_template(self.processor, config, PROMPT, num_images=1)
        self.max_tokens = max_tokens

    def ocr_image(self, png_path):
        res = self._generate(self.model, self.processor, self.prompt, image=png_path,
                             max_tokens=self.max_tokens, verbose=False)
        return getattr(res, "text", None) or str(res), getattr(res, "generation_tokens", 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--pages", default=None, help="1-based range e.g. 3-14 (default: all)")
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--mlx-path", default="models/surya-mlx-4bit")
    args = ap.parse_args()

    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path

    npages = int(pdfinfo_from_path(args.pdf)["Pages"])
    if args.pages:
        idx = []
        for part in args.pages.split(","):
            if "-" in part:
                a, b = part.split("-"); idx += range(int(a) - 1, int(b))
            else:
                idx.append(int(part) - 1)
        idx = [i for i in idx if 0 <= i < npages]
    else:
        idx = list(range(npages))

    print(f"loading MLX model {args.mlx_path} ...")
    eng = MLXSuryaOCR(args.mlx_path)

    total_voters = 0
    t0 = time.time()
    for i in idx:
        png = f"/tmp/mlx_pdf_p{i+1}.png"
        convert_from_path(args.pdf, dpi=args.dpi, first_page=i + 1,
                          last_page=i + 1)[0].convert("RGB").save(png)
        t = time.time()
        html, gtok = eng.ocr_image(png)
        voters = parse_voters(html)
        # keep only real voter pages (interior pages have many records)
        n = len(voters)
        total_voters += n
        print(f"  page {i+1:>3}: {gtok:>5} tok, {time.time()-t:5.1f}s, {n:>2} voters"
              + (f"  e.g. {voters[0]['number']}|{voters[0]['id']}|{voters[0]['elector_name']}"
                 if voters else ""))

    elapsed = time.time() - t0
    print(f"\n{len(idx)} pages in {elapsed:.1f}s = {elapsed/len(idx):.1f}s/page, "
          f"{total_voters} voters total")


if __name__ == "__main__":
    sys.exit(main())
