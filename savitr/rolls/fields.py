"""Geometry-based field extraction from OCR lines (engine-agnostic).

Vendored verbatim from parse_unsearchable_rolls/scripts/manipur/fields.py so savitr is
self-contained. savitr's MLX pipeline uses only `parse_cover_page` (+ `page_text`), fed a
synthetic line-list built from the model's text; the rest is kept for byte-compatibility.

Each page is ``{"width","height","lines":[{text,conf,x0,y0,x1,y1,cx,cy}]}`` (see
``ocr_engines.py``). Interior voter pages are a fixed 3-column x 10-row grid of 30 voter
boxes; each box holds a serial number + EPIC (top), then Name / relation / House / Age /
Gender. We group lines into boxes by 2D nearest-"Name :" anchor, so each voter's EPIC is the
EPIC line physically closest to that voter -- robust to OCR reading-order quirks.

Patterns use ``\\s*`` between tokens because some OCR engines drop spaces inside ALL-CAPS
runs (e.g. ``RAMESHPOUDEL``, ``PoliceStation``).
"""

import re

EPIC_RE = re.compile(r"\b([A-Z]{2,3}\d{6,8})\b")
SERIAL_LINE_RE = re.compile(r"^\s*(#)?\s*(\d{1,4})\s*$")
NAME_ANCHOR_RE = re.compile(r"^\s*Name\s*[:.]", re.I)  # voter name line, not "Name of ..."
NAME_RE = re.compile(r"\bName\s*[:.]?\s*(.+)", re.I)
REL_RE = re.compile(r"(Father|Husband|Mother)'?s?\s*Name\s*[:.]?\s*(.+)", re.I)
HOUSE_RE = re.compile(r"House\s*Number\s*[:.]?\s*(.+)", re.I)
AGE_RE = re.compile(r"Age\s*[:.]?\s*(\d{1,3})", re.I)
GENDER_RE = re.compile(r"Gender\s*[:.]?\s*(Male|Female|Third|Other)", re.I)

HEADER_RE = re.compile(r"Assembly\s*Constituency|Part\s*No|Section\s*No", re.I)
FOOTER_RE = re.compile(r"Age\s*as\s*on|Modified|Date\s*of\s*Publication|Total\s*Pages", re.I)

REL_CODE = {"father": "F", "husband": "H", "mother": "M"}
SEX_CODE = {"male": "M", "female": "F", "third": "T", "other": "T"}


def page_text(page):
    """All line text top-to-bottom, left-to-right (for cover detection/parsing)."""
    lines = sorted(page.get("lines", []), key=lambda l: (round(l["cy"] / 8), l["cx"]))
    return "\n".join(l["text"] for l in lines)


def _body_lines(page):
    return [
        l
        for l in page.get("lines", [])
        if not HEADER_RE.search(l["text"]) and not FOOTER_RE.search(l["text"])
    ]


def _anchors(lines):
    return [l for l in lines if NAME_ANCHOR_RE.match(l["text"])]


def is_interior_page(page):
    return len(_anchors(_body_lines(page))) >= 5


def parse_voter_blob(blob):
    """Extract one voter's fields from the joined text of its box lines."""
    number = modified = ""
    for ln in blob.splitlines():
        sm = SERIAL_LINE_RE.match(ln)
        if sm:
            modified = bool(sm.group(1))
            number = sm.group(2)
            break

    name = ""
    nm = NAME_RE.search(blob)
    if nm:
        name = re.split(r"(Father|Husband|Mother)'?s?\s*Name", nm.group(1), flags=re.I)[0].strip()

    rel_type = rel_name = ""
    rm = REL_RE.search(blob)
    if rm:
        rel_type = REL_CODE.get(rm.group(1).lower(), "")
        rel_name = re.split(r"House\s*Number|Age\s*[:.]", rm.group(2), flags=re.I)[0].strip()

    house = ""
    hm = HOUSE_RE.search(blob)
    if hm:
        house = re.split(r"Age\s*[:.]", hm.group(1), flags=re.I)[0].strip()

    am = AGE_RE.search(blob)
    age = am.group(1) if am else ""
    gm = GENDER_RE.search(blob)
    sex = SEX_CODE.get(gm.group(1).lower(), "") if gm else ""
    em = EPIC_RE.search(blob)

    return {
        "number": number,
        "id": em.group(1) if em else "",
        "elector_name": name,
        "father_or_husband_name": rel_name,
        "relationship": rel_type,
        "house_no": house,
        "age": age,
        "sex": sex,
        "original_or_amendment": "amendment" if modified else "original",
    }


def parse_interior_page(page):
    """Group body lines into the 30-voter grid, then parse each box."""
    width = page.get("width") or max((l["x1"] for l in page.get("lines", [])), default=1)
    body = _body_lines(page)
    anchors = _anchors(body)
    if not anchors:
        return []

    def col(line):
        return min(2, int(line["cx"] * 3 / width))

    anchor_cols = [col(a) for a in anchors]
    groups = {i: [] for i in range(len(anchors))}
    for line in body:
        c = col(line)
        cand = [i for i in range(len(anchors)) if anchor_cols[i] == c]
        if not cand:
            cand = range(len(anchors))
        best = min(cand, key=lambda i: abs(line["cy"] - anchors[i]["cy"]))
        groups[best].append(line)

    voters = []
    for i in groups:
        lines = sorted(groups[i], key=lambda l: (round(l["cy"] / 8), l["cx"]))
        blob = "\n".join(l["text"] for l in lines)
        v = parse_voter_blob(blob)
        if v["elector_name"] or v["number"]:
            voters.append(v)
    return voters


def dedupe_voters(voters):
    """Collapse duplicate serials, keeping the fullest record."""

    def score(v):
        return sum(
            1
            for k in ("id", "elector_name", "father_or_husband_name", "house_no", "age", "sex")
            if v.get(k)
        )

    best = {}
    extras = []
    for v in voters:
        key = v["number"]
        if not key:
            extras.append(v)
        elif key not in best or score(v) > score(best[key]):
            best[key] = v
    out = list(best.values()) + extras
    return sorted(out, key=lambda v: int(v["number"]) if v["number"].isdigit() else 1e9)


# --------------------------------------------------------------------- cover page

_COVER_FIELDS = {
    "ac": r"Assembly\s*Constituency[^:]*[:.]?\s*([0-9]+\s*-\s*[A-Z][A-Z .'()/-]+?)(?:\s*Part\s*No|\n|$)",
    "parl_constituency": r"Parliamentary\s*Constituency.*?\b(\d+\s*-\s*[A-Z][A-Z .'()/-]+)",
    "year": r"Year\s*of\s*Revision\s*[:.]?\s*(\d{4})",
    "qualifying_date": r"Qualifying\s*Date\s*[:.]?\s*([\d-]{8,10})",
    "publication_date": r"Date\s*of\s*Publication\s*[:.]?\s*([\d-]{8,10})",
    "part_no": r"Part\s*No\.?\s*[:.]?\s*(\d+)",
    "main_town": r"Main\s*Town\s*or\s*Village\s*[:.]?\s*([A-Z][A-Za-z .'/-]+)",
    "police_station": r"Police\s*Station\s*[:.]?\s*([A-Z][A-Za-z .'/-]+)",
    "pin_code": r"Pin\s*code\s*[:.]?\s*(\d{6})",
    "polling_station_name": r"Name\s*of\s*Polling\s*Station\s*[:.]?.{0,80}?(\d+\s*-\s*[A-Z][A-Za-z .'/-]+?)(?:\s*Address|\n|$)",
    "polling_station_address": r"Address\s*of\s*Polling\s*Station\b.*?([A-Z][A-Za-z0-9 .,'/&-]*?"
    r"(?:School|College|Vidyalaya|Building|Office|Hall|Centre|Center|Bhavan|Quarter)s?\b)",
}
_DISTRICT_RE = re.compile(r"^District\s*[:.]?\s*([A-Z][A-Za-z .'/-]+)$", re.M)


def parse_cover_page(page):
    text = page_text(page)
    flat = re.sub(r"[ \t]+", " ", text)
    meta = {}
    for key, pat in _COVER_FIELDS.items():
        m = re.search(pat, flat, re.I | re.S)
        meta[key] = m.group(1).strip(" :-") if m else ""
    dm = _DISTRICT_RE.search(text)
    meta["district"] = dm.group(1).strip() if dm else ""

    if meta.get("ac"):
        am = re.match(r"(\d+)\s*-\s*(.+)", meta["ac"])
        if am:
            meta["ac_number"] = am.group(1)
            meta["ac_name"] = re.sub(r"\s*\(.*\)\s*$", "", am.group(2)).strip()

    nm = re.search(
        r"NUMBER\s*OF\s*ELECTORS.*?(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)", flat, re.I | re.S
    )
    if nm:
        (
            meta["start_serial"],
            meta["end_serial"],
            meta["net_electors_male"],
            meta["net_electors_female"],
            meta["net_electors_third_gender"],
            meta["net_electors_total"],
        ) = nm.groups()
    return meta
