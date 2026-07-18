"""Electoral-roll parsing: Surya output (HTML or terse) -> voter records.

This is the roll-specific layer. The generic MLX engine (:class:`savitr.mlx_ocr.MLXSuryaOCR`)
turns a page image into text; these functions turn that text into voter dicts with the canonical
fields ``number, id, elector_name, relationship, father_or_husband_name, house_no, age, sex``.

- :func:`parse_voters` / :func:`dedupe_voters` parse the base model's verbose HTML.
- :func:`parse_terse` parses the distilled terse model's one-line-per-voter output.
- :func:`to_terse` renders voter dicts back to the terse training target.
- :func:`resolve_terse_model` locates (or downloads) the terse model weights.
"""

import os
import re

#: Hugging Face repo for the distilled terse roll model (used when no local copy is present).
TERSE_REPO = os.environ.get("SAVITR_TERSE_REPO", "gojiberries/savitr")


def resolve_terse_model(local: str = "models/surya-terse-8bit") -> str:
    """Return a local terse-model dir if present, else download :data:`TERSE_REPO` from the Hub."""
    if os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download

    print(f"fetching terse model {TERSE_REPO} from Hugging Face ...")
    return snapshot_download(TERSE_REPO)


# Layout-robust voter extraction: anchor on "Name :", read fields forward, attach the
# nearest preceding serial + EPIC. Works regardless of <td>/<tr> nesting.
TAG = re.compile(r"<[^>]+>")
EPIC = re.compile(r"\b([A-Z]{2,3}\d{6,8})\b")
SERIAL = re.compile(r"(\d{1,4})")
NAME = re.compile(r"Name\s*:\s*(.*?)(?:Father|Husband|Mother|House|Age|Gender|$)", re.I | re.S)
REL = re.compile(
    r"(Father|Husband|Mother)'?s?\s*Name\s*:\s*(.*?)(?:House|Age|Gender|$)", re.I | re.S
)
HOUSE = re.compile(r"House\s*Number\s*:\s*(.*?)(?:Age|Gender|$)", re.I | re.S)
AGE = re.compile(r"Age\s*:\s*(\d{1,3})", re.I)
GENDER = re.compile(r"Gender\s*:\s*(Male|Female|Third|Other)", re.I)
REL_CODE = {"father": "F", "husband": "H", "mother": "M"}
SEX_CODE = {"male": "M", "female": "F", "third": "T", "other": "T"}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", TAG.sub(" ", s)).strip(" :")


def parse_voters(html: str) -> list[dict]:
    """Split page HTML into per-voter records by the ``Name :`` anchor (layout-agnostic)."""

    # Anchor on the ELECTOR "Name :" only. Exclude header/relation occurrences that also
    # contain "Name :": "Father's/Husband's/Mother's Name:", "... No and Name :" (section /
    # constituency / polling-station headers), "Name and Reservation ...".
    def _is_header(m: re.Match) -> bool:
        pre = html[max(0, m.start() - 16) : m.start()]
        post = html[m.end() : m.end() + 24]
        return bool(
            re.search(r"(Father|Husband|Mother)'?s?\s*$", pre, re.I)
            or re.search(r"\band\s*$", pre, re.I)
            or re.match(r"\s*and\s+Reservation", post, re.I)
        )

    # Ordered voter blocks: name/relation/house/age/sex are co-located with the name in both
    # table layouts the model emits. Capture a per-block LEADING serial too (layout A renders
    # it as a plain "31<br/>Name :" right before the name).
    name_iters = [m for m in re.finditer(r"Name\s*:", html, re.I) if not _is_header(m)]
    voters, starts = [], []
    for i, m in enumerate(name_iters):
        blob = html[
            m.start() : (name_iters[i + 1].start() if i + 1 < len(name_iters) else len(html))
        ]
        nm = NAME.search(blob)
        name = _clean(nm.group(1)) if nm else ""
        am, gm = AGE.search(blob), GENDER.search(blob)
        # a real voter record has a letter-name AND an age or gender (headers have neither)
        if not name or name[0].isdigit() or not (am or gm):
            continue
        rel, hm = REL.search(blob), HOUSE.search(blob)
        pre = html[max(0, m.start() - 28) : m.start()]
        lead = re.search(r"(#)?\s*(\d{1,4})\s*(?:<br\s*/?>)\s*(?:<[^>]*>\s*)*$", pre)
        starts.append(m.start())
        voters.append(
            {
                "elector_name": name,
                "father_or_husband_name": _clean(rel.group(2)) if rel else "",
                "relationship": REL_CODE.get(rel.group(1).lower(), "") if rel else "",
                "house_no": _clean(hm.group(1)) if hm else "",
                "age": am.group(1) if am else "",
                "sex": SEX_CODE.get(gm.group(1).lower(), "") if gm else "",
                "number": lead.group(2) if lead else "",
                "original_or_amendment": "amendment" if (lead and lead.group(1)) else "original",
            }
        )

    # De-loop: the model sometimes repeats voters (a decode loop). Cut at the first time a
    # (name, age) identity recurs, and truncate the HTML at the SAME point so the EPIC/serial
    # index-alignment below reads only the clean first copy.
    seen, cut = set(), len(voters)
    for j, v in enumerate(voters):
        key = (re.sub(r"\s+", " ", v["elector_name"].lower()).strip(), v["age"])
        if key in seen:
            cut = j
            break
        seen.add(key)
    clean = html[: starts[cut]] if cut < len(voters) else html
    voters = voters[:cut]

    # EPICs (and layout-B <b> serials) in document order, aligned to voters BY INDEX — correct
    # within each row-group. Only fill what the per-block leading serial missed.
    epics = EPIC.findall(clean)
    bold = re.findall(r"<b>\s*(#)?\s*(\d{1,4})\s*</b>", clean)
    for k, v in enumerate(voters):
        v["id"] = epics[k] if k < len(epics) else ""
        if not v["number"] and k < len(bold):
            v["number"] = bold[k][1]
            v["original_or_amendment"] = "amendment" if bold[k][0] else "original"
    return voters


def dedupe_voters(voters: list[dict]) -> list[dict]:
    """Collapse duplicated rows, keeping the fullest record per voter.

    Keys by EPIC id when present (the unique voter key — a repeated EPIC means the model
    mis-associated or looped, so those rows must collapse), else by identity (name+relation+age).
    Robust to the messy duplication the VLM emits on looped pages.
    """

    def score(v: dict) -> int:
        return sum(
            1
            for k in ("id", "number", "father_or_husband_name", "house_no", "age", "sex")
            if v.get(k)
        )

    best: dict = {}
    for v in voters:
        key = (
            ("epic", v["id"])
            if v.get("id")
            else (
                "id",
                re.sub(r"\s+", " ", v["elector_name"].lower()).strip(),
                re.sub(r"\s+", " ", v.get("father_or_husband_name", "").lower()).strip(),
                v.get("age", ""),
            )
        )
        if key not in best or score(v) > score(best[key]):
            best[key] = v
    out = list(best.values())
    return sorted(out, key=lambda v: int(v["number"]) if v.get("number", "").isdigit() else 1e9)


# ---- terse format (distillation target / output) -----------------------------
#: Canonical column order for one terse voter line.
TERSE_COLS = [
    "number",
    "id",
    "elector_name",
    "relationship",
    "father_or_husband_name",
    "house_no",
    "age",
    "sex",
]
#: Instruction given to the distilled model.
TERSE_PROMPT = (
    "Extract every voter from this electoral-roll page as pipe-delimited rows, "
    "one per line, columns: serial|epic|name|relation(F/H/M)|relation_name|house|age|sex"
)

_EPIC_TOK = re.compile(r"[A-Z]{1,3}\d{5,9}")
_AGE_TOK = re.compile(r"\d{1,3}")


def _san(v) -> str:
    return re.sub(r"[|\r\n]+", " ", str(v)).strip()


def to_terse(voters: list[dict]) -> str:
    """Render voter dicts to terse pipe-delimited text (the training target)."""
    return "\n".join("|".join(_san(v.get(c, "")) for c in TERSE_COLS) for v in voters)


def parse_terse(text: str) -> list[dict]:
    """Parse the terse model's output into voter dicts (value-anchored, not positional).

    The distilled model occasionally drops the relation-code column on hard pages, which would
    shift every field if split blindly. So we anchor on recognizable values — EPIC at the front,
    sex (M/F/T) and age at the end, relation only when it is literally F/H/M — and the free-form
    name/relname/house fall out of what remains. This keeps the reliable fields (EPIC, name, age,
    sex) aligned even when a middle column is missing.
    """
    voters = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        while parts and parts[0] == "":  # serial column is usually blank
            parts.pop(0)
        # gold targets lead with a numeric serial (model output usually omits it); drop it
        # when an EPIC follows, so the EPIC anchor lands on parts[0] either way.
        if len(parts) > 1 and parts[0].isdigit() and _EPIC_TOK.fullmatch(parts[1]):
            parts.pop(0)
        if len(parts) < 2:
            continue
        v = {c: "" for c in TERSE_COLS}
        if _EPIC_TOK.fullmatch(parts[0]):
            v["id"] = parts.pop(0)
        v["elector_name"] = parts.pop(0) if parts else ""
        if parts and parts[-1].upper() in ("M", "F", "T"):  # sex anchors the tail
            v["sex"] = parts.pop().upper()
        if parts and _AGE_TOK.fullmatch(parts[-1]):
            v["age"] = parts.pop()
        # remaining middle = [relation?, relname, house]
        if parts and parts[0].upper() in ("F", "H", "M"):
            v["relationship"] = parts.pop(0).upper()
        if parts:
            v["house_no"] = parts.pop()
        if parts:
            v["father_or_husband_name"] = " ".join(parts)
        if not v["elector_name"]:
            continue
        # Real voter rows lead with an EPIC. Rows where no EPIC anchored (blank id) are usually
        # page-footer boilerplate ("Age as on 01-01-2025", "# - Modified as per supplement") or a
        # decode-loop artifact that mis-split — drop them if they carry stray EPIC tokens (a loop
        # leaks EPICs mid-line), don't start with a letter, or lack the age+sex a voter always has.
        # This only scrutinizes EPIC-less rows, so genuine voters (and the eval numbers) are untouched.
        if not v["id"]:
            nm = v["elector_name"]
            if _EPIC_TOK.search(line) or not nm[:1].isalpha() or not (v["age"] and v["sex"]):
                continue
        v["original_or_amendment"] = "original"
        voters.append(v)
    return voters
