"""Convert Surya HTML or terse output to voter records.

The generic MLX engine turns page images into text. These functions parse that
text into the canonical electoral-roll fields.

- :func:`parse_voters` / :func:`dedupe_voters` parse the base model's verbose HTML.
- :func:`parse_terse` parses the distilled terse model's one-line-per-voter output.
- :func:`to_terse` renders voter dicts back to the terse training target.
- :func:`resolve_terse_model` locates (or downloads) the terse model weights.
"""

import os
import re
from pathlib import Path

#: Hugging Face repository for the distilled roll model.
TERSE_REPO = os.environ.get("SAVITR_TERSE_REPO", "gojiberries/savitr")
TERSE_REVISION = os.environ.get(
    "SAVITR_TERSE_REVISION", "c850ccd21031bb86595f1ba5f9679e6b401ec04f"
)


def resolve_terse_model(local: str = "models/surya-terse-8bit") -> str:
    """Resolve a local model directory or download the pinned Hub snapshot."""
    if Path(local).is_dir():
        return local
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=TERSE_REPO, revision=TERSE_REVISION)


# Anchor on ``Name :`` so parsing does not depend on table nesting.
TAG = re.compile(r"<[^>]+>")
EPIC = re.compile(r"\b([A-Z]{2,3}\d{6,8})\b")
SERIAL = re.compile(r"(\d{1,4})")
NAME = re.compile(
    r"Name\s*:\s*(.*?)(?:Father|Husband|Mother|House|Age|Gender|$)",
    re.IGNORECASE | re.DOTALL,
)
REL = re.compile(
    r"(Father|Husband|Mother)'?s?\s*Name\s*:\s*(.*?)(?:House|Age|Gender|$)",
    re.IGNORECASE | re.DOTALL,
)
HOUSE = re.compile(
    r"House\s*Number\s*:\s*(.*?)(?:Age|Gender|$)",
    re.IGNORECASE | re.DOTALL,
)
AGE = re.compile(r"Age\s*:\s*(\d{1,3})", re.IGNORECASE)
GENDER = re.compile(r"Gender\s*:\s*(Male|Female|Third|Other)", re.IGNORECASE)
REL_CODE = {"father": "F", "husband": "H", "mother": "M"}
SEX_CODE = {"male": "M", "female": "F", "third": "T", "other": "T"}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", TAG.sub(" ", s)).strip(" :")


def parse_voters(html: str) -> list[dict]:
    """Split page HTML into voter records using the ``Name :`` anchor."""

    # Exclude relationship fields and common heading phrases.
    def _is_header(m: re.Match) -> bool:
        pre = html[max(0, m.start() - 16) : m.start()]
        post = html[m.end() : m.end() + 24]
        return bool(
            re.search(r"(Father|Husband|Mother)'?s?\s*$", pre, re.IGNORECASE)
            or re.search(r"\band\s*$", pre, re.IGNORECASE)
            or re.match(r"\s*and\s+Reservation", post, re.IGNORECASE)
        )

    name_iters = [
        match
        for match in re.finditer(r"Name\s*:", html, re.IGNORECASE)
        if not _is_header(match)
    ]
    voters = []
    for i, m in enumerate(name_iters):
        blob = html[
            m.start() : (
                name_iters[i + 1].start() if i + 1 < len(name_iters) else len(html)
            )
        ]
        nm = NAME.search(blob)
        name = _clean(nm.group(1)) if nm else ""
        am, gm = AGE.search(blob), GENDER.search(blob)
        # A voter record has a text name and at least an age or gender.
        if not name or name[0].isdigit() or not (am or gm):
            continue
        rel, hm = REL.search(blob), HOUSE.search(blob)
        pre = html[max(0, m.start() - 28) : m.start()]
        lead = re.search(r"(#)?\s*(\d{1,4})\s*(?:<br\s*/?>)\s*(?:<[^>]*>\s*)*$", pre)
        voters.append(
            {
                "elector_name": name,
                "father_or_husband_name": _clean(rel.group(2)) if rel else "",
                "relationship": REL_CODE.get(rel.group(1).lower(), "") if rel else "",
                "house_no": _clean(hm.group(1)) if hm else "",
                "age": am.group(1) if am else "",
                "sex": SEX_CODE.get(gm.group(1).lower(), "") if gm else "",
                "number": lead.group(2) if lead else "",
                "original_or_amendment": "amendment"
                if (lead and lead.group(1))
                else "original",
            }
        )

    # Align EPICs and bold serials in document order.
    epics = EPIC.findall(html)
    bold = re.findall(r"<b>\s*(#)?\s*(\d{1,4})\s*</b>", html)
    for k, v in enumerate(voters):
        v["id"] = epics[k] if k < len(epics) else ""
        if not v["number"] and k < len(bold):
            v["number"] = bold[k][1]
            v["original_or_amendment"] = "amendment" if bold[k][0] else "original"
    return voters


def dedupe_voters(voters: list[dict]) -> list[dict]:
    """Collapse duplicated rows, keeping the fullest record per voter.

    Keys by EPIC id when present, then serial number, then identity
    (name + relation + age). A serial prevents two legitimate voters with the
    same name and age from being collapsed. The fullest repeated record wins.
    """

    def score(v: dict) -> int:
        return sum(
            1
            for k in (
                "id",
                "number",
                "father_or_husband_name",
                "house_no",
                "age",
                "sex",
            )
            if v.get(k)
        )

    best: dict = {}
    for v in voters:
        if v.get("id"):
            key = ("epic", v["id"])
        elif v.get("number"):
            key = (
                "number",
                v.get("original_or_amendment", "original"),
                v["number"],
            )
        else:
            key = (
                "identity",
                re.sub(r"\s+", " ", v["elector_name"].lower()).strip(),
                re.sub(
                    r"\s+", " ", v.get("father_or_husband_name", "").lower()
                ).strip(),
                v.get("age", ""),
            )
        if key not in best or score(v) > score(best[key]):
            best[key] = v
    out = list(best.values())
    return sorted(
        out, key=lambda v: int(v["number"]) if v.get("number", "").isdigit() else 1e9
    )


# Terse distillation target and model output.
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
    "one per line, columns: serial|epic|name|relation(F/H/M)|relation_name|"
    "house|age|sex"
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

    The model can drop the relation-code column. Anchoring the EPIC at the
    front and age and sex at the end keeps other fields aligned.
    """
    voters = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        while parts and parts[0] == "":  # serial column is usually blank
            parts.pop(0)
        if not parts:
            continue
        # Gold targets lead with a numeric serial; model output may omit it.
        number = parts.pop(0) if parts[0].isdigit() else ""
        if len(parts) < 2:
            continue
        v = dict.fromkeys(TERSE_COLS, "")
        v["number"] = number
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
        # EPIC-less rows must still look like complete voter records. This
        # rejects footer text and decode-loop fragments.
        if not v["id"]:
            nm = v["elector_name"]
            if (
                _EPIC_TOK.search(line)
                or not nm[:1].isalpha()
                or not (v["age"] and v["sex"])
            ):
                continue
        v["original_or_amendment"] = "original"
        voters.append(v)
    return voters
