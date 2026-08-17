"""Convert Surya output to canonical electoral-roll records."""

from . import fields
from .parse import (
    TERSE_PROMPT,
    dedupe_voters,
    parse_terse,
    parse_voters,
    resolve_terse_model,
    to_terse,
)
from .schema import (
    COLUMNS,
    STATE,
    YEAR,
    ac_part_from_filename,
)

__all__ = [
    "COLUMNS",
    "STATE",
    "TERSE_PROMPT",
    "YEAR",
    "ac_part_from_filename",
    "dedupe_voters",
    "fields",
    "parse_terse",
    "parse_voters",
    "resolve_terse_model",
    "to_terse",
]
