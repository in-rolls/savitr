"""The electoral-roll application: parsing, vendored schema, and the PDF -> CSV pipeline."""

from savitr.rolls import fields  # noqa: F401
from savitr.rolls.parse import (  # noqa: F401
    TERSE_PROMPT,
    dedupe_voters,
    parse_terse,
    parse_voters,
    resolve_terse_model,
    to_terse,
)
from savitr.rolls.schema import (  # noqa: F401
    COLUMNS,
    STATE,
    YEAR,
    ac_part_from_filename,
)

__all__ = [
    "fields",
    "parse_voters",
    "dedupe_voters",
    "parse_terse",
    "to_terse",
    "resolve_terse_model",
    "TERSE_PROMPT",
    "COLUMNS",
    "STATE",
    "YEAR",
    "ac_part_from_filename",
]
