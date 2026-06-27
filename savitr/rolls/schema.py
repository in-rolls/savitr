"""Canonical in-rolls voter CSV schema (vendored from parse_unsearchable_rolls).

Kept byte-compatible with parse_unsearchable_rolls/scripts/manipur/parse_manipur_2025.py so
savitr's CSV output drops into the same downstream pipeline.
"""

import re

# Canonical in-rolls columns + Manipur-specific metadata the form provides.
COLUMNS = [
    "number",
    "id",
    "elector_name",
    "father_or_husband_name",
    "relationship",
    "house_no",
    "age",
    "sex",
    "ac_name",
    "parl_constituency",
    "part_no",
    "year",
    "state",
    "filename",
    "main_town",
    "police_station",
    "mandal",
    "revenue_division",
    "district",
    "pin_code",
    "polling_station_name",
    "polling_station_address",
    "net_electors_male",
    "net_electors_female",
    "net_electors_third_gender",
    "net_electors_total",
    "original_or_amendment",
]

STATE = "Manipur"
YEAR = "2025"


def ac_part_from_filename(name: str) -> tuple[str, str]:
    """AC01_part001_final_ENG.pdf -> ('1', '1')."""
    m = re.search(r"AC(\d+)_part(\d+)", name, re.I)
    return (str(int(m.group(1))), str(int(m.group(2)))) if m else ("", "")
