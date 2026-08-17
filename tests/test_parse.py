from pathlib import Path

import pytest

from savitr.rolls import pipeline
from savitr.rolls.parse import dedupe_voters, parse_terse, parse_voters, to_terse


def voter(**overrides: str) -> dict[str, str]:
    row = {
        "number": "31",
        "id": "ABC123456",
        "elector_name": "Asha",
        "relationship": "F",
        "father_or_husband_name": "Ram",
        "house_no": "12",
        "age": "30",
        "sex": "F",
    }
    row.update(overrides)
    return row


def test_terse_round_trip_preserves_every_column() -> None:
    original = voter()

    parsed = parse_terse(to_terse([original]))

    assert [{key: row[key] for key in original} for row in parsed] == [original]


def test_terse_parser_keeps_tail_fields_aligned_without_relation_code() -> None:
    parsed = parse_terse("ABC123456|Asha|Ram|12|30|F")

    assert parsed[0] == {
        **voter(number="", relationship=""),
        "original_or_amendment": "original",
    }


def test_dedupe_uses_serial_before_fallback_identity() -> None:
    same_identity = voter(id="", number="32")

    assert len(dedupe_voters([voter(id=""), same_identity])) == 2


def test_dedupe_keeps_fullest_copy_of_a_repeated_serial() -> None:
    incomplete = voter(id="", house_no="", sex="")
    complete = voter(id="")

    assert dedupe_voters([incomplete, complete]) == [complete]


def test_html_parser_does_not_truncate_legitimate_same_name_and_age() -> None:
    html = """
    <td><b>31</b> ABC123456 Name : Asha<br>Father's Name : Ram<br>
    House Number : 12<br>Age : 30 Gender : Female</td>
    <td><b>32</b> DEF123456 Name : Asha<br>Husband's Name : Dev<br>
    House Number : 14<br>Age : 30 Gender : Female</td>
    """

    parsed = parse_voters(html)

    assert [row["number"] for row in parsed] == ["31", "32"]
    assert [row["id"] for row in parsed] == ["ABC123456", "DEF123456"]


def test_pipeline_removes_temporary_image_when_ocr_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[Path] = []

    def render(_pdf: str, _page: int, _dpi: int, output: str) -> str:
        path = Path(output)
        path.write_bytes(b"image")
        rendered.append(path)
        return output

    class BrokenEngine:
        def ocr_image(self, _path: str) -> tuple[str, int]:
            raise RuntimeError("inference failed")

    monkeypatch.setattr(pipeline, "page_count", lambda _pdf: 1)
    monkeypatch.setattr(pipeline, "render_page", render)

    with pytest.raises(RuntimeError, match="inference failed"):
        pipeline.parse_pdf_mlx(BrokenEngine(), "AC01_part001_final_ENG.pdf", 192)

    assert rendered
    assert not rendered[0].exists()
