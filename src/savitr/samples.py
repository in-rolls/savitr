"""Locate bundled sample data (a small public electoral-roll PDF for the quickstart)."""

from importlib.resources import files


def sample_roll_path() -> str:
    """Return the filesystem path to the bundled sample roll PDF.

    The file contains two public Manipur 2025 English roll pages, enough for
    an end-to-end ``savitr ocr`` example.
    """
    return str(files("savitr").joinpath("data/sample_roll.pdf"))
