"""Locate bundled sample data (a small public electoral-roll PDF for the quickstart)."""

from importlib.resources import files


def sample_roll_path() -> str:
    """Return the filesystem path to the bundled sample roll PDF.

    Two held-out (public) Manipur 2025 English roll pages — enough to run ``savitr ocr`` end to end
    without supplying your own PDF. Works from any install (``savitr ocr "$(savitr sample)"``).
    """
    return str(files("savitr").joinpath("data/sample_roll.pdf"))
