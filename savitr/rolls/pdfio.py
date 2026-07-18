"""PDF rendering helpers with a friendly poppler check (shared by `ocr` and `parse-rolls`)."""

import shutil


def require_poppler() -> None:
    """Exit with an install hint if poppler's ``pdfinfo``/``pdftoppm`` aren't on PATH.

    ``pdf2image`` shells out to poppler, which isn't a pip dependency; without it every PDF
    command dies with an opaque ``PDFInfoNotInstalledError``. Turn that into one clear line.
    """
    if shutil.which("pdfinfo") and shutil.which("pdftoppm"):
        return
    raise SystemExit(
        "savitr needs poppler to read PDFs (pdfinfo/pdftoppm not found on PATH).\n"
        "  macOS:   brew install poppler\n"
        "  Debian:  sudo apt-get install poppler-utils\n"
        "  conda:   conda install -c conda-forge poppler"
    )


def page_count(pdf_path: str) -> int:
    """Return the number of pages in ``pdf_path`` (validates poppler first)."""
    require_poppler()
    from pdf2image.pdf2image import pdfinfo_from_path

    return int(pdfinfo_from_path(pdf_path)["Pages"])


def render_page(pdf_path: str, page_1based: int, dpi: int, out_png: str) -> str:
    """Render one 1-based page of ``pdf_path`` to ``out_png`` (RGB) and return the path."""
    from pdf2image import convert_from_path

    img = convert_from_path(pdf_path, dpi=dpi, first_page=page_1based, last_page=page_1based)[0]
    img.convert("RGB").save(out_png)
    return out_png
