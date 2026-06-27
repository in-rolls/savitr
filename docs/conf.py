"""Sphinx configuration for the savitr documentation site (furo + MyST + autodoc)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

project = "savitr"
author = "in-rolls"
try:
    release = _version("savitr")
except PackageNotFoundError:  # building without an install
    release = "0.1.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

# The heavy / Apple-Silicon-only deps are lazy-imported in the code, but mock them so the docs
# build on any runner (ubuntu CI) without installing mlx/mlx-vlm/surya.
autodoc_mock_imports = ["mlx_vlm", "mlx", "pdf2image", "huggingface_hub", "surya", "openai"]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = True

html_theme = "furo"
html_title = "savitr"

myst_enable_extensions = ["colon_fence"]
exclude_patterns = ["_build"]
