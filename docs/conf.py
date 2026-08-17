"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(
    globals(),
    autodoc_mock_imports=["mlx", "mlx_vlm"],
)
