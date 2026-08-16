"""A floor, not coverage.

This repo had no tests at all, and `pytest` did not report that: with no
`testpaths` it walked the whole tree, collected `training/test_adapter_local.py`
-- a manual sanity script that imports `peft` from the `train` extra -- and
aborted on the import. The failure looked like a broken test rather than an
absent suite.

These assertions are deliberately shallow. They catch the breakage that costs
users the most and is easiest to ship unnoticed: a package that no longer
imports, or a name promised in `__all__` that is not there. They are not a
substitute for testing what the parsers actually do.
"""

import importlib

import savitr


def test_package_imports() -> None:
    """The package imports and reports a version."""
    assert savitr.__version__


def test_every_name_in_dunder_all_resolves() -> None:
    """Every name __all__ promises is actually reachable."""
    missing = [name for name in savitr.__all__ if not hasattr(savitr, name)]
    assert missing == [], f"promised by __all__ but absent: {missing}"


def test_cli_module_imports() -> None:
    """The console script's entry point, which packaging problems break first."""
    assert importlib.import_module("savitr.cli") is not None
