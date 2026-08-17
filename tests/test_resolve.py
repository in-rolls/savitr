"""Test local and Hugging Face model resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from savitr.mlx_ocr import BASE_REPO, base_model_path
from savitr.rolls.parse import TERSE_REPO, TERSE_REVISION, resolve_terse_model


@pytest.fixture(autouse=True)
def _no_ambient_model(tmp_path, monkeypatch):
    """Prevent ambient local models from affecting tests."""
    monkeypatch.delenv("SAVITR_BASE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)


def test_an_explicit_path_that_exists_is_used(tmp_path):
    model = tmp_path / "my-model"
    model.mkdir()
    assert base_model_path(str(model)) == str(model)


def test_the_environment_override_is_honoured(tmp_path, monkeypatch):
    model = tmp_path / "from-env"
    model.mkdir()
    monkeypatch.setenv("SAVITR_BASE_PATH", str(model))
    assert base_model_path() == str(model)


def test_the_repo_layout_still_resolves(tmp_path):
    """bench/ and training/ assume `models/surya-mlx-4bit` relative to the repo root."""
    model = tmp_path / "models" / "surya-mlx-4bit"
    model.mkdir(parents=True)
    assert Path(base_model_path()).samefile(model)


def test_an_explicit_path_wins_over_the_environment(tmp_path, monkeypatch):
    chosen, other = tmp_path / "chosen", tmp_path / "other"
    chosen.mkdir()
    other.mkdir()
    monkeypatch.setenv("SAVITR_BASE_PATH", str(other))
    assert base_model_path(str(chosen)) == str(chosen)


def test_with_no_model_anywhere_the_error_names_the_convert_command():
    """The point of the whole change: an error a reader can act on."""
    with pytest.raises(FileNotFoundError) as raised:
        base_model_path()
    message = str(raised.value)
    assert "mlx_vlm convert" in message
    assert BASE_REPO in message
    # Include searched locations so users can diagnose misplaced models.
    assert "models/surya-mlx-4bit" in message


def test_the_error_points_at_the_roll_model_too():
    """Mention the downloadable model in base-model errors."""
    with pytest.raises(FileNotFoundError) as raised:
        base_model_path()
    assert "resolve_terse_model" in str(raised.value)


def test_a_local_terse_model_is_used_without_touching_the_network(tmp_path):
    """Prefer a local model without calling the Hub."""
    model = tmp_path / "terse"
    model.mkdir()
    with patch("huggingface_hub.snapshot_download") as download:
        assert resolve_terse_model(str(model)) == str(model)
    download.assert_not_called()


def test_a_missing_terse_model_uses_the_pinned_hub_revision(tmp_path):
    missing = tmp_path / "missing"
    with patch(
        "huggingface_hub.snapshot_download", return_value="/cache/savitr"
    ) as download:
        assert resolve_terse_model(str(missing)) == "/cache/savitr"
    download.assert_called_once_with(repo_id=TERSE_REPO, revision=TERSE_REVISION)


def test_the_hub_revision_is_an_immutable_commit():
    assert TERSE_REPO == "gojiberries/savitr"
    assert len(TERSE_REVISION) == 40
    assert set(TERSE_REVISION) <= set("0123456789abcdef")


@pytest.mark.live
def test_the_pinned_hub_revision_exists():
    from huggingface_hub import list_repo_files

    assert list_repo_files(TERSE_REPO, revision=TERSE_REVISION)
