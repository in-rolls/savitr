"""Where savitr looks for a model, and what it says when there is not one.

The first tests in this repo. They exist because model resolution is the one piece of savitr that
every entry point depends on and no platform can check: it runs before MLX is imported, so it is
testable on any machine, with no Apple Silicon, no network and no model — and it was wrong for as
long as it went untested. `MLXSuryaOCR()` with no argument raised `RepositoryNotFoundError` for
`models/models/surya-mlx-4bit`, a Hub repo that has never existed, because a bare relative path was
handed to a loader that reads unknown strings as repo ids.

The assertion that matters is not that the error is raised. It is that the message names
`mlx_vlm convert`, because that is the whole difference between the old behaviour and the new one.
"""

import os
from unittest.mock import patch

import pytest

from savitr.mlx_ocr import BASE_REPO, base_model_path
from savitr.rolls.parse import TERSE_REPO, TERSE_REVISION, resolve_terse_model


@pytest.fixture(autouse=True)
def _no_ambient_model(tmp_path, monkeypatch):
    """Run from an empty directory with no env override, so nothing on the machine leaks in."""
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
    assert os.path.samefile(base_model_path(), model)


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
    # and it says where it looked, so a maintainer with a model elsewhere knows why it was missed
    assert "models/surya-mlx-4bit" in message


def test_the_error_points_at_the_roll_model_too():
    """Most people reaching for savitr want rolls, and that model does download itself."""
    with pytest.raises(FileNotFoundError) as raised:
        base_model_path()
    assert "resolve_terse_model" in str(raised.value)


def test_a_local_terse_model_is_used_without_touching_the_network(tmp_path):
    """`resolve_terse_model` only downloads when there is no local copy - assert the local path,
    because the download path needs the network and this suite must not."""
    model = tmp_path / "terse"
    model.mkdir()
    with patch("huggingface_hub.snapshot_download") as download:
        assert resolve_terse_model(str(model)) == str(model)
    download.assert_not_called()


def test_a_missing_terse_model_uses_the_pinned_hub_revision(tmp_path):
    missing = tmp_path / "missing"
    with patch("huggingface_hub.snapshot_download", return_value="/cache/savitr") as download:
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
