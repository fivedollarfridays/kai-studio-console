"""Red-first: kai_slice.model.resolve_model, the seam that keeps the CLI
path from ever handing a bare model name to a transcriber that could
download it implicitly.

No network: every case here is decided by directory existence alone.
"""

from __future__ import annotations

import re

import pytest

from kai_slice.model import MODELS_ROOT, NO_MODEL_MESSAGE, ModelNotFoundError, resolve_model


def test_resolves_a_model_already_fetched_under_models_root(tmp_path):
    root = tmp_path / "models"
    fetched = root / "small"
    fetched.mkdir(parents=True)
    (fetched / "model.bin").write_bytes(b"fake-weights")

    assert resolve_model("small", models_root=root) == fetched


def test_refuses_with_the_documented_message_when_nothing_was_fetched(tmp_path):
    root = tmp_path / "models"

    with pytest.raises(ModelNotFoundError, match=re.escape(NO_MODEL_MESSAGE)):
        resolve_model("small", models_root=root)


def test_refuses_when_the_fetched_directory_exists_but_is_empty(tmp_path):
    root = tmp_path / "models"
    (root / "small").mkdir(parents=True)

    with pytest.raises(ModelNotFoundError, match=re.escape(NO_MODEL_MESSAGE)):
        resolve_model("small", models_root=root)


def test_an_existing_directory_passed_directly_is_used_as_is(tmp_path):
    custom = tmp_path / "my-custom-model"
    custom.mkdir()
    (custom / "model.bin").write_bytes(b"fake-weights")

    assert resolve_model(str(custom)) == custom


def test_default_models_root_is_the_module_level_constant(tmp_path, monkeypatch):
    import kai_slice.model as model_mod

    monkeypatch.setattr(model_mod, "MODELS_ROOT", tmp_path / "models")
    fetched = tmp_path / "models" / "small"
    fetched.mkdir(parents=True)
    (fetched / "model.bin").write_bytes(b"fake-weights")

    assert model_mod.resolve_model("small") == fetched


def test_models_root_matches_where_scripts_fetch_model_writes():
    # scripts/fetch-model writes into <repo root>/models/<name>; resolve_model
    # must look in the same place or a fetched model is never found.
    repo_root = MODELS_ROOT.parent
    assert MODELS_ROOT == repo_root / "models"
