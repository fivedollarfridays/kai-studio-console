"""Red-first: scripts/fetch-model, the one command that touches the network.

Loaded via spec_from_file_location (the file has no .py suffix, by design —
it is a direct ``./scripts/fetch-model`` entry point, not an importable
module). The downloader itself is exercised in test_model_fetch.py; here we
only prove the script's own wiring: default model/dest, overrides, and the
printed size line.
"""

from __future__ import annotations

import ast
from pathlib import Path

from conftest import load_script

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fetch-model"


def _load():
    return load_script("fetch-model")


def test_default_model_is_imported_from_transcribe_not_duplicated():
    """Cleanup: the script must import transcribe.DEFAULT_MODEL rather than
    carrying its own ``DEFAULT_MODEL = "small"`` literal that could drift
    from the real default."""
    tree = ast.parse(_SCRIPT_PATH.read_text())
    assigns_default_model = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DEFAULT_MODEL" for t in node.targets)
        for node in ast.walk(tree)
    )
    imports_default_model = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "kai_slice.transcribe"
        and any(alias.name == "DEFAULT_MODEL" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert not assigns_default_model
    assert imports_default_model


def test_models_root_is_imported_from_kai_slice_model_not_duplicated():
    """Cleanup: the script must import MODELS_ROOT from kai_slice.model
    rather than recomputing it, so resolve_model (used on the CLI run path)
    and this fetch step always agree on where a model lives."""
    tree = ast.parse(_SCRIPT_PATH.read_text())
    assigns_models_root = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MODELS_ROOT" for t in node.targets)
        for node in ast.walk(tree)
    )
    imports_models_root = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "kai_slice.model"
        and any(alias.name == "MODELS_ROOT" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert not assigns_models_root
    assert imports_models_root


def test_default_model_and_dest_resolve_under_package_models_dir(tmp_path, monkeypatch):
    module = _load()
    calls = []

    def fake_fetch_model(size_or_id, dest_dir, **kwargs):
        calls.append((size_or_id, Path(dest_dir)))
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "model.bin").write_bytes(b"x" * 10)
        return dest_dir

    monkeypatch.setattr(module, "fetch_model", fake_fetch_model)
    monkeypatch.setattr(module, "MODELS_ROOT", tmp_path / "models")

    rc = module.main([])

    assert rc == 0
    assert calls == [("small", tmp_path / "models" / "small")]


def test_model_override_is_passed_through(tmp_path, monkeypatch):
    module = _load()
    calls = []

    def fake_fetch_model(size_or_id, dest_dir, **kwargs):
        calls.append(size_or_id)
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        (Path(dest_dir) / "model.bin").write_bytes(b"x" * 10)
        return Path(dest_dir)

    monkeypatch.setattr(module, "fetch_model", fake_fetch_model)
    monkeypatch.setattr(module, "MODELS_ROOT", tmp_path / "models")

    rc = module.main(["--model", "base"])

    assert rc == 0
    assert calls == ["base"]


def test_an_unverifiable_pin_refuses_by_name_not_traceback(tmp_path, monkeypatch, capsys):
    from kai_slice.model import ModelIntegrityError, ModelRevisionError

    module = _load()
    for error in (ModelRevisionError("revision 'abc' unverified"), ModelIntegrityError("tampered")):

        def refuse(size_or_id, dest_dir, _error=error, **kwargs):
            raise _error

        monkeypatch.setattr(module, "fetch_model", refuse)
        monkeypatch.setattr(module, "MODELS_ROOT", tmp_path / "models")

        assert module.main([]) == 1
        assert str(error) in capsys.readouterr().err


def test_prints_the_downloaded_size(tmp_path, monkeypatch, capsys):
    module = _load()

    def fake_fetch_model(size_or_id, dest_dir, **kwargs):
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "model.bin").write_bytes(b"x" * 1024)
        return dest_dir

    monkeypatch.setattr(module, "fetch_model", fake_fetch_model)
    monkeypatch.setattr(module, "MODELS_ROOT", tmp_path / "models")

    module.main([])
    out = capsys.readouterr().out
    assert "small" in out
    assert "models" in out


def _with_fake_downloader(module, monkeypatch, calls: list) -> None:
    """Run the REAL fetch_model (so its refusals are exercised), with a fake
    downloader injected -- no network."""
    from conftest import fake_pins, write_fake_model

    from kai_slice import model

    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        calls.append(size_or_id)
        return write_fake_model(output_dir)

    def fetch(size_or_id, dest_dir, **kwargs):
        pins = fake_pins(model.DEFAULT_MODEL_REVISION)
        return model.fetch_model(
            size_or_id, dest_dir, downloader=fake, pinned_sha256=pins, **kwargs
        )

    monkeypatch.setattr(module, "fetch_model", fetch)


def test_an_unpinned_model_refuses_without_the_flag(tmp_path, monkeypatch, capsys):
    module = _load()
    calls = []
    _with_fake_downloader(module, monkeypatch, calls)
    monkeypatch.setattr(module, "MODELS_ROOT", tmp_path / "models")

    rc = module.main(["--model", "base"])

    assert rc == 1
    assert "--allow-unpinned" in capsys.readouterr().err
    assert calls == []


def test_allow_unpinned_fetches_with_a_warning(tmp_path, monkeypatch, capsys):
    module = _load()
    calls = []
    _with_fake_downloader(module, monkeypatch, calls)
    monkeypatch.setattr(module, "MODELS_ROOT", tmp_path / "models")

    rc = module.main(["--model", "base", "--allow-unpinned"])

    assert rc == 0
    assert calls == ["base"]
    assert "warning" in capsys.readouterr().err.lower()


def test_the_default_model_needs_no_flag_and_prints_no_warning(tmp_path, monkeypatch, capsys):
    module = _load()
    calls = []
    _with_fake_downloader(module, monkeypatch, calls)
    monkeypatch.setattr(module, "MODELS_ROOT", tmp_path / "models")

    assert module.main([]) == 0
    assert calls == ["small"]
    assert "warning" not in capsys.readouterr().err.lower()


def test_a_non_empty_dest_refuses_by_name_not_traceback(tmp_path, monkeypatch, capsys):
    module = _load()
    calls = []
    _with_fake_downloader(module, monkeypatch, calls)
    dest = tmp_path / "mine"
    dest.mkdir()
    (dest / "thesis.docx").write_text("mine")

    rc = module.main(["--dest", str(dest)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "thesis.docx" in err and "Traceback" not in err
    assert calls == [] and (dest / "thesis.docx").read_text() == "mine"
