"""No file under the public tree carries a private development path pattern.

Path patterns ONLY — the real-name check needs private data, so it lives in
the source project's export gate and in its own tests/public/, never here.
Needles are assembled from fragments so this file does not match itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TREE = Path(__file__).resolve().parent.parent

# Built from fragments on purpose (see module docstring).
NEEDLES = [
    "E:" + "/",
    "/mnt/" + "nas",
    "/mnt/" + "c/",
    "/mnt/" + "e/",  # the render machine's media mount, as in the export gate
    "/" + "home/",
    "kai-studio" + "-media",
    "ops/" + "data",
]

# Build artifacts and dependency trees a clone grows locally: a judge who runs
# `python -m venv .venv && pip install -e .` inside the clone (the documented
# "clone it, run it" workflow) must not get a false hit off a third-party
# package that legitimately carries one of these path shapes in its own source.
SKIP_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "site-packages",
    "node_modules",
    ".git",
}

# `demo-out/` (run output) and `models/` (fetched model weights)
# only ever grow at the TREE root -- an authored `kai_slice/models/` would be
# source, so this pair is matched on rel.parts[0], never at any depth.
TOP_LEVEL_SKIP_DIRS = {"demo-out", "models"}
SKIP_NAMES = {Path(__file__).name}


def _text_files() -> list[Path]:
    out = []
    for path in _TREE.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(_TREE)
        if rel.parts[0] in TOP_LEVEL_SKIP_DIRS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_NAMES:
            continue
        out.append(path)
    return out


def test_demo_out_is_excluded_from_the_scrub_walk(tmp_path, monkeypatch):
    # `./demo` writes `demo-out/ingest/manifest.json` carrying the absolute
    # host path of the card it ingested -- a run artifact, not authored
    # source, so the scrub walk must never look inside it.
    monkeypatch.setattr(sys.modules[__name__], "_TREE", tmp_path)
    run_artifact = tmp_path / "demo-out" / "ingest" / "manifest.json"
    run_artifact.parent.mkdir(parents=True)
    absolute_host_path = "/" + "home/someuser/cards/card-01"
    run_artifact.write_text('{"source": "' + absolute_host_path + '"}', encoding="utf-8")
    assert run_artifact not in _text_files()


def test_models_skip_is_top_level_only(tmp_path, monkeypatch):
    # `models/` and `demo-out/` only ever grow at the TREE root. An authored
    # file nested under `kai_slice/models/` is source, not a fetched model or
    # run artifact, and must still be scrubbed for private path patterns.
    monkeypatch.setattr(sys.modules[__name__], "_TREE", tmp_path)
    absolute_host_path = "/" + "home/someuser/cards/card-01"

    nested_authored = tmp_path / "kai_slice" / "models" / "loader.py"
    nested_authored.parent.mkdir(parents=True)
    nested_authored.write_text('"""' + absolute_host_path + '"""', encoding="utf-8")

    top_level_demo_out = tmp_path / "demo-out" / "manifest.json"
    top_level_demo_out.parent.mkdir(parents=True)
    top_level_demo_out.write_text(absolute_host_path, encoding="utf-8")

    top_level_models = tmp_path / "models" / "weights.bin"
    top_level_models.parent.mkdir(parents=True)
    top_level_models.write_text(absolute_host_path, encoding="utf-8")

    files = _text_files()
    assert nested_authored in files
    assert top_level_demo_out not in files
    assert top_level_models not in files


def test_no_private_path_patterns_in_public_tree():
    offenders = []
    for path in _text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for needle in NEEDLES:
            if needle in text:
                offenders.append(f"{path.relative_to(_TREE)}: {needle!r}")
    assert not offenders, "private path patterns leaked into the public tree:\n" + "\n".join(
        offenders
    )
