"""Structural guarantee: scripts/fetch-model is the only entry point that
can touch the network; demo/make-sample/make-stand-in-sample never import
a network module or the model-download path directly (mirrors
test_structural.py's package-level guard, extended to the extensionless
scripts/ entry points this task adds)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _ROOT / "scripts"
NETWORK_MODULES = {"socket", "urllib", "requests", "httpx", "huggingface_hub"}
NETWORK_ALLOWED = "fetch-model"


def _entry_points() -> list[Path]:
    scripts = sorted(p for p in _SCRIPTS_DIR.iterdir() if p.is_file())
    return scripts + [_ROOT / "demo"]


def _top_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _imports_model_fetch(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "kai_slice.model":
            return True
    return False


@pytest.mark.parametrize("path", _entry_points(), ids=lambda p: p.name)
def test_only_fetch_model_can_reach_the_network(path):
    tree = ast.parse(path.read_text())
    net = _top_imports(tree) & NETWORK_MODULES
    touches_model_download = _imports_model_fetch(tree)
    if path.name == NETWORK_ALLOWED:
        assert touches_model_download, "fetch-model must call kai_slice.model.fetch_model"
    else:
        assert not net, f"{path.name} imports network modules {net}"
        assert not touches_model_download, f"{path.name} touches the model-download network path"


def test_fetch_model_is_the_only_allowed_entry_point():
    names = {p.name for p in _entry_points()}
    assert NETWORK_ALLOWED in names
