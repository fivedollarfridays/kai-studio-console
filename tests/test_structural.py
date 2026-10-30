"""Structural guarantees: no imports from the private development repo, no stray network access."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent / "kai_slice"

FORBIDDEN_IMPORTS = {"ops", "kai_studio"}
NETWORK_MODULES = {"socket", "urllib", "requests", "httpx", "http"}
# The one documented exception: `kai_slice serve`, which may only LISTEN on
# 127.0.0.1. model.py's model fetch goes through
# faster-whisper's own downloader, not a raw NETWORK_MODULES import, so it
# needs no entry here (see test_model_py_imports_no_raw_network_module below) --
# a real urllib/socket/requests/httpx import added to model.py later would
# trip test_network_only_in_documented_exceptions like any other module.
NETWORK_ALLOWED = {"serve.py"}

# serve.py's exception is scoped to the LISTEN side: adding
# `http` above would otherwise either trip on serve.py's own `http.server` or
# -- if serve.py were simply exempt -- let an `http.client` / `urllib.request`
# import there go unseen. So serve.py may import only these network modules,
# and may make none of the outbound calls below; any other module still may
# import no network module at all.
SERVE_LISTEN_SIDE = ("http.server", "urllib.parse", "socket")
OUTBOUND_CALLS = {
    "urlopen",
    "create_connection",
    "connect",
    "connect_ex",
    "HTTPConnection",
    "HTTPSConnection",
    "sendto",
}


def _modules() -> list[Path]:
    return sorted(_PKG.glob("*.py"))


def _top_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _dotted_imports(tree: ast.AST) -> set[str]:
    """Full dotted names: ``import http.client`` -> ``http.client``, and
    ``from http import client`` -> ``http`` plus ``http.client``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _serve_violations(source: str) -> list[str]:
    tree = ast.parse(source)
    bad = [
        name
        for name in _dotted_imports(tree)
        if name.split(".")[0] in NETWORK_MODULES
        and not any(name == ok or name.startswith(ok + ".") for ok in SERVE_LISTEN_SIDE)
        and name not in ("http", "urllib")  # the parent of an allowed `from x import y`
    ]
    for node in ast.walk(tree):
        called = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
        if called in OUTBOUND_CALLS:
            bad.append(called)
    return bad


def test_serve_only_listens():
    assert _serve_violations((_PKG / "serve.py").read_text()) == []


@pytest.mark.parametrize(
    "source",
    [
        "import http.client",
        "from http import client",
        "from http.client import HTTPConnection",
        "import urllib.request",
        "from urllib.request import urlopen",
        "import requests",
        "import http\nhttp.client.HTTPConnection('example.com')",
        "import urllib\nurllib.request.urlopen('http://example.com')",
        "import socket\nsocket.create_connection(('example.com', 80))",
        "import socket\ns = socket.socket()\ns.connect(('example.com', 80))",
    ],
)
def test_the_serve_rule_catches_outbound_use(source):
    assert _serve_violations(source), source


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_internal_package_imports(path):
    imports = _top_imports(ast.parse(path.read_text()))
    assert not (imports & FORBIDDEN_IMPORTS), f"{path.name} imports {imports & FORBIDDEN_IMPORTS}"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_network_only_in_documented_exceptions(path):
    imports = _top_imports(ast.parse(path.read_text()))
    net = imports & NETWORK_MODULES
    if net:
        assert path.name in NETWORK_ALLOWED, f"{path.name} reaches the network ({net})"


def test_model_py_imports_no_raw_network_module():
    # model.py is NOT in NETWORK_ALLOWED: its fetch is delegated to
    # faster-whisper's own downloader, never a raw NETWORK_MODULES import.
    # If one were added later, test_network_only_in_documented_exceptions
    # (above) would fail because model.py is no longer an allowed exception.
    imports = _top_imports(ast.parse((_PKG / "model.py").read_text()))
    assert not (imports & NETWORK_MODULES)
