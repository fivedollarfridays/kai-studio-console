"""The committed console build (``console/``) inlines nothing as
a base64 ``data:`` URI, and carries no orphaned asset: every file under
``console/assets/`` is referenced by ``index.html`` or by a file it loads.

The bundle is generated (``npm run build:public`` in the source project) and
committed as-is, so this guards the bytes that actually ship.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CONSOLE = Path(__file__).resolve().parent.parent / "console"
ASSETS = CONSOLE / "assets"
BASE64_DATA_URI = re.compile(rb"data:[^,\"')\s]*;base64,")
TEXT_SUFFIXES = {".html", ".css", ".js", ".json", ".svg", ".map"}


def _bundle_files() -> list[Path]:
    return sorted(p for p in CONSOLE.rglob("*") if p.is_file())


def _text_files() -> list[Path]:
    return [p for p in _bundle_files() if p.suffix in TEXT_SUFFIXES]


def test_the_bundle_is_present():
    assert (CONSOLE / "index.html").is_file()
    assert _text_files(), "no text files in console/"


@pytest.mark.parametrize("path", _bundle_files(), ids=lambda p: p.relative_to(CONSOLE).as_posix())
def test_no_file_carries_a_base64_data_uri(path):
    assert not BASE64_DATA_URI.search(path.read_bytes()), path


def test_the_scan_catches_an_inlined_font():
    # assembled at runtime so this file itself never holds a data: URI literal
    inlined = b"src:url(" + b"data:font/woff2" + b";base64," + b"AAAA" + b")"
    assert BASE64_DATA_URI.search(inlined)


def test_no_orphaned_asset():
    referenced = b"\n".join(p.read_bytes() for p in _text_files())
    orphans = [p.name for p in ASSETS.iterdir() if p.name.encode() not in referenced]
    assert orphans == []


# --- no write path in the COMMITTED JS --------------------------------------
# The bytes that ship, not a fresh in-memory build. Case-insensitive verbs;
# every browser request API that is not a plain fetch GET is named.

WRITE_VERB = re.compile(rb"""["'`](post|put|delete|patch)["'`]""", re.IGNORECASE)
NON_GET_METHOD = re.compile(rb"""method\s*:\s*(?!["'`]get["'`])""", re.IGNORECASE)
WRITE_STRINGS = (b"X-Kai-Console", b"/jobs", b"/approve", b"/reject", b"/routing")
REQUEST_APIS = re.compile(rb"XMLHttpRequest|sendBeacon|WebSocket", re.IGNORECASE)


def _write_hits(js: bytes) -> list[str]:
    hits = [m.group(0).decode() for m in WRITE_VERB.finditer(js)]
    hits += [m.group(0).decode() for m in NON_GET_METHOD.finditer(js)]
    hits += [s.decode() for s in WRITE_STRINGS if s in js]
    hits += [m.group(0).decode() for m in REQUEST_APIS.finditer(js)]
    return hits


def _committed_js() -> list[Path]:
    return sorted(ASSETS.glob("*.js"))


def test_the_committed_bundle_has_js_to_scan():
    assert _committed_js()


@pytest.mark.parametrize("path", _committed_js(), ids=lambda p: p.name)
def test_the_committed_js_carries_no_write_path(path):
    assert _write_hits(path.read_bytes()) == [], path.name


@pytest.mark.parametrize(
    "snippet",
    [
        # assembled at runtime so this file itself carries none of them
        b'fetch(u,{method:"' + b"po" + b'st"})',
        b"fetch(u,{method:'" + b"DEL" + b"ETE'})",
        b'fetch(u,{method:"' + b"PUT" + b'"})',
        b"fetch(u,{method:m})",  # a computed, non-GET-literal method
        b'h["X-Kai-' + b'Console"]="1"',
        b'fetch("/' + b'approve")',
        b"new XMLHttp" + b"Request()",
        b"navigator.send" + b"Beacon(u)",
        b"new Web" + b"Socket(u)",
    ],
)
def test_the_detector_catches_each_write_shape(snippet):
    assert _write_hits(snippet), snippet


def test_a_plain_get_fetch_is_not_a_write():
    assert _write_hits(b'fetch(u,{method:"GET",cache:"no-store"})') == []
