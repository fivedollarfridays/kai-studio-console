"""``kai_slice serve`` serves the shipped console UI itself.

Chrome refuses the console's module script and stylesheet from ``file://``
(origin ``null``), so the documented replay path is
``python -m kai_slice serve`` then ``http://127.0.0.1:8920/``: the UI comes
from this tree's ``console/``, and a run's ``*.json`` from ``<out>/console/``
-- one loopback origin, so the console never needs wildcard CORS.
"""

from __future__ import annotations

import http.client
import re
import threading

import pytest

from kai_slice.cli import _build_parser
from kai_slice.serve import DEFAULT_UI_DIR, make_server


@pytest.fixture
def served():
    started = []

    def _start(out_dir=None, **kwargs):
        httpd = make_server(out_dir, host="127.0.0.1", port=0, **kwargs)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        started.append((httpd, thread))
        return httpd

    yield _start
    for httpd, thread in started:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(httpd, path):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def test_default_ui_dir_is_this_trees_console_build():
    assert (DEFAULT_UI_DIR / "index.html").is_file()


def test_replay_needs_no_out_dir_and_serves_the_shipped_index(served):
    httpd = served()
    resp, body = _get(httpd, "/")
    assert resp.status == 200
    assert body == (DEFAULT_UI_DIR / "index.html").read_bytes()


def test_every_asset_the_index_references_is_served(served):
    httpd = served()
    index = (DEFAULT_UI_DIR / "index.html").read_text()
    refs = re.findall(r'(?:src|href)="\./([^"]+)"', index)
    assert refs, "index.html references no asset"
    for ref in refs:
        resp, body = _get(httpd, "/" + ref)
        assert resp.status == 200, ref
        assert body == (DEFAULT_UI_DIR / ref).read_bytes()


def test_without_an_out_dir_run_json_is_404(served):
    httpd = served()
    assert _get(httpd, "/ingest.json")[0].status == 404


def test_run_json_comes_only_from_the_out_dir_never_the_ui(served, tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("ui")
    (ui / "build.json").write_text('{"from": "ui"}')
    out = tmp_path / "out"
    (out / "console").mkdir(parents=True)
    (out / "console" / "ingest.json").write_text('{"from": "run"}')
    (out / "console" / "index.html").write_text("run")
    httpd = served(out, ui_dir=ui)
    assert _get(httpd, "/ingest.json")[1] == b'{"from": "run"}'
    assert _get(httpd, "/build.json")[0].status == 404  # never the ui copy
    assert _get(httpd, "/")[1] == b"ui"  # never the run dir's html


def test_cli_serve_out_dir_is_optional():
    args = _build_parser().parse_args(["serve"])
    assert args.out_dir is None
    assert _build_parser().parse_args(["serve", "demo-out"]).out_dir.name == "demo-out"


def test_readme_documents_the_served_replay_path_not_file_url():
    readme = (DEFAULT_UI_DIR.parent / "README.md").read_text()
    section = readme[readme.index("## 4. See the run in the console") :]
    assert "python -m kai_slice serve\n" in section  # replay: no out dir
    assert "http://127.0.0.1:8920/" in section
    assert "open `console/index.html` in a browser" not in readme
