"""Red-first: ``serve`` hardening follow-ups.

* a non-GET method (POST, PUT, DELETE, ...) answers 405 with an
  ``Allow: GET`` header, rather than ``http.server``'s default 501;
* ``--help`` for the ``serve`` subcommand documents ``out_dir``, ``--host``
  and ``--port``;
* the "Serving the console at ..." banner is flushed immediately even when
  stdout is piped (not a tty, hence block-buffered by default), so a wrapper
  script sees it right away instead of it sitting in Python's text-stream
  buffer.
"""

from __future__ import annotations

import http.client
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kai_slice.cli import _build_parser
from kai_slice.serve import make_server

_PKG_ROOT = Path(__file__).resolve().parent.parent


class _Server:
    def __init__(self, out_dir, **kwargs):
        kwargs.setdefault("host", "127.0.0.1")
        self.httpd = make_server(out_dir, port=0, **kwargs)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self.httpd

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def out_dir(tmp_path):
    console = tmp_path / "out" / "console"
    console.mkdir(parents=True)
    (console / "ingest.json").write_text('{"ok": true}')
    return tmp_path / "out"


@pytest.fixture
def ui_dir(tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html><title>t</title>")
    return ui


# --- 405 + Allow: GET -------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_non_get_method_is_405_with_allow_get(out_dir, ui_dir, method):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
        conn.request(method, "/ingest.json")
        resp = conn.getresponse()
        resp.read()
        conn.close()
    assert resp.status == 405
    assert resp.getheader("Allow") == "GET"


def test_405_still_carries_the_common_security_headers(out_dir, ui_dir):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
        conn.request("POST", "/ingest.json")
        resp = conn.getresponse()
        resp.read()
        conn.close()
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    assert resp.getheader("Content-Security-Policy") is not None


def test_get_is_unaffected_by_the_405_handling(out_dir, ui_dir):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
        conn.request("GET", "/ingest.json")
        resp = conn.getresponse()
        resp.read()
        conn.close()
    assert resp.status == 200


# --- --help text -------------------------------------------------------------


def _arg_listing_line(help_text: str, token: str) -> str:
    """The "positional arguments:"/"options:" listing line for *token* --
    never the usage line, which also contains the bare token/metavar."""
    return next(
        line.strip()
        for line in help_text.splitlines()
        if line.strip().startswith(token) and not line.strip().startswith("usage:")
    )


def test_serve_help_documents_out_dir_host_and_port(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--help"])
    help_text = capsys.readouterr().out

    # not just the flag names: each needs an actual help string, not a bare
    # "  out_dir" / "  --host HOST" / "  --port PORT" with nothing after it
    assert len(_arg_listing_line(help_text, "out_dir")) > len("out_dir")
    assert len(_arg_listing_line(help_text, "--host")) > len("--host HOST")
    assert len(_arg_listing_line(help_text, "--port")) > len("--port PORT")


def test_serve_help_cli_subprocess_documents_out_dir_host_and_port():
    result = subprocess.run(
        [sys.executable, "-m", "kai_slice", "serve", "--help"],
        cwd=_PKG_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0
    assert len(_arg_listing_line(result.stdout, "out_dir")) > len("out_dir")
    assert len(_arg_listing_line(result.stdout, "--host")) > len("--host HOST")
    assert len(_arg_listing_line(result.stdout, "--port")) > len("--port PORT")


# --- banner flush when stdout is piped --------------------------------------


def test_serve_banner_is_flushed_promptly_when_stdout_is_a_pipe(out_dir):
    """``subprocess.PIPE`` is never a tty, so the child's stdout is
    block-buffered by default: without an explicit flush, the one-line
    banner sits in the child's own text-stream buffer and a wrapper script
    reading line-by-line would see nothing until the buffer filled -- which
    never happens here, since ``serve_forever`` runs indefinitely after it."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "kai_slice", "serve", str(out_dir), "--port", "0"],
        cwd=_PKG_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True)
    reader.start()
    try:
        line = lines.get(timeout=5)
    except queue.Empty:
        line = None
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    assert line is not None, "banner was not flushed to the pipe within 5s"
    assert "Serving the console" in line
