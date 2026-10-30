"""``kai_slice serve`` hardening: an absolute per-REQUEST
deadline (a client dripping one byte at a time resets the per-operation
socket timeout on every read, so it alone never trips), a generic ``Server``
header that does not name the Python version, and ``Content-Security-Policy``
/ ``Referrer-Policy`` on every response.
"""

from __future__ import annotations

import http.client
import socket
import threading
import time

import pytest

from kai_slice.serve import make_server


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


def _get(httpd, path: str) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp


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


# --- per-request deadline --------------------------------------------------


def test_a_dripping_client_is_dropped_within_the_request_deadline(out_dir, ui_dir):
    """One byte every 0.3 s never idles longer than the (generous) per-op
    socket timeout, so only an absolute per-request deadline catches it."""
    with _Server(out_dir, ui_dir=ui_dir, request_timeout=30, request_deadline=1.0) as httpd:
        drip = socket.create_connection(("127.0.0.1", httpd.server_port), timeout=5)
        started = time.monotonic()
        try:
            request_line = b"GET / HTTP/1.1\r\n"
            for byte in request_line[:-2]:  # never send the closing CRLF
                try:
                    drip.send(bytes([byte]))
                except OSError:
                    break
                time.sleep(0.3)
                if time.monotonic() - started > 5:
                    break
            assert drip.recv(1) == b""  # the server closed it
        finally:
            drip.close()
        elapsed = time.monotonic() - started
        assert elapsed < 5  # dropped near the 1 s deadline, not held 20 s+


def test_a_fast_request_completes_well_within_a_short_deadline(out_dir, ui_dir):
    with _Server(out_dir, ui_dir=ui_dir, request_deadline=5.0) as httpd:
        assert _get(httpd, "/ingest.json").status == 200


# --- Server header does not leak the Python version -----------------------


@pytest.mark.parametrize("path", ["/ingest.json", "/missing.json"])
def test_server_header_does_not_name_python(out_dir, ui_dir, path):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        resp = _get(httpd, path)
    server = resp.getheader("Server")
    assert server is not None
    assert "Python" not in server
    assert "." not in server  # no version number of any kind


# --- CSP + Referrer-Policy --------------------------------------------------
#
# ``style-src-attr 'unsafe-inline'`` is the one loosening beyond 'self',
# needed because the console bundle's shared DOM helper
# (apps/console/src/ui/dom.ts:17) writes dynamic inline styling via
# ``el.setAttribute("style", ...)``. Confirmed as the SOLE CSP violator --
# one call site, 65 hits, nothing else -- in a headless-Chromium load
# against a real run's ``serve``. Everything else (scripts, fetches, fonts,
# <style> blocks) stays on 'self'; this is pinned to the exact string so a
# future change can't widen it silently.


@pytest.mark.parametrize("path", ["/ingest.json", "/", "/missing.json"])
def test_csp_and_referrer_policy_are_pinned_on_every_response(out_dir, ui_dir, path):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        resp = _get(httpd, path)
    assert resp.getheader("Content-Security-Policy") == (
        "default-src 'self'; style-src-attr 'unsafe-inline'"
    )
    assert resp.getheader("Referrer-Policy") == "no-referrer"
