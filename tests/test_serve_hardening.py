"""Security audit (``::1``) -- ``kai_slice serve``
hardening: a NUL in the path is a 404, never an uncaught error; every
response carries ``nosniff`` and a Content-Type that matches the file; no
wildcard CORS (the console is served from this same origin); a silent client
never blocks a second request; ``::1`` really binds IPv6.
"""

from __future__ import annotations

import http.client
import socket
import threading
import time

import pytest

from kai_slice.serve import make_server, safe_console_path


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


def _get(httpd, path: str, host: str = "127.0.0.1") -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection(host, httpd.server_port, timeout=5)
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
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>t</title>")
    for name in ("a.css", "a.js", "a.woff", "a.woff2", "a.unknownext"):
        (ui / "assets" / name).write_bytes(b"x")
    return ui


# --- a NUL byte ------------------------------------------------------------


@pytest.mark.parametrize("raw", ["/%00", "/ingest.json%00", "/%00/ingest.json", "/\x00"])
def test_safe_path_with_a_nul_byte_is_none_not_an_error(out_dir, raw):
    assert safe_console_path(out_dir / "console", raw) is None


def test_server_404s_a_nul_byte_path(out_dir, ui_dir):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        assert _get(httpd, "/%00").status == 404
        assert _get(httpd, "/ingest.json%00").status == 404
        assert _get(httpd, "/ingest.json").status == 200  # still serving after


# --- headers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "ctype"),
    [
        ("/ingest.json", "application/json"),
        ("/", "text/html; charset=utf-8"),
        ("/index.html", "text/html; charset=utf-8"),
        ("/assets/a.css", "text/css; charset=utf-8"),
        ("/assets/a.js", "text/javascript; charset=utf-8"),
        ("/assets/a.woff", "font/woff"),
        ("/assets/a.woff2", "font/woff2"),
        ("/assets/a.unknownext", "application/octet-stream"),
    ],
)
def test_content_type_matches_the_extension(out_dir, ui_dir, path, ctype):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        resp = _get(httpd, path)
    assert resp.status == 200
    assert resp.getheader("Content-Type") == ctype


@pytest.mark.parametrize("path", ["/ingest.json", "/", "/missing.json", "/%00"])
def test_every_response_is_nosniff_and_carries_no_wildcard_cors(out_dir, ui_dir, path):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        resp = _get(httpd, path)
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    assert resp.getheader("Access-Control-Allow-Origin") is None


# --- a slow client ---------------------------------------------------------


def test_a_silent_connection_does_not_block_a_second_request(out_dir, ui_dir):
    with _Server(out_dir, ui_dir=ui_dir, request_timeout=30) as httpd:
        silent = socket.create_connection(("127.0.0.1", httpd.server_port), timeout=5)
        try:
            started = time.monotonic()
            assert _get(httpd, "/ingest.json").status == 200
            assert time.monotonic() - started < 3
        finally:
            silent.close()


def test_a_silent_connection_is_dropped_after_the_request_timeout(out_dir, ui_dir):
    with _Server(out_dir, ui_dir=ui_dir, request_timeout=0.3) as httpd:
        silent = socket.create_connection(("127.0.0.1", httpd.server_port), timeout=5)
        try:
            assert silent.recv(1) == b""  # server closed it; recv did not hit our 5 s
        finally:
            silent.close()


# --- ::1 ---------------------------------------------------------------------


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="no IPv6 loopback on this host")
def test_ipv6_loopback_binds_ipv6_and_serves(out_dir, ui_dir):
    with _Server(out_dir, ui_dir=ui_dir, host="::1") as httpd:
        assert httpd.address_family == socket.AF_INET6
        assert _get(httpd, "/ingest.json", host="::1").status == 200


# --- DNS rebinding: only a loopback Host header is answered ----------------


def _get_with_host(httpd, path: str, host_header: str | None) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
    conn.putrequest("GET", path, skip_host=True)
    if host_header is not None:
        conn.putheader("Host", host_header)
    conn.endheaders()
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost", "LocalHost", "[::1]"])
def test_a_loopback_host_header_with_the_bound_port_is_served(out_dir, ui_dir, name):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        resp = _get_with_host(httpd, "/ingest.json", f"{name}:{httpd.server_port}")
    assert resp.status == 200


@pytest.mark.parametrize(
    "header",
    [
        "evil.example:{port}",  # a rebound DNS name pointing at 127.0.0.1
        "evil.example",
        "127.0.0.1:{other}",  # right name, wrong port
        "127.0.0.1",  # no port while bound off :80
        "127.0.0.1.evil.example:{port}",
        "",
        None,  # no Host header at all
    ],
)
def test_any_other_host_header_is_refused_4xx(out_dir, ui_dir, header):
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        port = httpd.server_port
        value = None if header is None else header.format(port=port, other=port + 1)
        resp = _get_with_host(httpd, "/ingest.json", value)
    assert 400 <= resp.status < 500
    assert resp.getheader("X-Content-Type-Options") == "nosniff"


# --- .json vs UI is decided on the DECODED path -----------------------------


def test_an_encoded_json_path_is_still_routed_to_the_run_dir_only(out_dir, ui_dir):
    (ui_dir / "build.json").write_text('{"from": "ui"}')
    with _Server(out_dir, ui_dir=ui_dir) as httpd:
        assert _get(httpd, "/ingest%2Ejson").status == 200  # the run's file
        assert _get(httpd, "/build%2ejson").status == 404  # never the ui copy
        assert _get(httpd, "/%2F").status == 404  # an encoded "/" is not the index
