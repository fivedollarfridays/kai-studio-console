"""``kai_slice serve``: a read-only static server for
``<out>/console/``. Loopback-only bind; 404 on any path that would read
outside ``<out>/console/`` (``..``, an absolute path smuggled in, or a
symlink that resolves out).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from kai_slice.serve import (
    DEFAULT_PORT,
    NonLoopbackHostError,
    check_loopback_host,
    make_server,
    safe_console_path,
)


def test_default_port_is_a_private_port_above_1024():
    assert DEFAULT_PORT > 1024


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_accepted(host):
    check_loopback_host(host)  # must not raise


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com", ""])
def test_non_loopback_host_refuses_before_any_bind(host):
    with pytest.raises(NonLoopbackHostError):
        check_loopback_host(host)


def test_make_server_refuses_non_loopback_without_binding(tmp_path):
    (tmp_path / "console").mkdir()
    with pytest.raises(NonLoopbackHostError):
        make_server(tmp_path, host="0.0.0.0", port=0)


def _console(tmp_path, files: dict[str, bytes | str]):
    console = tmp_path / "console"
    console.mkdir()
    for name, content in files.items():
        p = console / name
        p.write_bytes(content) if isinstance(content, bytes) else p.write_text(content)
    return console


def test_safe_path_serves_a_real_file_under_console(tmp_path):
    console = _console(tmp_path, {"ingest.json": '{"ok": true}'})
    resolved = safe_console_path(console, "/ingest.json")
    assert resolved == (console / "ingest.json").resolve()


def test_safe_path_refuses_parent_traversal(tmp_path):
    console = _console(tmp_path, {"ingest.json": "{}"})
    (tmp_path / "secret.txt").write_text("private")
    assert safe_console_path(console, "/../secret.txt") is None
    assert safe_console_path(console, "/..%2fsecret.txt") is None


def test_safe_path_refuses_absolute_path_smuggled_in(tmp_path):
    console = _console(tmp_path, {"ingest.json": "{}"})
    outside = tmp_path.parent / "outside-not-under-console.txt"
    outside.write_text("private")
    # pathlib's `Path(base) / "/abs"` would DISCARD base and return the
    # absolute path verbatim -- this proves that footgun is defused.
    assert safe_console_path(console, "/" + str(outside)) is None
    assert not str(outside) in str(safe_console_path(console, "/" + str(outside)) or "")


def test_safe_path_refuses_symlink_escaping_console_dir(tmp_path):
    console = _console(tmp_path, {"ingest.json": "{}"})
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (console / "escape.json").symlink_to(outside)
    assert safe_console_path(console, "/escape.json") is None


def test_safe_path_refuses_directory_and_missing_file(tmp_path):
    console = _console(tmp_path, {"ingest.json": "{}"})
    (console / "sub").mkdir()
    assert safe_console_path(console, "/sub") is None
    assert safe_console_path(console, "/does-not-exist.json") is None
    assert safe_console_path(console, "/") is None


def test_safe_path_refuses_when_console_dir_missing(tmp_path):
    assert safe_console_path(tmp_path / "console", "/ingest.json") is None


class _ServerThread:
    def __init__(self, out_dir):
        self.httpd = make_server(out_dir, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self.httpd.server_port

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def test_server_binds_loopback_only(tmp_path):
    (tmp_path / "console").mkdir()
    server = _ServerThread(tmp_path)
    assert server.httpd.server_address[0] == "127.0.0.1"
    with server as port:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/does-not-exist.json")
        assert exc.value.code == 404


def test_server_serves_a_real_console_file_over_http(tmp_path):
    _console(tmp_path, {"ingest.json": json.dumps({"ok": True})})
    with (
        _ServerThread(tmp_path) as port,
        urllib.request.urlopen(f"http://127.0.0.1:{port}/ingest.json") as r,
    ):
        assert r.status == 200
        assert json.loads(r.read()) == {"ok": True}
        # no wildcard CORS: the console is served from this same origin
        # (see kai_slice/serve.py's module docstring)
        assert r.headers["Access-Control-Allow-Origin"] is None


@pytest.mark.parametrize("path", ["/../secret.txt", "/sub", "/does-not-exist.json", "/escape.json"])
def test_server_404s_every_unsafe_or_missing_path(tmp_path, path):
    console = _console(tmp_path, {"ingest.json": "{}"})
    (console / "sub").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (console / "escape.json").symlink_to(outside)
    (tmp_path / "secret.txt").write_text("private")
    with _ServerThread(tmp_path) as port:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}{path}")
            pytest.fail("expected HTTPError 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
