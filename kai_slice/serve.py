"""kai_slice.serve — a tiny read-only static server for the console.

``python -m kai_slice serve [<out_dir>]`` serves two things on one loopback
origin: the shipped console UI (this tree's ``console/`` build) and, when
*out_dir* is given, the run's own ``*.json`` from ``<out_dir>/console/``, so
the console's "local run" mode shows a judge's own capture, decisions and
timings without a model, a database, or any server but this one.

It binds ``127.0.0.1`` (also accepts ``localhost`` or ``::1``, the latter on
a real IPv6 socket) ONLY, refusing any other ``--host`` before ever opening a
socket, and answers only a request whose ``Host`` header is one of those
loopback names with the bound port (421 otherwise: DNS rebinding). A request
path resolves only to a file inside the directory it is routed to: ``..``, an absolute path, a NUL byte, or a symlink that resolves
outside gets a 404 -- never a directory listing and never a read outside.

No ``Access-Control-Allow-Origin`` header: the console page is served from
this same origin, so it needs no CORS, and leaving it off means no other web
page open in the browser can read a run's JSON off this port. (``file://``
cannot host the console at all -- Chrome blocks its module script there.)
Every response is ``nosniff`` with a Content-Type from the file's extension.
Each connection gets its own thread and a socket timeout, so a client that
connects and goes silent never blocks anyone else.
"""

from __future__ import annotations

import functools
import http.server
import socket
import threading
from pathlib import Path
from urllib.parse import unquote

DEFAULT_PORT = 8920
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_UI_DIR = Path(__file__).resolve().parent.parent / "console"
REQUEST_TIMEOUT = 10.0  # seconds a connection may sit silent before it is dropped
REQUEST_DEADLINE = 10.0  # seconds a request has, total, no matter how it drips
# `style-src-attr 'unsafe-inline'` is required, and only it: the console
# bundle builds its DOM through one shared helper that applies each element's
# inline styling with a single `setAttribute("style", ...)` call site. A
# headless-browser load of the console against a real `serve` shows that call
# as the only thing the stricter `default-src 'self'` would block. Nothing
# else needs to leave 'self': no inline <style> blocks, no eval, no
# cross-origin fetch/img/font.
CONTENT_SECURITY_POLICY = "default-src 'self'; style-src-attr 'unsafe-inline'"
CONTENT_TYPES = {
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
}


class NonLoopbackHostError(ValueError):
    """``--host`` is not a loopback name/address this server will bind."""


def check_loopback_host(host: str) -> None:
    if host not in LOOPBACK_HOSTS:
        raise NonLoopbackHostError(
            f"refusing to bind {host!r}: kai_slice serve only ever listens on "
            "127.0.0.1 (also accepts 'localhost' or '::1')"
        )


def safe_console_path(console_dir: Path, raw_path: str) -> Path | None:
    """The real file under *console_dir* that URL path *raw_path* names, or
    None when it is unsafe or does not exist.

    Every leading ``/`` is stripped before the path is ever joined to
    *console_dir* -- ``Path(base) / "/etc/passwd"`` would otherwise discard
    *base* entirely, since pathlib treats joining an absolute path as a
    replacement, not a traversal. A ``..`` or ``.`` segment or a NUL byte
    refuses outright (never resolved away), and the final resolved path --
    following any symlink -- must still sit inside *console_dir*.
    """
    if not console_dir.is_dir():
        return None
    path_only = raw_path.split("?", 1)[0].split("#", 1)[0]
    rel = unquote(path_only).lstrip("/")
    if not rel or "\x00" in rel:
        return None
    parts = rel.split("/")
    if any(part in ("", "..", ".") for part in parts):
        return None
    try:
        console_real = console_dir.resolve(strict=True)
        resolved = (console_dir / Path(*parts)).resolve(strict=True)
    except (OSError, ValueError):
        return None
    try:
        resolved.relative_to(console_real)
    except ValueError:
        return None
    return resolved if resolved.is_file() else None


def allowed_host(header: str | None, port: int) -> bool:
    """True only for a loopback ``Host`` naming the bound *port*. A DNS
    rebinding page reaches 127.0.0.1 under its OWN name (``evil.example``),
    so refusing every other Host keeps it from reading a run off this port.
    A bare name is accepted only on port 80, where browsers omit the port."""
    if not header:
        return False
    names = ("127.0.0.1", "localhost", "[::1]")
    allowed = {f"{name}:{port}" for name in names}
    if port == 80:
        allowed.update(names)
    return header.strip().lower() in allowed


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


class _ConsoleHandler(http.server.BaseHTTPRequestHandler):
    # overridden per-server in make_server
    run_dir: Path | None = None
    ui_dir: Path = DEFAULT_UI_DIR
    timeout = REQUEST_TIMEOUT  # StreamRequestHandler sets it on the socket
    request_deadline = REQUEST_DEADLINE
    # BaseHTTPRequestHandler types `server` as the base `socketserver.BaseServer`;
    # `make_server` only ever binds this handler to an `http.server.HTTPServer`
    # subclass, which actually has `server_port`.
    server: http.server.HTTPServer

    def setup(self) -> None:
        super().setup()
        # A per-op socket timeout resets on every successful read, so a
        # client that drips one byte at a time never idles long enough to
        # trip it. This timer is an absolute wall-clock cap on the whole
        # request, armed independently of how the socket is behaving.
        self._deadline_timer = threading.Timer(self.request_deadline, self._past_deadline)
        self._deadline_timer.daemon = True
        self._deadline_timer.start()

    def _past_deadline(self) -> None:
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def finish(self) -> None:
        self._deadline_timer.cancel()
        super().finish()

    def _route(self) -> Path | None:
        """A run's ``*.json`` only from the run dir; everything else only
        from the UI build -- neither can shadow the other. The split is
        decided on the DECODED path (``/ingest%2Ejson`` is a ``.json``), the
        same decoding ``safe_console_path`` then applies, once, to the raw
        path."""
        decoded = unquote(self.path.split("?", 1)[0].split("#", 1)[0])
        if decoded == "/":
            return safe_console_path(self.ui_dir, "/index.html")
        if decoded.lower().endswith(".json"):
            return None if self.run_dir is None else safe_console_path(self.run_dir, self.path)
        return safe_console_path(self.ui_dir, self.path)

    def do_GET(self) -> None:
        if not allowed_host(self.headers.get("Host"), self.server.server_port):
            self._empty(421)  # Misdirected Request: not a loopback Host
            return
        resolved = self._route()
        if resolved is None:
            self._empty(404)
            return
        data = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type(resolved))
        self.send_header("Content-Length", str(len(data)))
        self._common_headers()
        self.end_headers()
        self.wfile.write(data)

    def _empty(self, code: int, *, allow: str | None = None) -> None:
        self.send_response(code)
        if allow is not None:
            self.send_header("Allow", allow)
        self._common_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # Only GET is served; every other verb is 405 with Allow: GET, not
    # http.server's default 501 (which would otherwise fire for any verb
    # with no matching do_* method). partialmethod, not a new do_* def per
    # verb, keeps this one rule in one place.
    _method_not_allowed = functools.partialmethod(_empty, 405, allow="GET")
    do_HEAD = _method_not_allowed
    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_OPTIONS = _method_not_allowed

    def version_string(self) -> str:
        return "kai-slice"  # no Python/OS version for a fingerprinting probe to read

    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.send_header("Referrer-Policy", "no-referrer")

    def log_message(self, *args: object) -> None:  # pragma: no cover -- quiet
        pass


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


class _Server6(_Server):
    address_family = socket.AF_INET6


def make_server(
    out_dir: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    ui_dir: Path | None = None,
    request_timeout: float = REQUEST_TIMEOUT,
    request_deadline: float = REQUEST_DEADLINE,
) -> http.server.ThreadingHTTPServer:
    """Build the server. Refuses (raising, binding nothing) before any
    socket opens when *host* is not a loopback name/address."""
    check_loopback_host(host)
    attrs = {
        "run_dir": None if out_dir is None else Path(out_dir) / "console",
        "ui_dir": DEFAULT_UI_DIR if ui_dir is None else Path(ui_dir),
        "timeout": request_timeout,
        "request_deadline": request_deadline,
    }
    handler = type("BoundConsoleHandler", (_ConsoleHandler,), attrs)
    server_cls = _Server6 if host == "::1" else _Server
    return server_cls((host, port), handler)


def serve(
    out_dir: Path | None = None, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT
) -> None:
    httpd = make_server(out_dir, host=host, port=port)
    shown = f"[{host}]" if ":" in host else host
    run = "replay only" if out_dir is None else f"run data from {Path(out_dir) / 'console'}/"
    # flush=True: stdout is block-buffered (not line-buffered) whenever it is
    # piped rather than a tty, so a wrapper script reading this line would
    # otherwise not see it until the buffer filled or the process exited.
    print(
        f"Serving the console at http://{shown}:{httpd.server_port}/ ({run}; read-only, GET only)",
        flush=True,
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_UI_DIR",
    "LOOPBACK_HOSTS",
    "NonLoopbackHostError",
    "allowed_host",
    "check_loopback_host",
    "content_type",
    "make_server",
    "safe_console_path",
    "serve",
]
