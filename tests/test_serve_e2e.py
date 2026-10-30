"""E2E Conformance: a real run, then `serve`, then a scripted
fetch of every endpoint the public console calls in "local run" mode
(``ingest.json`` and ``build.json``), validated against the vendored
schemas. ffmpeg-gated, like ``test_run_e2e.py``'s lavfi card.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from conftest import requires_ffmpeg

from kai_slice import console
from kai_slice.run import run_pipeline
from kai_slice.serve import make_server


@requires_ffmpeg
def test_demo_then_serve_then_console_endpoints_fetch_clean(lavfi_card, tmp_path, monkeypatch):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"

    run_pipeline(card, out, cue=[])  # the real four-stage run ("./demo" minus its CLI wrapper)
    assert (out / "console" / "ingest.json").exists()
    assert (out / "console" / "build.json").exists()

    httpd = make_server(out, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_port
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ingest.json") as r:
            assert r.status == 200
            ingest = json.loads(r.read())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/build.json") as r:
            assert r.status == 200
            build = json.loads(r.read())
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    # the exact two endpoints the public console's local-run loader calls --
    # both must validate against the vendored schemas it also ships.
    console.validate(ingest, "ingest")
    console.validate(build, "build")
    # No NAS archive exists in this pipeline, so this is honestly unknown.
    assert ingest["cards"][0]["sha256_match"] is None
    assert [s["name"] for s in build["steps"]] == ["capture", "transcribe", "decide", "act"]
    assert all(s["rc"] == 0 for s in build["steps"])
    assert build["status"] == "done"


@requires_ffmpeg
def test_serve_refuses_a_path_outside_out_dir_even_during_a_real_run(
    lavfi_card, tmp_path, monkeypatch
):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"
    run_pipeline(card, out, cue=[])
    (tmp_path / "secret.txt").write_text("private")

    httpd = make_server(out, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_port
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/../secret.txt")
            raise AssertionError("expected a 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
