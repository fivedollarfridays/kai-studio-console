"""Red-first: verified capture, and a corrupt destination byte refusing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import kai_slice.capture as capmod
from kai_slice.capture import CardOffloadError, capture_stage
from kai_slice.run import run_pipeline


def test_capture_manifest_hashes_match_the_copied_files(tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    (card / "a.mp4").write_bytes(b"hello bytes")
    (card / "b.mp4").write_bytes(b"world!! more")
    out = tmp_path / "out"
    capture_stage(card, out)
    manifest = json.loads((out / "ingest" / "manifest.json").read_text())
    assert manifest["count"] == 2
    for rec in manifest["files"]:
        data = (out / "ingest" / rec["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == rec["sha256"]
        assert rec["bytes"] == len(data)


def test_corrupt_destination_byte_refuses_by_name_and_writes_no_later_stage(tmp_path, monkeypatch):
    card = tmp_path / "card"
    card.mkdir()
    (card / "clip.mp4").write_bytes(b"good original bytes")
    out = tmp_path / "out"

    def _corrupting_copy(src, dst):
        Path(dst).write_bytes(b"corrupted on the way in")

    monkeypatch.setattr(capmod, "copy2", _corrupting_copy)
    with pytest.raises(CardOffloadError) as exc:
        # an injected transcriber: this pins capture's refusal, not model lookup
        run_pipeline(card, out, cue=[], transcriber=lambda *_a: [])
    assert "clip.mp4" in str(exc.value)

    assert not (out / "decision.json").exists()
    assert not (out / "cut.mp4").exists()
    rows = [json.loads(line) for line in (out / "run-record.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["stage"] == "capture"
    assert rows[0]["rc"] == 1
    assert "clip.mp4" in rows[0]["error"]
