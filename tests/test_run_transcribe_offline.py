"""Red-first: the CLI run path (``run.py``'s ``_transcribe``) never lets a
transcriber download a model implicitly.

On ``main`` (before this task's fix), ``_transcribe`` hands the bare
``--model`` value straight to ``transcribe.WhisperTranscriber``, which asks
faster-whisper's own ``WhisperModel`` to resolve it -- and when the name is
not already a local directory, that reaches for the network. These tests
block every socket so that gap shows up as a blocked-network AssertionError
(not a clean refusal) on ``main``, and goes green only once model
resolution refuses -- or succeeds locally -- before any socket is touched.
"""

from __future__ import annotations

import re
import socket

import pytest

from kai_slice import model as model_mod
from kai_slice import run as run_mod
from kai_slice.model import NO_MODEL_MESSAGE, ModelNotFoundError


def _block_sockets(monkeypatch) -> None:
    def _blocked(*_a, **_k):
        raise AssertionError("a socket was touched")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def test_transcribe_refuses_before_any_network_call_when_no_model_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.delenv(run_mod.REPLAY_ENV, raising=False)
    _block_sockets(monkeypatch)

    ingest = tmp_path / "ingest"
    ingest.mkdir()
    (ingest / "CLIP_0001.mp4").write_bytes(b"not-real-video")

    with pytest.raises(ModelNotFoundError, match=re.escape(NO_MODEL_MESSAGE)):
        run_mod._transcribe(
            ingest, tmp_path / "transcripts", model="small", device="cpu", transcriber=None
        )


def test_a_fetched_model_is_used_locally_with_no_network_call(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    monkeypatch.setattr(model_mod, "MODELS_ROOT", models_root)
    monkeypatch.delenv(run_mod.REPLAY_ENV, raising=False)
    _block_sockets(monkeypatch)

    fetched = models_root / "small"
    fetched.mkdir(parents=True)
    (fetched / "model.bin").write_bytes(b"fake-weights")

    calls: list[tuple[str, str]] = []

    class _FakeTranscriber:
        def __init__(self, model, device):
            calls.append((model, device))

        def __call__(self, source, start, length):
            return [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}]

    monkeypatch.setattr(run_mod.transcribe, "WhisperTranscriber", _FakeTranscriber)
    monkeypatch.setattr(run_mod.probe, "duration", lambda _media: 1.0)

    ingest = tmp_path / "ingest"
    ingest.mkdir()
    (ingest / "CLIP_0001.mp4").write_bytes(b"not-real-video")

    run_mod._transcribe(
        ingest, tmp_path / "transcripts", model="small", device="cpu", transcriber=None
    )

    assert calls == [(str(fetched), "cpu")]


def test_replay_mode_never_resolves_a_model(tmp_path, monkeypatch):
    """The REPLAY_ENV hook (used by every other test in this suite) must stay
    model-free: it must not even look under MODELS_ROOT, let alone refuse."""
    monkeypatch.setattr(model_mod, "MODELS_ROOT", tmp_path / "models-that-do-not-exist")
    replay = tmp_path / "replay"
    replay.mkdir()
    ingest = tmp_path / "ingest"
    ingest.mkdir()
    media = ingest / "CLIP_0001.mp4"
    media.write_bytes(b"not-real-video")
    (replay / "CLIP_0001.stt.json").write_text(
        '{"file": "x", "duration": 1.0, "model": "fake", "segments": []}'
    )
    monkeypatch.setenv(run_mod.REPLAY_ENV, str(replay))

    run_mod._transcribe(
        ingest, tmp_path / "transcripts", model="small", device="cpu", transcriber=None
    )


def test_an_injected_transcriber_never_resolves_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod, "MODELS_ROOT", tmp_path / "models-that-do-not-exist")
    monkeypatch.delenv(run_mod.REPLAY_ENV, raising=False)
    monkeypatch.setattr(run_mod.probe, "duration", lambda _media: 1.0)

    def fake_transcriber(source, start, length):
        return [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}]

    ingest = tmp_path / "ingest"
    ingest.mkdir()
    (ingest / "CLIP_0001.mp4").write_bytes(b"not-real-video")

    run_mod._transcribe(
        ingest, tmp_path / "transcripts", model="small", device="cpu", transcriber=fake_transcriber
    )
