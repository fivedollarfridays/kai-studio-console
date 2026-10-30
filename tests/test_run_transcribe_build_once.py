"""Red-first: ``_transcribe`` (``run.py:44``) must build the Whisper model
ONCE for a run, not once per media file. On ``main``, ``t = transcriber or
transcribe.WhisperTranscriber(...)`` sits inside the per-media loop, so a
3-camera card loads the model 3 times.
"""

from __future__ import annotations

from kai_slice import model as model_mod
from kai_slice import run as run_mod


def test_three_media_files_construct_the_transcriber_once(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    monkeypatch.setattr(model_mod, "MODELS_ROOT", models_root)
    monkeypatch.delenv(run_mod.REPLAY_ENV, raising=False)

    fetched = models_root / "small"
    fetched.mkdir(parents=True)
    (fetched / "model.bin").write_bytes(b"fake-weights")

    calls: list[tuple[str, str]] = []

    class _CountingTranscriber:
        def __init__(self, model, device):
            calls.append((model, device))

        def __call__(self, source, start, length):
            return [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}]

    monkeypatch.setattr(run_mod.transcribe, "WhisperTranscriber", _CountingTranscriber)
    monkeypatch.setattr(run_mod.probe, "duration", lambda _media: 1.0)

    ingest = tmp_path / "ingest"
    ingest.mkdir()
    for i in range(3):
        (ingest / f"CLIP_000{i}.mp4").write_bytes(b"not-real-video")

    run_mod._transcribe(
        ingest, tmp_path / "transcripts", model="small", device="cpu", transcriber=None
    )

    assert len(calls) == 1, f"expected exactly one transcriber build, got {len(calls)}: {calls}"
    assert calls == [(str(fetched), "cpu")]
