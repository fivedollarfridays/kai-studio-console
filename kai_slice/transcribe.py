"""kai_slice.transcribe — local word-level STT (stage 2).

Re-homed from the source project's STT producer (only imports and
docstrings changed, and the model name parameterised rather than hard-coded).
Every doc names its own ``file``, ``model`` and ``duration`` and carries
word-level timings in ONE global clock, so a downstream cut pass reads the
clock instead of measuring it.

``build_doc``/``merge_docs`` are pure so a long take can be transcribed in
windows without the chunk clocks silently restarting at zero.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The laptop default: faster-whisper ``small`` on CPU ``int8`` (owner's
#: "on a laptop" constraint). Every entry point takes ``model`` as a
#: parameter; this is only the default, never a hard-coded transcriber.
DEFAULT_MODEL = "small"
STT_SUFFIX = ".stt.json"


def _shift_word(word: dict, offset_s: float) -> dict:
    return {
        "word": word["word"],
        "start": word["start"] + offset_s,
        "end": word["end"] + offset_s,
        "probability": word.get("probability"),
    }


def _shift_segment(segment: dict, offset_s: float) -> dict:
    return {
        "start": segment["start"] + offset_s,
        "end": segment["end"] + offset_s,
        "text": segment["text"],
        "words": [_shift_word(w, offset_s) for w in segment.get("words") or []],
    }


def build_doc(
    segments, *, source: str, duration: float, offset_s: float = 0.0, model: str = DEFAULT_MODEL
) -> dict:
    """The house sidecar shape, with provenance and a GLOBAL clock.

    ``offset_s`` is the window's start in the whole take; every timestamp is
    shifted by it so a chunked run reads back as one continuous transcript.
    """
    return {
        "file": source,
        "duration": float(duration),
        "model": model,
        "segments": [_shift_segment(s, offset_s) for s in segments],
    }


def merge_docs(docs: list[dict]) -> dict:
    """Concatenate windowed docs of ONE source into a single transcript."""
    if not docs:
        raise ValueError("merge_docs needs at least one doc")
    sources = {d["file"] for d in docs}
    if len(sources) != 1:
        raise ValueError(f"merge_docs needs docs of the same source, got {sorted(sources)}")
    merged = dict(docs[0])
    segments: list[dict] = []
    for doc in docs:
        segments.extend(doc["segments"])
    merged["segments"] = sorted(segments, key=lambda s: s["start"])
    return merged


def sidecar_path(out_dir: Path, source: Path) -> Path:
    """``<out_dir>/<stem>.stt.json`` for a take."""
    return Path(out_dir) / f"{Path(source).stem}{STT_SUFFIX}"


def save_doc(doc: dict, path: Path) -> Path:
    """Write the sidecar with a stable, diffable encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    return path


def load_doc(path: Path) -> dict:
    """Read a sidecar written by :func:`write_doc` back."""
    return json.loads(Path(path).read_text())


def plan_windows(duration: float, window_s: float) -> list[tuple[float, float]]:
    """Split a take into (start, length) windows covering it exactly once."""
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        windows.append((start, min(window_s, duration - start)))
        start += window_s
    return windows or [(0.0, float(duration))]


def transcribe_take(
    source, *, duration: float, transcriber, window_s: float = 600.0, model: str = DEFAULT_MODEL
) -> dict:
    """Transcribe a take window by window and merge into ONE global clock.

    ``transcriber(source, start, length)`` returns faster-whisper-shaped
    segments in the WINDOW's local clock; the offset is re-applied here so no
    caller has to remember to do it.
    """
    docs = [
        build_doc(
            transcriber(source, start, length),
            source=str(source),
            duration=duration,
            offset_s=start,
            model=model,
        )
        for start, length in plan_windows(duration, window_s)
    ]
    return merge_docs(docs)


class WhisperTranscriber:
    """faster-whisper with word timestamps, model and device parameterised.

    Heavy deps are imported lazily so this module stays importable without a
    model present. The audio is decoded ONCE per source and sliced per window.
    """

    SAMPLE_RATE = 16000

    def __init__(
        self, model: str = DEFAULT_MODEL, device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        from faster_whisper import WhisperModel

        self.model = model
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._audio_key: str | None = None
        self._audio = None

    def _decoded(self, source):
        from faster_whisper.audio import decode_audio

        if self._audio_key != str(source):
            self._audio = decode_audio(str(source), sampling_rate=self.SAMPLE_RATE)
            self._audio_key = str(source)
        return self._audio

    def __call__(self, source, start: float, length: float) -> list[dict]:
        audio = self._decoded(source)
        lo = round(start * self.SAMPLE_RATE)
        hi = round((start + length) * self.SAMPLE_RATE)
        segments, _info = self._model.transcribe(audio[lo:hi], language="en", word_timestamps=True)
        return [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (s.words or [])
                ],
            }
            for s in segments
        ]


__all__ = [
    "DEFAULT_MODEL",
    "WhisperTranscriber",
    "build_doc",
    "load_doc",
    "merge_docs",
    "plan_windows",
    "save_doc",
    "sidecar_path",
    "transcribe_take",
]
