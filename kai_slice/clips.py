"""kai_slice.clips — a take plus its transcript, as the decider sees it.

A :class:`Clip` pairs an ingested media file with its own-format transcript
and the metadata the decider's tie-break needs: the clip's recorded
``creation_time`` (ffprobe, never file mtime — a verified copy rewrites mtime)
and, when the card declares one, the speaker the take belongs to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from kai_slice import transcribe
from kai_slice.textmatch import normalise

MEDIA_SUFFIXES = {".mp4", ".mov", ".mxf", ".wav", ".m4a", ".mkv"}


@dataclass(frozen=True)
class Clip:
    name: str
    path: str
    doc: dict
    words: list[dict] = field(default_factory=list)
    creation_time: str = ""
    speaker: str | None = None


def _flatten_words(doc: dict) -> list[dict]:
    """Every transcript word with its surface form, timings and normal form."""
    out: list[dict] = []
    for seg in doc.get("segments", []):
        for w in seg.get("words", []):
            out.append(
                {
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "norm": normalise(w["word"]),
                }
            )
    return [w for w in out if w["norm"]]


def clip_from_doc(
    name: str, path: str, doc: dict, *, creation_time: str = "", speaker: str | None = None
) -> Clip:
    """A :class:`Clip` from an in-memory house doc (the unit-test seam)."""
    return Clip(
        name=name,
        path=str(path),
        doc=doc,
        words=_flatten_words(doc),
        creation_time=creation_time,
        speaker=speaker,
    )


def _media_files(ingest_dir: Path) -> list[Path]:
    return sorted(
        p for p in Path(ingest_dir).iterdir() if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES
    )


def load_clips(
    ingest_dir,
    transcripts_dir,
    *,
    takes: dict[str, str] | None = None,
    creation_time_of: Callable[[Path], str] | None = None,
) -> list[Clip]:
    """Load every ingested clip paired with its ``<stem>.stt.json`` transcript.

    ``takes`` maps a clip file name to its speaker; ``creation_time_of`` reads
    a clip's recorded creation time (defaults to the ffprobe seam). Clips are
    returned sorted by file name; the decider owns the real tie-break.
    """
    resolve_creation_time: Callable[[Path], str]
    if creation_time_of is None:
        from kai_slice.probe import creation_time as resolve_creation_time  # lazy: ffprobe seam
    else:
        resolve_creation_time = creation_time_of
    takes = takes or {}
    clips: list[Clip] = []
    for media in _media_files(ingest_dir):
        doc_path = transcribe.sidecar_path(transcripts_dir, media)
        doc = transcribe.load_doc(doc_path)
        clips.append(
            clip_from_doc(
                media.name,
                str(media),
                doc,
                creation_time=resolve_creation_time(media),
                speaker=takes.get(media.name),
            )
        )
    return clips


__all__ = ["Clip", "clip_from_doc", "load_clips"]
