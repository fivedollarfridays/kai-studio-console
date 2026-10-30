"""kai_slice.padding — air around a decided span, so a cut never clips a word.

A transcriber's word ``end`` marks where the word is recognised, not where
its sound stops: the decay runs on past it. A span that ends on that
timestamp chops the word. So every span is padded:

* **tail** — ``end`` moves :data:`TAIL_PAD_S` (0.25 s) past the last word's
  end, but never past the clip's duration nor into the next word's start;
* **head** — ``start`` moves :data:`HEAD_PAD_S` (0.10 s) before the first
  word's start, but never before 0 nor into the previous word's end.

Fixed pads, not a loudness search: the decider stays a pure function of the
transcript. The unpadded word boundaries are kept on the span as
``speech_start`` / ``speech_end``. Padded bounds are rounded OUTWARD to the
millisecond (start floors, end ceils) -- a nearest-rounding could land a
bound back inside the word -- and the end is still never past the clip's
duration.

The pads stop at the neighbouring WORDS, so two spans cut back to back from
the same clip can still overlap: one line's tail and the next line's head
both reach into the gap between them, and the cut would play that air twice.
:func:`split_seams` splits such a gap at its midpoint, between the earlier
span's ``speech_end`` and the later one's ``speech_start``: the earlier tail
ends and the later head starts on it, so neither pad crosses it and the two
spans play the source continuously.
"""

from __future__ import annotations

import itertools
import math

from kai_slice.clips import Clip

TAIL_PAD_S = 0.25
HEAD_PAD_S = 0.10
_MS = 1000
_NOISE = 6  # decimal places of float noise discarded before an outward round


def _floor_ms(t: float) -> float:
    return round(math.floor(round(t * _MS, _NOISE)) / _MS, 3)


def _ceil_ms(t: float) -> float:
    return round(math.ceil(round(t * _MS, _NOISE)) / _MS, 3)


def _neighbours(clip: Clip, start: float, end: float) -> tuple[float, float | None]:
    """The neighbouring words, by their WORD bounds: the latest end of any
    word that starts before ``start`` (0.0 if none), and the earliest start
    of any word that ends after ``end`` (``None`` if none). A word that
    straddles a span bound is a neighbour, so a pad never runs into it."""
    prev_end = 0.0
    next_start: float | None = None
    for seg in clip.doc.get("segments", []):
        for w in seg.get("words", []):
            w_start, w_end = float(w["start"]), float(w["end"])
            if w_start < start:
                prev_end = max(prev_end, w_end)
            if w_end > end and (next_start is None or w_start < next_start):
                next_start = w_start
    return prev_end, next_start


def pad_bounds(clip: Clip, start: float, end: float) -> tuple[float, float]:
    """``(start, end)`` widened by the head/tail pads, clamped as above.
    A clip with no recorded ``duration`` gets no tail past ``end``."""
    prev_end, next_start = _neighbours(clip, start, end)
    duration = clip.doc.get("duration")
    limit = float(duration) if duration is not None else end
    if next_start is not None:
        limit = min(limit, next_start)
    padded_start = max(_floor_ms(start - HEAD_PAD_S), prev_end, 0.0)
    padded_end = max(min(_ceil_ms(end + TAIL_PAD_S), limit), end)
    return min(padded_start, start), padded_end


def _split_pair(a: dict, b: dict) -> None:
    """Meet two same-clip spans at their word gap's midpoint, in place.
    Spans whose words themselves overlap are left alone: there is no gap to
    split without cutting a word."""
    early, late = (a, b) if a["speech_start"] <= b["speech_start"] else (b, a)
    if early["end"] < late["start"] or early["speech_end"] > late["speech_start"]:
        return
    mid = round((early["speech_end"] + late["speech_start"]) / 2, 6)
    early["end"] = mid
    late["start"] = mid


def split_seams(spans: list[dict]) -> list[dict]:
    """``spans`` (copied) with every consecutive same-clip pair whose padded
    bounds overlap or touch split at the midpoint of the word gap."""
    out = [dict(s) for s in spans]
    for a, b in itertools.pairwise(out):
        if a["clip"] == b["clip"]:
            _split_pair(a, b)
    return out


__all__ = ["HEAD_PAD_S", "TAIL_PAD_S", "pad_bounds", "split_seams"]
