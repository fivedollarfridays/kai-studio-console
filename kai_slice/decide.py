"""kai_slice.decide — the deterministic cut decider (stage 3).

Two modes, both emitting a ``decision.json`` of spans (clip, start, end, the
words each covers, and the rule that chose it):

* **script** — a plain-text script of numbered, speaker-tagged lines. Each
  line is matched, in order, to its best span across the takes by in-order
  word overlap (see :mod:`kai_slice.textmatch`). The best take has the highest
  score; ties go to the LATER take by recorded ``creation_time``, then by file
  name — never by file mtime, which a verified copy rewrites. A ``takes.json``
  restricts a line to its speaker's clips. A line scoring below the threshold
  refuses by name rather than skipping. Assembled in script order, the spans
  are a conversation cut from separately recorded takes.
* **cue** — spans that open and close on whole sentences (transcript
  segments), bounded by their words, not just the segment stamps (a
  transcriber can stamp a segment after its first word starts). With cue
  terms, every sentence carrying a term; with none, the densest-speech window
  up to a target length of speech.

Every span keeps its word boundaries as ``speech_start``/``speech_end``; its
``start``/``end`` are padded outward (0.10 s head, 0.25 s tail, clamped to
the clip and its neighbouring words; see :mod:`kai_slice.padding`), so a cut
never clips a word. Two back-to-back spans of one clip whose pads meet are
split at the midpoint of the gap between their words, so no air plays twice.
The run then adds ``cut_start``/``cut_end``: those bounds
snapped to whole video frames, exactly as act renders them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kai_slice.clips import Clip
from kai_slice.padding import pad_bounds, split_seams
from kai_slice.textmatch import normalise, normalise_words, score_line_in_clip

SCRIPT_THRESHOLD = 0.70
_LINE_RE = re.compile(r"^\s*(\d+)[.)]?\s+(?:([A-Za-z0-9_]+)\s*:\s*)?(.+?)\s*$")


class DecideError(RuntimeError):
    """A refusal: a script line that no take matches above the threshold."""


@dataclass(frozen=True)
class ScriptLine:
    number: int
    speaker: str | None
    text: str
    words: list[str]


def parse_script(text: str) -> list[ScriptLine]:
    """Parse numbered, optionally speaker-tagged lines; blanks/`#` ignored."""
    lines: list[ScriptLine] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(raw)
        if not m:
            raise DecideError(f"unparseable script line: {raw!r}")
        number, speaker, body = m.group(1), m.group(2), m.group(3)
        lines.append(ScriptLine(int(number), speaker, body, normalise_words(body)))
    return lines


def _candidates(clips: list[Clip], speaker: str | None, line_number: int) -> list[Clip]:
    """Clips eligible for a line: its speaker's, when the card declares takes.

    A speaker tag that no clip carries is a refusal, never a silent fall
    back to every clip -- that would let a typo'd speaker tag get matched
    against the wrong person's take.
    """
    has_takes = any(c.speaker is not None for c in clips)
    if has_takes and speaker is not None:
        restricted = [c for c in clips if c.speaker == speaker]
        if restricted:
            return restricted
        declared = sorted({c.speaker for c in clips if c.speaker is not None})
        raise DecideError(
            f"line {line_number}: speaker {speaker!r} not found in any take; "
            f"declared speakers are {', '.join(declared)}"
        )
    return clips


def _best(line_words: list[str], clips: list[Clip]) -> tuple[Clip, float, float, float, list[str]]:
    """The highest-scoring clip for a line; ties -> later creation_time, name."""
    scored = []
    for c in clips:
        score, start, end, words = score_line_in_clip(line_words, c)
        scored.append((score, c.creation_time, c.name, c, start, end, words))
    score, _ct, _name, clip, start, end, words = max(scored, key=lambda t: (t[0], t[1], t[2]))
    return clip, score, start, end, words


def _span(clip: Clip, start: float, end: float, words: list[str], **extra) -> dict:
    """A span over ``start``-``end`` (the word boundaries, kept as
    ``speech_start``/``speech_end``), padded so it never clips a word."""
    padded_start, padded_end = pad_bounds(clip, start, end)
    return {
        "clip": clip.name,
        "start": padded_start,
        "end": padded_end,
        "speech_start": start,
        "speech_end": end,
        "words": words,
        **extra,
    }


def decide_script(clips: list[Clip], script_text: str, threshold: float = SCRIPT_THRESHOLD) -> dict:
    """Assemble spans in script order; refuse a line below ``threshold``."""
    spans = []
    for line in parse_script(script_text):
        clip, score, start, end, words = _best(
            line.words, _candidates(clips, line.speaker, line.number)
        )
        if score < threshold:
            raise DecideError(
                f"line {line.number} not found in any take, best {score:.2f} in {clip.name}"
            )
        spans.append(
            _span(
                clip,
                start,
                end,
                words,
                line=line.number,
                speaker=line.speaker,
                score=score,
                rule="best in-order match (lcs/line_len)",
            )
        )
    return {"mode": "script", "threshold": threshold, "spans": split_seams(spans)}


def script_check(clips: list[Clip], script_text: str) -> list[dict]:
    """Per-line best score, clip and matched words — renders/writes nothing.

    An unknown speaker tag never stops the check (unlike :func:`decide_script`):
    that line's row carries ``error`` naming the speaker and the declared
    speakers, with ``best_score`` 0.0, ``best_clip`` None, and no transcript
    words; every other line is still scored.
    """
    rows: list[dict] = []
    for line in parse_script(script_text):
        try:
            candidates = _candidates(clips, line.speaker, line.number)
        except DecideError as exc:
            rows.append(
                {
                    "line": line.number,
                    "speaker": line.speaker,
                    "best_score": 0.0,
                    "best_clip": None,
                    "transcript_words": [],
                    "error": str(exc),
                }
            )
            continue
        clip, score, _start, _end, words = _best(line.words, candidates)
        rows.append(
            {
                "line": line.number,
                "speaker": line.speaker,
                "best_score": round(score, 4),
                "best_clip": clip.name,
                "transcript_words": words,
            }
        )
    return rows


def _speech_bounds(seg: dict) -> tuple[float, float]:
    """A sentence's speech bounds: the WIDER of its segment stamps and its
    words' -- a transcriber can stamp a segment after its first word starts
    (or before its last word ends), and a span on the stamp would clip it."""
    words = seg.get("words", [])
    start, end = float(seg["start"]), float(seg["end"])
    if words:
        start = min(start, *(float(w["start"]) for w in words))
        end = max(end, *(float(w["end"]) for w in words))
    return start, end


def _cue_spans(clip: Clip, cue_terms: list[str]) -> list[dict]:
    """Every sentence (segment) carrying a cue term, as a span over its
    speech bounds (see :func:`_speech_bounds`)."""
    terms = {normalise(t) for t in cue_terms}
    spans = []
    for seg in clip.doc.get("segments", []):
        seg_norm = {normalise(w["word"]) for w in seg.get("words", [])}
        hit = terms & seg_norm
        if hit:
            words = [w["word"] for w in seg["words"]]
            start, end = _speech_bounds(seg)
            spans.append(_span(clip, start, end, words, rule=f"cue:{min(hit)}"))
    return spans


def _densest_window(clip: Clip, target_s: float) -> dict | None:
    """The contiguous run of whole sentences up to ``target_s`` of speech
    with the most words; its span opens and closes on the sentences' speech
    bounds (:func:`_speech_bounds`), so the pads come on top of ``target_s``.

    Two-pointer, O(n): as the left edge ``i`` advances, the furthest
    feasible right edge never moves backward, so the running word count is
    maintained incrementally instead of resummed per (i, j) pair. The one
    subtlety is a trailing run of zero-word segments at a window's right
    edge -- they do not raise the word count, so the ORIGINAL quadratic-sum
    scan would have recorded its max at the first ``j`` reaching it, not the
    furthest feasible one; ``prev_nonzero`` trims back to that same ``j`` so
    results match it exactly, tie for tie.
    """
    segs = clip.doc.get("segments", [])
    n = len(segs)
    if n == 0:
        return None
    word_counts = [len(s["words"]) for s in segs]
    bounds = [_speech_bounds(s) for s in segs]
    prev_nonzero = [-1] * n
    last_nonzero = -1
    for k in range(n):
        if word_counts[k]:
            last_nonzero = k
        prev_nonzero[k] = last_nonzero

    best = None  # (word_count, i, j)
    j = -1
    window_wc = 0
    for i in range(n):
        if j < i - 1:
            j = i - 1
            window_wc = 0
        while j + 1 < n and bounds[j + 1][1] - bounds[i][0] <= target_s:
            j += 1
            window_wc += word_counts[j]
        if j >= i:
            j_eff = max(prev_nonzero[j], i)
            if best is None or window_wc > best[0]:
                best = (window_wc, i, j_eff)
            window_wc -= word_counts[i]
    if best is None:
        return None
    _wc, i, j = best
    words = [w["word"] for s in segs[i : j + 1] for w in s["words"]]
    span = _span(clip, bounds[i][0], bounds[j][1], words, rule="densest-window")
    span["_wc"] = _wc
    return span


def decide_cue(clips: list[Clip], cue_terms: list[str], target_s: float = 60.0) -> dict:
    """Cue-term sentences across all clips, or the single densest window.

    ``target_s`` is a SPEECH budget: it bounds the window's first word start
    to last word end; the head/tail pads (up to 0.35 s) come on top of it.
    """
    ordered = sorted(clips, key=lambda c: (c.creation_time, c.name))
    spans: list[dict] = []
    if cue_terms:
        for clip in ordered:
            spans.extend(_cue_spans(clip, cue_terms))
    else:
        best = None
        for clip in ordered:
            cand = _densest_window(clip, target_s)
            if cand and (best is None or cand["_wc"] > best["_wc"]):
                best = cand
        if best is not None:
            best.pop("_wc", None)
            spans = [best]
    return {
        "mode": "cue",
        "cue_terms": list(cue_terms),
        "target_s": target_s,
        "spans": split_seams(spans),
    }


__all__ = [
    "DecideError",
    "ScriptLine",
    "decide_cue",
    "decide_script",
    "parse_script",
    "script_check",
]
