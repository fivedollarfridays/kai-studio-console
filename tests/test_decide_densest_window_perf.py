"""Red-first (property test): ``decide._densest_window`` (``decide.py:167-185``)
must return results identical to the old O(n**3) brute-force search, across
randomized fixtures -- including segments with zero words, so a window's
sentence-boundary trim point is exercised, not just its total word count.
"""

from __future__ import annotations

import random
import time

from kai_slice import decide
from kai_slice.clips import clip_from_doc


def _old_densest_window(clip, target_s: float) -> dict | None:
    """The pre-optimization O(n**3) reference implementation, kept here only
    as the oracle this property test checks the real implementation against."""
    segs = clip.doc.get("segments", [])
    best = None  # (word_count, i, j)
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            if segs[j]["end"] - segs[i]["start"] > target_s:
                break
            wc = sum(len(s["words"]) for s in segs[i : j + 1])
            if best is None or wc > best[0]:
                best = (wc, i, j)
    if best is None:
        return None
    _wc, i, j = best
    words = [w["word"] for s in segs[i : j + 1] for w in s["words"]]
    span = decide._span(clip, segs[i]["start"], segs[j]["end"], words, rule="densest-window")
    span["_wc"] = _wc
    return span


def _random_doc(rng: random.Random, num_segments: int, allow_zero_words: bool) -> dict:
    segments = []
    t = 0.0
    for _ in range(num_segments):
        n_words = rng.randint(0 if allow_zero_words else 1, 4)
        words = []
        seg_start = t
        for _w in range(n_words):
            dur = rng.uniform(0.1, 0.6)
            words.append({"word": "w", "start": round(t, 3), "end": round(t + dur, 3)})
            t = round(t + dur, 3)
        seg_end = round(max(t, seg_start + rng.uniform(0.05, 0.3)), 3)
        t = seg_end
        segments.append({"start": seg_start, "end": seg_end, "words": words})
        t = round(t + rng.uniform(0.0, 0.5), 3)  # gap before next sentence
    return {"file": "x.mp4", "duration": t, "model": "fake", "segments": segments}


def test_densest_window_matches_old_implementation_on_randomized_fixtures():
    rng = random.Random(20261028)
    for case in range(300):
        num_segments = rng.randint(0, 12)
        allow_zero_words = case % 3 == 0  # exercise within-row ties sometimes
        doc = _random_doc(rng, num_segments, allow_zero_words)
        clip = clip_from_doc("A.mp4", "A.mp4", doc)
        target_s = rng.choice([0.0, 0.3, 1.0, 2.5, 5.0, 10.0, 1000.0])

        expected = _old_densest_window(clip, target_s)
        actual = decide._densest_window(clip, target_s)

        assert actual == expected, (
            f"case={case} num_segments={num_segments} target_s={target_s} "
            f"expected={expected} actual={actual}"
        )


def test_densest_window_is_fast_on_many_segments():
    """A huge ``target_s`` makes every window span the whole clip, so the old
    O(n**3) brute force (which recomputes each window's word count from
    scratch) takes several seconds at 800 one-word sentences; the O(n)
    two-pointer search finishes in well under a second. A regression back to
    the brute-force search shows up here as a slow test, not a CI timeout."""
    rng = random.Random(7)
    doc = _random_doc(rng, 800, allow_zero_words=False)
    clip = clip_from_doc("A.mp4", "A.mp4", doc)

    started = time.monotonic()
    result = decide._densest_window(clip, target_s=1e9)
    elapsed = time.monotonic() - started

    assert result is not None
    assert elapsed < 2.0, f"took {elapsed:.2f}s -- not O(n)"
