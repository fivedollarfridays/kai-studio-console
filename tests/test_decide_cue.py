"""Red-first: cue mode opens/closes on whole sentences, never inside a word.

The sentence boundaries are the span's ``speech_start``/``speech_end``; the
rendered ``start``/``end`` are those padded outward (``kai_slice.padding``),
never inward.
"""

from __future__ import annotations

import pytest

from kai_slice import decide
from kai_slice.clips import clip_from_doc


def _doc(sentences):
    """House doc with explicit per-sentence word timings."""
    segments = []
    t = 0.0
    for sentence in sentences:
        words = []
        seg_start = t
        for surface in sentence.split():
            words.append({"word": surface, "start": round(t, 3), "end": round(t + 0.4, 3)})
            t = round(t + 0.5, 3)
        segments.append(
            {"start": seg_start, "end": round(t - 0.1, 3), "text": sentence, "words": words}
        )
        t = round(t + 1.0, 3)
    return {"file": "x.mp4", "duration": t, "model": "fake", "segments": segments}


SENTENCES = [
    "the market opened higher today",
    "but inflation remains a worry",
    "analysts expect a rate cut soon",
]


def _clip():
    return clip_from_doc("A.mp4", "A.mp4", _doc(SENTENCES))


def _assert_on_sentence_boundaries(clip, spans):
    seg_starts = {round(s["start"], 3) for s in clip.doc["segments"]}
    seg_ends = {round(s["end"], 3) for s in clip.doc["segments"]}
    word_starts = {round(w["start"], 3) for s in clip.doc["segments"] for w in s["words"]}
    word_ends = {round(w["end"], 3) for s in clip.doc["segments"] for w in s["words"]}
    for span in spans:
        assert round(span["speech_start"], 3) in seg_starts
        assert round(span["speech_end"], 3) in seg_ends
        # a sentence boundary is also a word boundary, never mid-word
        assert round(span["speech_start"], 3) in word_starts
        assert round(span["speech_end"], 3) in word_ends
        assert span["start"] <= span["speech_start"]
        assert span["end"] >= span["speech_end"]


def test_cue_selects_matching_sentences_on_boundaries():
    clip = _clip()
    decision = decide.decide_cue([clip], cue_terms=["inflation"], target_s=30.0)
    assert decision["mode"] == "cue"
    spans = decision["spans"]
    assert len(spans) == 1
    assert "inflation" in " ".join(spans[0]["words"]).lower()
    _assert_on_sentence_boundaries(clip, spans)


def test_cue_multiple_terms_multiple_sentences():
    clip = _clip()
    decision = decide.decide_cue([clip], cue_terms=["market", "rate"], target_s=60.0)
    spans = decision["spans"]
    assert len(spans) == 2
    _assert_on_sentence_boundaries(clip, spans)


def test_no_cue_takes_densest_window_up_to_target():
    clip = _clip()
    decision = decide.decide_cue([clip], cue_terms=[], target_s=3.0)
    spans = decision["spans"]
    assert spans  # something chosen
    _assert_on_sentence_boundaries(clip, spans)
    speech = sum(s["speech_end"] - s["speech_start"] for s in spans)
    assert speech <= 3.0 + 1e-6


@pytest.mark.parametrize(
    "sentences,term,expect_words",
    [
        (["alpha beta gamma"], "beta", ["alpha", "beta", "gamma"]),
        (["one two", "three four"], "four", ["three", "four"]),
    ],
)
def test_cue_span_covers_whole_sentence(sentences, term, expect_words):
    clip = clip_from_doc("A.mp4", "A.mp4", _doc(sentences))
    spans = decide.decide_cue([clip], cue_terms=[term], target_s=60.0)["spans"]
    assert spans[0]["words"] == expect_words
