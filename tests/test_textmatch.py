"""Red-first: word normalisation, in-order LCS matching, and tight spans."""

from __future__ import annotations

import pytest

from kai_slice import textmatch
from kai_slice.clips import clip_from_doc


def _doc(words):
    """A one-segment house doc from (surface, start, end) triples."""
    return {
        "file": "x.mp4",
        "duration": 10.0,
        "model": "fake",
        "segments": [
            {
                "start": words[0][1],
                "end": words[-1][2],
                "text": " ".join(w[0] for w in words),
                "words": [{"word": w[0], "start": w[1], "end": w[2]} for w in words],
            }
        ],
    }


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hello,", "hello"),
        ("DON'T", "dont"),
        ("  spaced  ", "spaced"),
        ("123", "123"),
        ("--", ""),
        ("", ""),
    ],
)
def test_normalise(raw, expected):
    assert textmatch.normalise(raw) == expected


def test_normalise_words_drops_punctuation_only_tokens():
    assert textmatch.normalise_words("Hello, -- world!") == ["hello", "world"]


@pytest.mark.parametrize(
    "line,trans,n,idx",
    [
        (["a", "b", "c"], ["a", "x", "b", "y", "c"], 3, [0, 2, 4]),
        (["a", "b"], ["q", "r"], 0, []),
        (["a", "b", "c", "d"], ["a", "b", "c", "d"], 4, [0, 1, 2, 3]),
    ],
)
def test_lcs_exact(line, trans, n, idx):
    assert textmatch._lcs_length(line, trans) == n
    assert textmatch.lcs_match_indices(line, trans) == idx


def test_lcs_match_indices_are_strictly_increasing_and_match():
    line = ["a", "b", "c"]
    trans = ["c", "b", "a", "b", "c"]
    idx = textmatch.lcs_match_indices(line, trans)
    assert len(idx) == textmatch._lcs_length(line, trans)
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    # the matched transcript tokens are an in-order subsequence of the line
    matched = [trans[i] for i in idx]
    assert matched == [w for w in line if w in matched] or len(matched) <= len(line)


def test_score_and_span_open_close_on_word_boundaries():
    # "hello world" sits in the middle of the take; the span must open on the
    # first matched word's start and close on the last matched word's end.
    doc = _doc(
        [
            ("Intro.", 0.0, 0.5),
            ("Hello,", 1.0, 1.4),
            ("world!", 1.5, 2.0),
            ("later", 5.0, 5.4),
        ]
    )
    clip = clip_from_doc("A.mp4", "A.mp4", doc)
    score, start, end, span_words = textmatch.score_line_in_clip(["hello", "world"], clip)
    assert score == pytest.approx(1.0)
    assert start == pytest.approx(1.0)  # word 'Hello,' start
    assert end == pytest.approx(2.0)  # word 'world!' end
    assert span_words == ["Hello,", "world!"]


def test_partial_score_is_lcs_over_line_length():
    doc = _doc([("one", 0.0, 0.5), ("three", 1.0, 1.5)])
    clip = clip_from_doc("A.mp4", "A.mp4", doc)
    score, *_ = textmatch.score_line_in_clip(["one", "two", "three"], clip)
    assert score == pytest.approx(2 / 3)
