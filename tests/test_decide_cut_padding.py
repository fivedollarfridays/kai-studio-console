"""Red-first: a decided span never clips a word. Its ``end`` lands 0.25 s
past the last word's end (``speech_end``) and its ``start`` 0.10 s before
the first word's start (``speech_start``). Each pad is clamped to the clip's
duration and to the neighbouring words, so a pad never runs into another
line.

The fixture is a copy of the real ``./demo`` transcript of the shipped sample
(``samples/card-01/CLIP_0001.mp4``, faster-whisper ``small``). Only its
``file``/``model`` paths are neutralised. Every time below is from that
transcript. Its last word, "how.", ends at 31.80 s. Before this fix the demo
cut stopped there, while the word's decay runs to about 31.95 s.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kai_slice import decide
from kai_slice.clips import clip_from_doc

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "card-01-CLIP_0001.stt.json"


def _real_doc() -> dict:
    return json.loads(_FIXTURE.read_text())


def _clip(doc: dict | None = None):
    return clip_from_doc("CLIP_0001.mp4", "CLIP_0001.mp4", doc or _real_doc())


def test_demo_cut_on_the_real_sample_ends_a_quarter_second_after_how():
    span = decide.decide_cue([_clip()], cue_terms=[], target_s=60.0)["spans"][0]
    assert span["words"][-1] == " how."
    assert span["speech_end"] == pytest.approx(31.80)
    assert span["end"] == pytest.approx(32.05)
    # the first word starts at 0.00 s: the head pre-roll clamps to the clip start
    assert span["speech_start"] == pytest.approx(0.0)
    assert span["start"] == pytest.approx(0.0)


def test_tail_pad_clamps_to_the_clip_duration():
    doc = _real_doc()
    doc["duration"] = 31.90  # the same take, trimmed 0.10 s after "how."
    span = decide.decide_cue([_clip(doc)], cue_terms=[], target_s=60.0)["spans"][0]
    assert span["end"] == pytest.approx(31.90)


def test_head_pre_roll_backs_off_a_tenth_and_tail_takes_the_full_quarter():
    # "Social media marketing made easy." spans 4.54-6.20; "Studio." ends 3.62,
    # the next sentence starts 7.02 -- both pads fit in the gaps.
    span = decide.decide_cue([_clip()], cue_terms=["marketing"], target_s=60.0)["spans"][0]
    assert (span["speech_start"], span["speech_end"]) == pytest.approx((4.54, 6.20))
    assert span["start"] == pytest.approx(4.44)
    assert span["end"] == pytest.approx(6.45)


def test_pads_stop_on_the_neighbouring_words_never_overlap_them():
    # Script line "I'm Kevin, and this" runs 1.26-2.60. "Hey," ends 0.92, so
    # the head takes its full 0.10 s. The next word "is" starts at 2.60, the
    # same instant "this" ends, so the tail can take no air at all.
    span = decide.decide_script([_clip()], "1. I'm Kevin, and this")["spans"][0]
    assert (span["speech_start"], span["speech_end"]) == pytest.approx((1.26, 2.60))
    assert span["start"] == pytest.approx(1.16)
    assert span["end"] == pytest.approx(2.60)


def test_tail_takes_only_the_air_before_the_next_word():
    # Script line "Hey, I'm Kevin," ends 1.68; "and" starts 1.96, so the
    # 0.25 s tail fits in the 0.28 s gap.
    span = decide.decide_script([_clip()], "1. Hey, I'm Kevin,")["spans"][0]
    assert span["speech_end"] == pytest.approx(1.68)
    assert span["end"] == pytest.approx(1.93)


def test_cue_span_opens_on_the_first_words_start_not_the_later_segment_start():
    # "But not until I say yes." -- the transcriber stamps the segment at
    # 16.08 s, but its first word "But" starts at 15.82 s. A span opened on
    # the segment start would cut 0.26 s into "But".
    span = decide.decide_cue([_clip()], cue_terms=["i"], target_s=60.0)["spans"]
    but = next(s for s in span if s["words"][0] == " But")
    assert but["speech_start"] == pytest.approx(15.82)
    assert but["start"] == pytest.approx(15.72)  # the full 0.10 s head; "post." ends 14.30
    assert but["speech_end"] == pytest.approx(17.62)


def test_every_cue_span_covers_all_of_its_words():
    """The docstring's claim, checked on every sentence of the real take:
    a span's rendered bounds contain each of its words, start to end."""
    clip = _clip()
    spans = decide.decide_cue([clip], cue_terms=["i", "the", "it", "so", "all"], target_s=60.0)
    for span in spans["spans"]:
        seg = next(
            s for s in clip.doc["segments"] if [w["word"] for w in s["words"]] == span["words"]
        )
        assert span["start"] <= min(w["start"] for w in seg["words"])
        assert span["end"] >= max(w["end"] for w in seg["words"])


def test_densest_window_opens_on_word_bounds():
    # Restricted to the "But ..." sentence onward, the window must open on
    # "But" at 15.82, not its segment's 16.08.
    doc = _real_doc()
    doc["segments"] = doc["segments"][4:]
    span = decide.decide_cue([_clip(doc)], cue_terms=[], target_s=60.0)["spans"][0]
    assert span["words"][0] == " But"
    assert span["speech_start"] == pytest.approx(15.82)
    assert span["start"] == pytest.approx(15.72)


def test_head_pad_never_runs_into_a_word_that_straddles_the_span_start():
    # A word from an overlapping segment that starts before the span and
    # ends inside it is a neighbour: the head may not pad back into it.
    from kai_slice.padding import pad_bounds

    doc = {
        "duration": 10.0,
        "segments": [
            {"start": 1.0, "end": 2.2, "words": [{"word": "a", "start": 1.0, "end": 2.2}]},
            {"start": 2.0, "end": 3.0, "words": [{"word": "b", "start": 2.0, "end": 3.0}]},
        ],
    }
    start, end = pad_bounds(_clip(doc), 2.0, 3.0)
    assert start == pytest.approx(2.0)
    assert end == pytest.approx(3.25)


def test_padded_bounds_round_outward_to_the_millisecond():
    # Off-grid word times: a nearest-round would pull the end in to 5.250
    # and push the start up to 1.900 -- both inside the pad. Outward rounding
    # floors the start and ceils the end.
    from kai_slice.padding import pad_bounds

    doc = {"duration": 10.0, "segments": []}
    start, end = pad_bounds(_clip(doc), 2.0004, 5.0004)
    assert start == pytest.approx(1.900)
    assert end == pytest.approx(5.251)
    assert start <= 2.0004 - 0.10 and end >= 5.0004 + 0.25
