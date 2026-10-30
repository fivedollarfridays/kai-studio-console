"""Red-first: two spans cut back to back from the same clip never repeat audio.

Each span's pads stop at the neighbouring WORDS, not at the neighbouring
SPAN. On the real sample transcript (see ``test_decide_cut_padding``),
"Hey, I'm Kevin," ends at 1.68 s and its tail runs to 1.93 s, while
"and this is Kai Studio." starts at 1.96 s and its head runs back to
1.86 s. Played in order, the cut would repeat 1.86-1.93 s. Where two
consecutive same-clip spans overlap or touch after padding, the gap between
the words is split at its midpoint: 1.82 s. Neither pad crosses it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kai_slice import decide
from kai_slice.clips import clip_from_doc
from kai_slice.padding import split_seams

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "card-01-CLIP_0001.stt.json"


def _clip():
    doc = json.loads(_FIXTURE.read_text())
    return clip_from_doc("CLIP_0001.mp4", "CLIP_0001.mp4", doc)


def test_back_to_back_lines_meet_at_the_midpoint_of_the_word_gap():
    script = "1. Hey, I'm Kevin,\n2. and this is Kai Studio."
    first, second = decide.decide_script([_clip()], script)["spans"]
    assert (first["speech_end"], second["speech_start"]) == pytest.approx((1.68, 1.96))
    assert first["end"] == pytest.approx(1.82)
    assert second["start"] == pytest.approx(1.82)
    assert first["end"] <= second["start"]


def test_the_split_keeps_the_outer_pads_and_the_speech_bounds():
    script = "1. Hey, I'm Kevin,\n2. and this is Kai Studio."
    first, second = decide.decide_script([_clip()], script)["spans"]
    assert first["start"] == pytest.approx(0.0)
    assert (first["speech_start"], first["speech_end"]) == pytest.approx((0.0, 1.68))
    assert (second["speech_start"], second["speech_end"]) == pytest.approx((1.96, 3.62))
    assert second["end"] == pytest.approx(3.87)


def test_spans_with_room_between_them_keep_their_full_pads():
    # "Kevin," ends 1.68 (tail 1.93); "Social" starts 4.54 (head 4.44).
    script = "1. Hey, I'm Kevin,\n2. Social media marketing made easy."
    first, second = decide.decide_script([_clip()], script)["spans"]
    assert first["end"] == pytest.approx(1.93)
    assert second["start"] == pytest.approx(4.44)


def test_spans_from_different_clips_are_never_split():
    spans = [
        {"clip": "A.mp4", "start": 0.0, "end": 1.93, "speech_start": 0.0, "speech_end": 1.68},
        {"clip": "B.mp4", "start": 1.86, "end": 3.87, "speech_start": 1.96, "speech_end": 3.62},
    ]
    assert split_seams(spans) == spans


def test_a_later_line_cut_from_earlier_in_the_clip_is_split_the_same_way():
    # Script order reversed: the seam is the same word gap, in time order.
    spans = [
        {"clip": "A.mp4", "start": 1.86, "end": 3.87, "speech_start": 1.96, "speech_end": 3.62},
        {"clip": "A.mp4", "start": 0.0, "end": 1.93, "speech_start": 0.0, "speech_end": 1.68},
    ]
    later, earlier = split_seams(spans)
    assert later["start"] == pytest.approx(1.82)
    assert earlier["end"] == pytest.approx(1.82)
