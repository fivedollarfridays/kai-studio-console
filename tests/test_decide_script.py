"""Red-first: script mode assembles a conversation from separate takes."""

from __future__ import annotations

import os

import pytest

from kai_slice import decide
from kai_slice.clips import clip_from_doc, load_clips


def _doc(sentences, *, start=0.0):
    """House doc whose segments are the given sentences, 0.5s per word."""
    segments = []
    t = start
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


SCRIPT = """\
1 KEVIN: hello there how are you doing
2 KEVIN2: i am doing great thanks for asking
3 KEVIN: i think the honest answer is yes
4 KEVIN2: no i strongly disagree with that
"""


def _two_speaker_clips():
    # Each clip holds one speaker's lines, in a DIFFERENT order than the script.
    clip_a = clip_from_doc(
        "A.mp4",
        "A.mp4",
        _doc(["i think the honest answer is yes", "hello there how are you doing"]),
        creation_time="2026-10-26T10:00:00.000000Z",
    )
    clip_b = clip_from_doc(
        "B.mp4",
        "B.mp4",
        _doc(["no i strongly disagree with that", "i am doing great thanks for asking"]),
        creation_time="2026-10-26T10:05:00.000000Z",
    )
    return [clip_a, clip_b]


def test_assembles_in_script_order_from_right_clips():
    decision = decide.decide_script(_two_speaker_clips(), SCRIPT)
    assert decision["mode"] == "script"
    spans = decision["spans"]
    assert [s["line"] for s in spans] == [1, 2, 3, 4]
    assert [s["clip"] for s in spans] == ["A.mp4", "B.mp4", "A.mp4", "B.mp4"]
    for s in spans:
        assert s["score"] == pytest.approx(1.0)
        assert s["end"] > s["start"]
        # opens/closes on a word boundary: start is some word's start, end some end
    assert spans[0]["words"][0] == "hello"


def test_repeated_take_ties_to_later_creation_time():
    line = "repeat this exact line please now"
    early = clip_from_doc("T1.mp4", "T1.mp4", _doc([line]), creation_time="2026-10-26T09:00:00Z")
    late = clip_from_doc("T2.mp4", "T2.mp4", _doc([line]), creation_time="2026-10-26T11:00:00Z")
    decision = decide.decide_script([early, late], f"1 KEVIN: {line}\n")
    assert decision["spans"][0]["clip"] == "T2.mp4"
    assert decision["spans"][0]["score"] == pytest.approx(1.0)


def test_below_threshold_refuses_by_line_number_score_and_clip():
    clips = [
        clip_from_doc(
            "CLIP_0003.mp4", "CLIP_0003.mp4", _doc(["totally unrelated words appear here"])
        )
    ]
    script = "7 KEVIN: the quick brown fox jumps over\n"
    with pytest.raises(decide.DecideError) as exc:
        decide.decide_script(clips, script)
    msg = str(exc.value)
    assert msg.startswith("line 7 not found in any take, best ")
    assert msg.endswith(" in CLIP_0003.mp4")
    # the reported score is a two-decimal fraction below the 0.70 bar
    reported = float(msg.split("best ")[1].split(" in")[0])
    assert reported < 0.70


def test_takes_json_restricts_a_line_to_its_speakers_clips(tmp_path):
    # Both clips contain the line verbatim; B is later, so WITHOUT takes B wins.
    line = "the quick brown fox jumps high"
    ingest = tmp_path / "ingest"
    trans = tmp_path / "transcripts"
    ingest.mkdir()
    trans.mkdir()
    for name, _ct in (("A.mp4", "2026-10-26T09:00:00Z"), ("B.mp4", "2026-10-26T12:00:00Z")):
        (ingest / name).write_bytes(b"not-a-real-mp4")
        import json

        (trans / f"{name.split('.')[0]}.stt.json").write_text(json.dumps(_doc([line])))
    takes = {"A.mp4": "KEVIN", "B.mp4": "KEVIN2"}
    clips = load_clips(ingest, trans, takes=takes, creation_time_of=lambda p: takes and "")
    decision = decide.decide_script(clips, f"1 KEVIN: {line}\n")
    assert decision["spans"][0]["clip"] == "A.mp4"


def test_unknown_speaker_refuses_naming_line_speaker_and_declared(tmp_path):
    # Takes declared for K1 and K2 only; the script tags a line K3.
    line = "the quick brown fox jumps high"
    ingest = tmp_path / "ingest"
    trans = tmp_path / "transcripts"
    ingest.mkdir()
    trans.mkdir()
    for name, _ct in (("A.mp4", "2026-10-26T09:00:00Z"), ("B.mp4", "2026-10-26T12:00:00Z")):
        (ingest / name).write_bytes(b"not-a-real-mp4")
        import json

        (trans / f"{name.split('.')[0]}.stt.json").write_text(json.dumps(_doc([line])))
    takes = {"A.mp4": "K1", "B.mp4": "K2"}
    clips = load_clips(ingest, trans, takes=takes, creation_time_of=lambda p: takes and "")
    with pytest.raises(decide.DecideError) as exc:
        decide.decide_script(clips, f"5 K3: {line}\n")
    msg = str(exc.value)
    assert "5" in msg
    assert "K3" in msg
    assert "K1" in msg and "K2" in msg


def test_tie_resolves_by_name_independent_of_mtime(tmp_path):
    import json

    line = "identical content in both takes here"
    ingest = tmp_path / "ingest"
    trans = tmp_path / "transcripts"
    ingest.mkdir()
    trans.mkdir()
    paths = {}
    for name in ("A.mp4", "B.mp4"):
        p = ingest / name
        p.write_bytes(b"x")
        paths[name] = p
        (trans / f"{name.split('.')[0]}.stt.json").write_text(json.dumps(_doc([line])))
    # swap mtimes: give A a LATER mtime than B — it must not matter
    os.utime(paths["A.mp4"], (2_000_000_000, 2_000_000_000))
    os.utime(paths["B.mp4"], (1_000_000_000, 1_000_000_000))
    # equal creation_time for both → the tiebreak falls to file name
    clips = load_clips(ingest, trans, creation_time_of=lambda p: "2026-10-26T10:00:00Z")
    decision = decide.decide_script(clips, f"1 KEVIN: {line}\n")
    assert decision["spans"][0]["clip"] == "B.mp4"  # greater name = later take


def test_script_check_returns_rows_and_writes_nothing(tmp_path):
    rows = decide.script_check(_two_speaker_clips(), SCRIPT)
    assert [r["line"] for r in rows] == [1, 2, 3, 4]
    for r in rows:
        assert set(r) >= {"line", "speaker", "best_score", "best_clip", "transcript_words"}
    assert list(tmp_path.iterdir()) == []  # script-check renders/writes nothing


def test_script_check_reports_unknown_speaker_as_a_row_not_a_raise(tmp_path):
    # Takes declared for K1 and K2 only; the script tags lines K1, K3, K2 --
    # script_check must still return all three rows, scoring K1 and K2
    # normally and carrying an error on K3's row naming K3, K1 and K2.
    import json

    line = "the quick brown fox jumps high"
    ingest = tmp_path / "ingest"
    trans = tmp_path / "transcripts"
    ingest.mkdir()
    trans.mkdir()
    takes = {"A.mp4": "K1", "B.mp4": "K2"}
    for name in takes:
        (ingest / name).write_bytes(b"not-a-real-mp4")
        (trans / f"{name.split('.')[0]}.stt.json").write_text(json.dumps(_doc([line])))
    clips = load_clips(
        ingest, trans, takes=takes, creation_time_of=lambda p: "2026-10-26T10:00:00Z"
    )
    script = f"1 K1: {line}\n2 K3: {line}\n3 K2: {line}\n"

    rows = decide.script_check(clips, script)

    assert [r["line"] for r in rows] == [1, 2, 3]
    k3_row = rows[1]
    assert k3_row["best_score"] == 0.0
    assert k3_row["best_clip"] is None
    assert k3_row["transcript_words"] == []
    assert "K3" in k3_row["error"] and "K1" in k3_row["error"] and "K2" in k3_row["error"]
    for r in (rows[0], rows[2]):
        assert "error" not in r
        assert r["best_score"] == pytest.approx(1.0)
