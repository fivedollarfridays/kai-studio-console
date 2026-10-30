"""Red-first: act_stage validates span timings before building the ffmpeg
filter graph -- a NaN/inf/negative value never reaches the formatted
command string. No ffmpeg needed: every case here refuses before
subprocess.run would be called.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from conftest import requires_ffmpeg

from kai_slice.act import _snap_to_frames, act_stage
from kai_slice.decide import DecideError

_SAMPLE_CLIP = Path(__file__).resolve().parent.parent / "samples" / "card-01" / "CLIP_0001.mp4"


def _span(**overrides) -> dict:
    span = {"clip": "A.mp4", "start": 0.0, "end": 1.0}
    span.update(overrides)
    return span


@pytest.mark.parametrize(
    "bad_value",
    [math.nan, math.inf, -math.inf, -1.0],
    ids=["nan", "inf", "-inf", "negative"],
)
def test_act_stage_refuses_non_finite_or_negative_start(tmp_path, monkeypatch, bad_value):
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: pytest.fail("ffmpeg must not run on a bad span")
    )
    spans = [_span(start=bad_value)]

    with pytest.raises(DecideError, match="invalid span timing"):
        act_stage(spans, {"A.mp4": "A.mp4"}, tmp_path)


def test_act_stage_refuses_end_before_start(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: pytest.fail("ffmpeg must not run on a bad span")
    )
    spans = [_span(start=5.0, end=1.0)]

    with pytest.raises(DecideError, match="invalid span timing"):
        act_stage(spans, {"A.mp4": "A.mp4"}, tmp_path)


def test_act_stage_refuses_no_spans_with_a_decide_error(tmp_path):
    with pytest.raises(DecideError, match="no spans to render"):
        act_stage([], {}, tmp_path)


@requires_ffmpeg
def test_act_stage_edit_xml_declares_the_sample_clips_real_dimensions(tmp_path):
    """The shipped sample (``samples/card-01``) is a real 1280x720 clip --
    its edit.xml must carry that size for both the sequence and the media,
    not the previous 160x120 hard-code."""
    spans = [_span(clip="CLIP_0001.mp4", start=0.0, end=0.5)]

    result = act_stage(spans, {"CLIP_0001.mp4": str(_SAMPLE_CLIP)}, tmp_path)

    edit_xml = Path(result["edit"]).read_text()
    assert edit_xml.count("<width>1280</width><height>720</height>") == 2
    assert "<width>160</width><height>120</height>" not in edit_xml


# The shipped sample is 30000/1001 fps; its demo span is 0.00-32.05 s (the
# 31.80 s end of "how." plus the 0.25 s tail -- see test_decide_cut_padding).
_SAMPLE_FPS = 30000 / 1001


def test_frame_snap_widens_a_span_outward_to_whole_frames():
    """A video trim can only stop on a frame, so act snaps each span OUTWARD
    (start floors, end ceils): the rounding never cuts into the padded air,
    and the cut's length is exactly the sum of the snapped spans."""
    snapped = _snap_to_frames([_span(start=1.16, end=32.05)], _SAMPLE_FPS)[0]
    assert snapped["start"] == pytest.approx(34 / _SAMPLE_FPS, abs=1e-6)  # 1.16 s = frame 34.77
    assert snapped["end"] == pytest.approx(961 / _SAMPLE_FPS, abs=1e-6)  # 32.05 s = frame 960.55
    assert snapped["start"] <= 1.16 and snapped["end"] >= 32.05


def test_frame_snap_keeps_an_on_grid_span_unchanged():
    snapped = _snap_to_frames([_span(start=0.0, end=960 / _SAMPLE_FPS)], _SAMPLE_FPS)[0]
    assert snapped["start"] == 0.0
    assert snapped["end"] == pytest.approx(960 / _SAMPLE_FPS, abs=1e-6)


# The shipped sample's video stream is 997 frames (33.2666 s) while its
# format/audio duration is 33.28 s -- the transcript's duration, so a tail
# clamped "to the clip" can still sit past the last video frame.
_SAMPLE_FRAMES = 997


def test_frame_snap_clamps_the_end_to_the_last_whole_video_frame():
    snapped = _snap_to_frames(
        [_span(clip="CLIP_0001.mp4", start=32.5, end=33.28)],
        _SAMPLE_FPS,
        {"CLIP_0001.mp4": _SAMPLE_FRAMES},
    )[0]
    assert snapped["end"] == pytest.approx(_SAMPLE_FRAMES / _SAMPLE_FPS, abs=1e-6)
    assert snapped["start"] == pytest.approx(974 / _SAMPLE_FPS, abs=1e-6)  # 32.5 s = frame 974.03


def test_frame_snap_keeps_at_least_one_frame_when_the_start_is_past_the_last_frame():
    snapped = _snap_to_frames(
        [_span(clip="CLIP_0001.mp4", start=33.27, end=33.28)],
        _SAMPLE_FPS,
        {"CLIP_0001.mp4": _SAMPLE_FRAMES},
    )[0]
    assert snapped["end"] == pytest.approx(_SAMPLE_FRAMES / _SAMPLE_FPS, abs=1e-6)
    assert snapped["start"] == pytest.approx((_SAMPLE_FRAMES - 1) / _SAMPLE_FPS, abs=1e-6)


def _frames(path: Path) -> int:
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip())


@requires_ffmpeg
def test_act_on_a_tail_clamped_at_the_clip_end_never_exceeds_the_media(tmp_path):
    """``act_stage([{32.5, 33.28}])`` on the real sample: the cut holds
    exactly the frames 974..996, and every XMEML ``<out>`` is within the
    file's ``<duration>``."""
    import re

    spans = [_span(clip="CLIP_0001.mp4", start=32.5, end=33.28)]

    result = act_stage(spans, {"CLIP_0001.mp4": str(_SAMPLE_CLIP)}, tmp_path)

    assert _frames(Path(result["cut"])) == _SAMPLE_FRAMES - 974
    xml = Path(result["edit"]).read_text()
    file_duration = int(re.search(r"<file id=[^>]*>.*?<duration>(\d+)</duration>", xml).group(1))
    assert file_duration == _SAMPLE_FRAMES
    outs = [int(o) for o in re.findall(r"<out>(\d+)</out>", xml)]
    assert outs and max(outs) <= file_duration
