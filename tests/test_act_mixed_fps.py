"""Red-first: act snaps each span on ITS OWN clip's frame grid.

Before this fix one fps -- the first input's -- snapped every span, so a span
from a second clip at another rate was trimmed between that clip's frames
and clamped to a "last frame" counted at the wrong rate. The two clips here
are cut from the shipped sample (``samples/card-01/CLIP_0001.mp4``) by
ffmpeg: its first 2 s re-encoded at 30000/1001 fps (59 frames) and at
25 fps (50 frames).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import requires_ffmpeg

from kai_slice.act import act_stage, rendered_bounds

_SAMPLE_CLIP = Path(__file__).resolve().parent.parent / "samples" / "card-01" / "CLIP_0001.mp4"
_NTSC = 30000 / 1001


def _cut_from_sample(out: Path, rate: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(_SAMPLE_CLIP), "-t", "2", "-r", rate,
         "-vf", "scale=320:180", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(out)],
        check=True, capture_output=True,
    )  # fmt: skip
    return out


@pytest.fixture
def mixed_fps_clips(tmp_path) -> dict[str, str]:
    return {
        "NTSC.mp4": str(_cut_from_sample(tmp_path / "NTSC.mp4", "30000/1001")),
        "PAL.mp4": str(_cut_from_sample(tmp_path / "PAL.mp4", "25")),
    }


def _frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip())


@requires_ffmpeg
def test_a_second_clip_snaps_on_its_own_frame_grid(mixed_fps_clips):
    spans = [
        {"clip": "NTSC.mp4", "start": 0.5, "end": 1.23},
        {"clip": "PAL.mp4", "start": 0.5, "end": 1.23},
    ]
    ntsc, pal = rendered_bounds(spans, mixed_fps_clips)
    # 0.5 s = NTSC frame 14.99 -> 14; 1.23 s = frame 36.86 -> 37
    assert ntsc["start"] == pytest.approx(14 / _NTSC, abs=1e-6)
    assert ntsc["end"] == pytest.approx(37 / _NTSC, abs=1e-6)
    # 0.5 s = PAL frame 12.5 -> 12; 1.23 s = frame 30.75 -> 31
    assert pal["start"] == pytest.approx(12 / 25, abs=1e-6)
    assert pal["end"] == pytest.approx(31 / 25, abs=1e-6)


@requires_ffmpeg
def test_each_clip_clamps_to_its_own_last_frame(mixed_fps_clips):
    spans = [
        {"clip": "NTSC.mp4", "start": 1.5, "end": 9.0},
        {"clip": "PAL.mp4", "start": 1.5, "end": 9.0},
    ]
    ntsc, pal = rendered_bounds(spans, mixed_fps_clips)
    paths = mixed_fps_clips
    assert ntsc["end"] == pytest.approx(_frames(Path(paths["NTSC.mp4"])) / _NTSC, abs=1e-6)
    assert pal["end"] == pytest.approx(_frames(Path(paths["PAL.mp4"])) / 25, abs=1e-6)
    assert pal["end"] == pytest.approx(2.0, abs=1e-6)


@requires_ffmpeg
def test_act_renders_a_mixed_rate_cut_of_the_snapped_length(mixed_fps_clips, tmp_path):
    spans = [
        {"clip": "NTSC.mp4", "start": 0.5, "end": 1.23},
        {"clip": "PAL.mp4", "start": 0.5, "end": 1.23},
    ]
    result = act_stage(spans, mixed_fps_clips, tmp_path)
    expected = (37 - 14) / _NTSC + (31 - 12) / 25
    assert result["spans_total_s"] == pytest.approx(expected, abs=1e-6)
    assert Path(result["cut"]).stat().st_size > 0
