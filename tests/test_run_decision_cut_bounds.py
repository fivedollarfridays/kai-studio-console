"""Red-first: ``decision.json`` records the bounds the cut is actually
rendered on. act snaps every span outward to whole frames and clamps it to
the clip's last video frame; the decide stage writes those same bounds as
``cut_start``/``cut_end`` beside the padded ``start``/``end``, so reading
decision.json tells you exactly what is in ``cut.mp4``.

ffmpeg-gated; the lavfi card's transcripts come through the documented
``KAI_SLICE_STT_REPLAY_DIR`` hook, so no model is needed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from conftest import requires_ffmpeg

from kai_slice.run import run_pipeline

_LAVFI_FPS = 30.0


def _video_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip())


@requires_ffmpeg
def test_decision_json_carries_the_frame_snapped_bounds_act_renders(
    lavfi_card, tmp_path, monkeypatch
):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"

    result = run_pipeline(card, out, cue=["i"])

    spans = json.loads((out / "decision.json").read_text())["spans"]
    assert spans
    frames = 0
    for s in spans:
        assert s["cut_start"] <= s["start"] and s["cut_end"] >= s["end"]
        for key in ("cut_start", "cut_end"):
            assert s[key] * _LAVFI_FPS == pytest.approx(round(s[key] * _LAVFI_FPS), abs=1e-4)
        frames += round((s["cut_end"] - s["cut_start"]) * _LAVFI_FPS)
    assert result["spans_total_s"] == pytest.approx(
        sum(s["cut_end"] - s["cut_start"] for s in spans), abs=1e-6
    )
    assert _video_frames(out / "cut.mp4") == frames
