"""End-to-end: the real four-stage run on an ffmpeg lavfi card.

Uses the fake transcriber injected through the documented ``KAI_SLICE_STT_REPLAY_DIR``
hook, so no model is needed. ffmpeg-gated.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import requires_ffmpeg

from kai_slice.console import TooManyCamerasError
from kai_slice.decide import DecideError
from kai_slice.run import run_pipeline

_PKG_ROOT = Path(__file__).resolve().parent.parent

SCRIPT = """\
1 KEVIN: hello there how are you doing
2 KEVIN2: i am doing great thanks for asking
3 KEVIN: i think the honest answer is yes
4 KEVIN2: no i strongly disagree with that
"""


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def _assert_manifest_hashes(out: Path) -> dict:
    manifest = json.loads((out / "ingest" / "manifest.json").read_text())
    for rec in manifest["files"]:
        data = (out / "ingest" / rec["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == rec["sha256"]
    return manifest


def _assert_cut_is_the_rendered_spans(out: Path, result: dict, spans: list[dict]) -> None:
    """The cut is exactly act's frame-snapped spans, and never shorter than
    the decided (padded) spans -- snapping only ever widens them."""
    decided = sum(s["end"] - s["start"] for s in spans)
    assert result["spans_total_s"] >= decided - 1e-6
    assert result["spans_total_s"] - decided <= 2 * len(spans) / 30.0
    assert abs(_ffprobe_duration(out / "cut.mp4") - result["spans_total_s"]) <= 1.0 / 30.0


@requires_ffmpeg
def test_run_pipeline_cue_leaves_every_artifact(lavfi_card, tmp_path, monkeypatch):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"
    result = run_pipeline(card, out, cue=[])

    _assert_manifest_hashes(out)
    assert (out / "decision.json").exists()
    assert (out / "cut.mp4").exists()
    assert "<xmeml" in (out / "edit.xml").read_text()
    assert (out / "console" / "ingest.json").exists()
    assert (out / "console" / "build.json").exists()

    rows = [json.loads(line) for line in (out / "run-record.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["capture", "transcribe", "decide", "act"]
    for r in rows:
        assert r["rc"] == 0
        assert r["wall_seconds"] >= 0.0

    _assert_cut_is_the_rendered_spans(out, result, result["decision"]["spans"])


@requires_ffmpeg
def test_run_pipeline_script_assembles_from_right_clips(lavfi_card, tmp_path, monkeypatch):
    card, replay = lavfi_card
    (card / "takes.json").write_text(
        json.dumps({"CLIP_0001.mp4": "KEVIN", "CLIP_0002.mp4": "KEVIN2"})
    )
    script = tmp_path / "script.txt"
    script.write_text(SCRIPT)
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"
    result = run_pipeline(card, out, script=script)
    spans = result["decision"]["spans"]
    assert [s["line"] for s in spans] == [1, 2, 3, 4]
    assert [s["clip"] for s in spans] == [
        "CLIP_0001.mp4",
        "CLIP_0002.mp4",
        "CLIP_0001.mp4",
        "CLIP_0002.mp4",
    ]
    _assert_cut_is_the_rendered_spans(out, result, spans)
    # console ingest still describes exactly the two media cards (not takes.json)
    ingest = json.loads((out / "console" / "ingest.json").read_text())
    assert [c["camera"] for c in ingest["cards"]] == ["A", "B"]


@requires_ffmpeg
def test_script_line_below_threshold_refuses_and_renders_nothing(lavfi_card, tmp_path, monkeypatch):
    card, replay = lavfi_card
    script = tmp_path / "script.txt"
    script.write_text(SCRIPT + "5 KEVIN: the quick brown fox jumps over the lazy dog\n")
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"
    with pytest.raises(DecideError) as exc:
        run_pipeline(card, out, script=script)
    assert str(exc.value).startswith("line 5 not found in any take, best ")
    assert not (out / "cut.mp4").exists()
    rows = [json.loads(line) for line in (out / "run-record.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["capture", "transcribe", "decide"]
    assert rows[-1]["rc"] == 1
    assert "line 5 not found" in rows[-1]["error"]


def test_card_with_more_than_three_clips_refuses_before_render(tmp_path, monkeypatch):
    """No ffmpeg needed: the camera-count refusal fires in the capture stage,
    before transcribe/decide/act ever run, so the extra "clips" only need to
    exist as files -- real media content is irrelevant to this check."""
    card = tmp_path / "card"
    card.mkdir()
    for i in range(4):
        (card / f"CLIP_000{i}.mp4").write_bytes(b"not-real-video")
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(tmp_path / "replay"))
    out = tmp_path / "out"

    with pytest.raises(TooManyCamerasError, match="at most 3 cameras"):
        run_pipeline(card, out, cue=[])

    assert not (out / "cut.mp4").exists()
    rows = [json.loads(line) for line in (out / "run-record.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["capture"]
    assert rows[-1]["rc"] == 1
    assert "at most 3 cameras" in rows[-1]["error"]
    # The refusal record is written from what is already known -- it must not
    # re-run the camera check (which would raise a second error in the handler).
    build = json.loads((out / "console" / "build.json").read_text())
    assert build["status"] == "refused"
    assert build["stopped_at"] == "capture"
    first = build["steps"][0]
    assert first["state"] == "refused"
    assert "at most 3 cameras" in first["refusal_reason"]
    assert not (out / "console" / "ingest.json").exists()


def _four_clip_card(tmp_path: Path) -> Path:
    card = tmp_path / "card"
    card.mkdir()
    for i in range(4):
        (card / f"CLIP_000{i}.mp4").write_bytes(b"not-real-video")
    return card


def test_a_failing_refusal_record_never_masks_the_original_refusal(tmp_path, monkeypatch):
    """A later-stage refusal whose console record then fails to write (disk,
    schema) must still surface the ORIGINAL refusal, not the record error."""
    import kai_slice.console as console_mod
    import kai_slice.run as run_mod

    card = tmp_path / "card"
    card.mkdir()
    (card / "CLIP_0001.mp4").write_bytes(b"not-real-video")

    def refuse(*_a, **_k):
        raise DecideError("transcribe refused: original reason")

    def boom(*_a, **_k):
        raise OSError("disk full while writing the console record")

    monkeypatch.setattr(run_mod, "_transcribe", refuse)
    monkeypatch.setattr(console_mod, "write_build", boom, raising=False)
    monkeypatch.setattr(console_mod, "write_ingest", boom, raising=False)
    monkeypatch.setattr(console_mod, "write_console", boom)

    with pytest.raises(DecideError, match="original reason"):
        run_pipeline(card, tmp_path / "out", cue=[], transcriber=lambda *_a: [])


def test_refusal_reason_in_build_json_carries_no_absolute_local_path(tmp_path, monkeypatch):
    import kai_slice.run as run_mod

    card = _four_clip_card(tmp_path)
    out = tmp_path / "out"

    def refuse(card_dir, out_dir):
        raise RuntimeError(f"cannot read {Path(card_dir).resolve()}/CLIP_0001.mp4 into {out_dir}")

    monkeypatch.setattr(run_mod, "capture_stage", refuse)
    with pytest.raises(RuntimeError, match="cannot read"):
        run_pipeline(card, out, cue=[], transcriber=lambda *_a: [])

    reason = json.loads((out / "console" / "build.json").read_text())["steps"][0]["refusal_reason"]
    assert str(tmp_path) not in reason
    assert reason == "cannot read <card>/CLIP_0001.mp4 into <out>"


@requires_ffmpeg
def test_cue_with_no_matching_term_refuses_instead_of_crashing(lavfi_card, tmp_path, monkeypatch):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"

    with pytest.raises(DecideError):
        run_pipeline(card, out, cue=["zzz_no_such_term_zzz"])

    assert not (out / "cut.mp4").exists()
    rows = [json.loads(line) for line in (out / "run-record.jsonl").read_text().splitlines()]
    assert rows[-1]["stage"] == "act"
    assert rows[-1]["rc"] == 1


@requires_ffmpeg
def test_refused_run_writes_console_build_json_with_refusal(lavfi_card, tmp_path, monkeypatch):
    card, replay = lavfi_card
    script = tmp_path / "script.txt"
    script.write_text(SCRIPT + "5 KEVIN: the quick brown fox jumps over the lazy dog\n")
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"

    with pytest.raises(DecideError):
        run_pipeline(card, out, script=script)

    build = json.loads((out / "console" / "build.json").read_text())
    assert build["status"] == "refused"
    assert build["stopped_at"] == "decide"
    decide_step = next(s for s in build["steps"] if s["name"] == "decide")
    assert decide_step["state"] == "refused"
    assert "line 5 not found" in decide_step["refusal_reason"]


@requires_ffmpeg
def test_cli_run_refuses_with_named_message_not_traceback_on_no_match_cue(lavfi_card, tmp_path):
    card, replay = lavfi_card
    out = tmp_path / "out"
    env = {
        **os.environ,
        "KAI_SLICE_STT_REPLAY_DIR": str(replay),
        "PYTHONPATH": os.pathsep.join([str(_PKG_ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kai_slice",
            "run",
            str(card),
            "--out",
            str(out),
            "--cue",
            "zzz_no_such_term_zzz",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "no spans to render" in proc.stderr
    assert not (out / "cut.mp4").exists()


@requires_ffmpeg
def test_cli_run_exits_zero_and_leaves_artifacts(lavfi_card, tmp_path):
    card, replay = lavfi_card
    out = tmp_path / "out"
    env = {
        **os.environ,
        "KAI_SLICE_STT_REPLAY_DIR": str(replay),
        "PYTHONPATH": os.pathsep.join([str(_PKG_ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "kai_slice", "run", str(card), "--out", str(out)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "cut.mp4").exists()
    assert (out / "decision.json").exists()
    assert (out / "console" / "build.json").exists()
    rows = (out / "run-record.jsonl").read_text().splitlines()
    assert len(rows) == 4
