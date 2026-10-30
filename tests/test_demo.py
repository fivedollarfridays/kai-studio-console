"""Red-first: ./demo — run kai_slice on a card and print the per-stage
Measured Performance table. Uses the documented KAI_SLICE_STT_REPLAY_DIR
hook (same as test_run_e2e.py) so no real model is needed for the fast
suite; a real-model run is verified manually, not in CI, per the project's
offline-by-default policy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import load_script, requires_ffmpeg

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEMO = _PKG_ROOT / "demo"


@requires_ffmpeg
def test_run_demo_prints_a_row_per_stage_and_a_total(lavfi_card, tmp_path, monkeypatch, capsys):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    module = load_script("demo")
    out = tmp_path / "demo-out"

    rc = module.main(
        ["--card", str(card), "--out", str(out), "--models-root", str(tmp_path / "models")]
    )

    assert rc == 0
    printed = capsys.readouterr().out
    for stage in ("capture", "transcribe", "decide", "act"):
        assert stage in printed
    assert "total" in printed.lower()
    assert (out / "cut.mp4").exists()


@requires_ffmpeg
def test_run_demo_refuses_without_a_model_or_replay(lavfi_card, tmp_path, monkeypatch):
    card, _replay = lavfi_card
    monkeypatch.delenv("KAI_SLICE_STT_REPLAY_DIR", raising=False)
    module = load_script("demo")
    out = tmp_path / "demo-out"
    models_root = tmp_path / "models"  # deliberately empty: no model fetched

    rc = module.main(["--card", str(card), "--out", str(out), "--models-root", str(models_root)])

    assert rc != 0
    assert not (out / "cut.mp4").exists()


def test_run_demo_refuses_when_out_dir_has_unrelated_content(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "important.txt"
    sentinel.write_text("do not delete me")
    module = load_script("demo")

    with pytest.raises(module.DemoOutputError):
        module.run_demo(out=out, models_root=tmp_path / "models")

    assert sentinel.read_text() == "do not delete me"


def test_run_demo_refuses_when_out_is_a_file(tmp_path):
    out = tmp_path / "out"
    out.write_text("not a directory")
    module = load_script("demo")

    with pytest.raises(module.DemoOutputError):
        module.run_demo(out=out, models_root=tmp_path / "models")

    assert out.read_text() == "not a directory"


def test_cli_demo_refuses_with_named_message_not_traceback(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "important.txt").write_text("keep me")
    module = load_script("demo")

    rc = module.main(["--out", str(out), "--models-root", str(tmp_path / "models")])

    assert rc == 1
    printed_err = capsys.readouterr().err
    assert "Traceback" not in printed_err
    assert (out / "important.txt").exists()


@requires_ffmpeg
def test_cli_demo_forced_ffmpeg_failure_prints_named_step_not_traceback(
    lavfi_card, tmp_path, monkeypatch, capsys
):
    """A forced ffmpeg failure (act.py's ``subprocess.run(..., check=True)``
    raising ``CalledProcessError``) must print one clean line naming the
    failed stage, not a raw traceback."""
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    module = load_script("demo")
    out = tmp_path / "demo-out"

    def _boom(inputs, graph, out):
        raise subprocess.CalledProcessError(
            1, ["ffmpeg", "-i", inputs[0]], output=b"", stderr=b"ffmpeg: boom"
        )

    monkeypatch.setattr("kai_slice.act._ffmpeg_concat", _boom)

    rc = module.main(
        ["--card", str(card), "--out", str(out), "--models-root", str(tmp_path / "models")]
    )

    assert rc == 1
    printed_err = capsys.readouterr().err
    assert "Traceback" not in printed_err
    assert "act" in printed_err
    assert len(printed_err.strip().splitlines()) == 1


@requires_ffmpeg
def test_run_demo_writes_exactly_the_entries_it_may_later_delete(lavfi_card, tmp_path, monkeypatch):
    """Pins DEMO_ENTRIES to what a real run writes: a new output the list
    misses would make every re-run refuse; an entry the list names that no
    run writes would let a user's file of that name be deleted."""
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    module = load_script("demo")
    out = tmp_path / "demo-out"

    module.run_demo(card=card, out=out, models_root=tmp_path / "models")

    assert {p.name for p in out.iterdir()} == module.DEMO_ENTRIES
    assert module.DEMO_ENTRIES == {
        ".kai-demo-out",
        "console",
        "cut.mp4",
        "decision.json",
        "edit.xml",
        "ingest",
        "run-record.jsonl",
        "transcripts",
    }


@requires_ffmpeg
def test_run_demo_replaces_its_own_previous_output(lavfi_card, tmp_path, monkeypatch):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    module = load_script("demo")
    out = tmp_path / "demo-out"
    models_root = tmp_path / "models"

    first = module.run_demo(card=card, out=out, models_root=models_root)
    assert (out / module.MARKER_NAME).is_file()
    assert Path(first["result"]["cut"]).exists()
    (out / "decision.json").write_text("stale previous decision")

    second = module.run_demo(card=card, out=out, models_root=models_root)

    assert (out / "decision.json").read_text() != "stale previous decision"
    assert Path(second["result"]["cut"]).exists()
    assert [p.name for p in tmp_path.iterdir() if "kai-demo-old" in p.name] == []


@requires_ffmpeg
def test_run_demo_refuses_a_user_file_added_to_its_own_previous_output(
    lavfi_card, tmp_path, monkeypatch
):
    # The empty-dir takeover case: ./demo marked a pre-existing dir, the user
    # later saved a file in it -- a re-run must not delete that file.
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    module = load_script("demo")
    out = tmp_path / "demo-out"
    out.mkdir()
    module.run_demo(card=card, out=out, models_root=tmp_path / "models")
    (out / "notes.txt").write_text("mine")

    with pytest.raises(module.DemoOutputError, match="notes.txt"):
        module.run_demo(card=card, out=out, models_root=tmp_path / "models")

    assert (out / "notes.txt").read_text() == "mine"
    assert (out / "cut.mp4").exists()


@requires_ffmpeg
def test_run_demo_uses_an_existing_empty_out_dir_as_is(lavfi_card, tmp_path, monkeypatch):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    module = load_script("demo")
    out = tmp_path / "demo-out"
    out.mkdir()

    data = module.run_demo(card=card, out=out, models_root=tmp_path / "models")

    assert Path(data["result"]["cut"]).exists()


@requires_ffmpeg
def test_cli_demo_exits_zero_with_replay(lavfi_card, tmp_path):
    card, replay = lavfi_card
    out = tmp_path / "demo-out"
    env = {
        **os.environ,
        "KAI_SLICE_STT_REPLAY_DIR": str(replay),
        "PYTHONPATH": os.pathsep.join([str(_PKG_ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(_DEMO),
            "--card",
            str(card),
            "--out",
            str(out),
            "--models-root",
            str(tmp_path / "models"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "cut.mp4").exists()
