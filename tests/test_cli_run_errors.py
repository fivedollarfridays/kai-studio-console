"""Red-first: ``python -m kai_slice run`` prints one clean line naming the
failed stage for an ffmpeg or decode crash, not a raw traceback
(the same gap ``./demo``'s CLI had, ``kai_slice/cli.py``'s own
entry point)."""

from __future__ import annotations

import subprocess

from conftest import requires_ffmpeg

from kai_slice import act, cli


@requires_ffmpeg
def test_run_forced_ffmpeg_failure_prints_named_step_not_traceback(
    lavfi_card, tmp_path, monkeypatch, capsys
):
    card, replay = lavfi_card
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(replay))
    out = tmp_path / "out"

    def _boom(inputs, graph, cut_path):
        raise subprocess.CalledProcessError(
            1, ["ffmpeg", "-i", inputs[0]], output=b"", stderr=b"ffmpeg: boom"
        )

    monkeypatch.setattr(act, "_ffmpeg_concat", _boom)

    rc = cli.main(["run", str(card), "--out", str(out)])

    assert rc == 1
    printed_err = capsys.readouterr().err
    assert "Traceback" not in printed_err
    assert "act" in printed_err
    assert len(printed_err.strip().splitlines()) == 1
