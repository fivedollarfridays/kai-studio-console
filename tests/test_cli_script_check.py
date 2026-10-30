"""Unit: ``kai_slice.cli._script_check`` prints REFUSED rows and exits non-zero.

Monkeypatches ``run_script_check`` so this stays a fast unit test of the CLI's
own printing/exit-code logic -- no card, model or ffmpeg needed.
"""

from __future__ import annotations

from types import SimpleNamespace

from kai_slice import cli


def test_script_check_prints_refused_line_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    rows = [
        {
            "line": 1,
            "speaker": "K1",
            "best_score": 1.0,
            "best_clip": "A.mp4",
            "transcript_words": ["hi"],
        },
        {
            "line": 2,
            "speaker": "K3",
            "best_score": 0.0,
            "best_clip": None,
            "transcript_words": [],
            "error": "line 2: speaker 'K3' not found in any take; declared speakers are K1, K2",
        },
        {
            "line": 3,
            "speaker": "K2",
            "best_score": 1.0,
            "best_clip": "B.mp4",
            "transcript_words": ["hi"],
        },
    ]
    monkeypatch.setattr(cli, "run_script_check", lambda *a, **k: rows)
    args = SimpleNamespace(
        card_dir=tmp_path,
        script=tmp_path / "script.txt",
        out=tmp_path / "out",
        model="small",
        device="cpu",
    )

    rc = cli._script_check(args)

    printed = capsys.readouterr().out
    assert rc == 1
    assert "line 2 [K3]: REFUSED " in printed
    assert "K3" in printed and "K1" in printed and "K2" in printed
    assert "line 1 [K1]: best 1.00 in A.mp4" in printed
    assert "line 3 [K2]: best 1.00 in B.mp4" in printed


def test_script_check_exits_zero_when_every_line_scores(tmp_path, monkeypatch, capsys):
    rows = [
        {
            "line": 1,
            "speaker": "K1",
            "best_score": 1.0,
            "best_clip": "A.mp4",
            "transcript_words": ["hi"],
        }
    ]
    monkeypatch.setattr(cli, "run_script_check", lambda *a, **k: rows)
    args = SimpleNamespace(
        card_dir=tmp_path,
        script=tmp_path / "script.txt",
        out=tmp_path / "out",
        model="small",
        device="cpu",
    )

    rc = cli._script_check(args)

    assert rc == 0
    assert "REFUSED" not in capsys.readouterr().out
