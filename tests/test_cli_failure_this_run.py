"""The CLI's one-line failure message comes from THIS invocation only.

``run-record.jsonl`` is append-only, so an ``--out`` reused across runs
carries an earlier run's failed rows. A failure this invocation never
recorded in the ledger (``script-check`` writes no ledger rows at all; a
missing model refuses before the first stage) must print its own text, never
the stale row an earlier run left behind.
"""

from __future__ import annotations

import json

from kai_slice import cli
from kai_slice import model as model_mod
from kai_slice import run as run_mod
from kai_slice.model import NO_MODEL_MESSAGE

_OLD = "an earlier run's decide refusal"


def _seed_old_failure(out) -> None:
    out.mkdir(parents=True, exist_ok=True)
    row = {"stage": "decide", "wall_seconds": 0.1, "rc": 1, "inputs": [], "outputs": []}
    (out / "run-record.jsonl").write_text(json.dumps({**row, "error": _OLD}) + "\n")


def _card(tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    (card / "CLIP_0001.mp4").write_bytes(b"not-real-video")
    return card


def test_script_check_prints_its_own_failure_not_a_prior_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(run_mod.REPLAY_ENV, str(tmp_path / "replay"))
    out = tmp_path / "out"
    _seed_old_failure(out)
    script = tmp_path / "script.txt"
    script.write_text("HOST: hi\n")

    rc = cli.main(
        ["script-check", str(tmp_path / "no-such-card"), "--script", str(script), "--out", str(out)]
    )

    err = capsys.readouterr().err
    assert rc == 1
    assert _OLD not in err
    assert "card source not found" in err


def test_run_refused_before_any_stage_prints_its_own_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(model_mod, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.delenv(run_mod.REPLAY_ENV, raising=False)
    out = tmp_path / "out"
    _seed_old_failure(out)

    rc = cli.main(["run", str(_card(tmp_path)), "--out", str(out), "--cue", "hi"])

    err = capsys.readouterr().err
    assert rc == 1
    assert _OLD not in err
    assert NO_MODEL_MESSAGE in err


def test_demo_prints_its_own_refusal_not_a_prior_runs_ledger(tmp_path, capsys):
    """``./demo`` refusing an --out that is not its own (here: one an earlier
    ``kai_slice run`` left a failed ledger in) must print that refusal."""
    from conftest import load_script

    module = load_script("demo")
    out = tmp_path / "out"
    _seed_old_failure(out)

    rc = module.main(["--out", str(out), "--models-root", str(tmp_path / "models")])

    err = capsys.readouterr().err
    assert rc == 1
    assert _OLD not in err
    assert "refusing" in err
