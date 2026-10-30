"""A missing model is refused BEFORE capture copies the card.

Capture is a full verified copy of the card into ``<out>/ingest/``; a run that
is going to refuse at transcribe for want of a model must not spend that copy
(or leave a half-run ``ingest/`` behind) first.
"""

from __future__ import annotations

import re

import pytest

from kai_slice import model as model_mod
from kai_slice import run as run_mod
from kai_slice.model import NO_MODEL_MESSAGE, ModelNotFoundError


def _card(tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    (card / "CLIP_0001.mp4").write_bytes(b"not-real-video")
    return card


@pytest.fixture
def no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.delenv(run_mod.REPLAY_ENV, raising=False)


def test_run_without_a_model_writes_no_ingest_dir(tmp_path, no_model):
    out = tmp_path / "out"
    with pytest.raises(ModelNotFoundError, match=re.escape(NO_MODEL_MESSAGE)):
        run_mod.run_pipeline(_card(tmp_path), out, cue=["hi"])
    assert not (out / "ingest").exists()


def test_script_check_without_a_model_writes_no_ingest_dir(tmp_path, no_model):
    out = tmp_path / "out"
    script = tmp_path / "script.txt"
    script.write_text("HOST: hi\n")
    with pytest.raises(ModelNotFoundError, match=re.escape(NO_MODEL_MESSAGE)):
        run_mod.run_script_check(_card(tmp_path), script=script, out_dir=out)
    assert not (out / "ingest").exists()
