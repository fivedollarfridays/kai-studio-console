"""Red-first: fetch_model refuses BEFORE downloading when

- (a) the model has no shipped sha256 pin and the caller did not pass
  ``allow_unpinned=True`` (``scripts/fetch-model --allow-unpinned``);
- (b) ``dest`` is non-empty and is not already a model directory for this
  model -- a user's files are never downloaded into, and never named for
  the user to delete.

Every downloader here is a fake that records its calls; no network.
"""

from __future__ import annotations

import pytest
from conftest import fake_pins, write_fake_model

from kai_slice.model import (
    DEFAULT_MODEL_REVISION,
    UNPINNED_MARKER,
    ModelDestError,
    ModelRevisionError,
    fetch_model,
)

PINS = fake_pins(DEFAULT_MODEL_REVISION)
REV = "cafe" * 10


def _recorder(calls: list):
    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        calls.append(size_or_id)
        return write_fake_model(output_dir)

    return fake


def test_a_model_with_no_shipped_pin_refuses_without_the_flag(tmp_path):
    calls = []

    with pytest.raises(ModelRevisionError, match="--allow-unpinned"):
        fetch_model("base", tmp_path / "m", downloader=_recorder(calls), pinned_sha256=PINS)

    assert calls == []
    assert not (tmp_path / "m").exists()


def test_an_explicit_revision_with_no_shipped_pin_refuses_without_the_flag(tmp_path):
    calls = []

    with pytest.raises(ModelRevisionError, match="--allow-unpinned"):
        fetch_model(
            "small", tmp_path / "m", downloader=_recorder(calls), pinned_sha256={}, revision=REV
        )

    assert calls == []


def test_allow_unpinned_fetches_and_marks_the_dest_as_this_models(tmp_path):
    calls = []
    dest = tmp_path / "m"

    fetch_model("base", dest, downloader=_recorder(calls), pinned_sha256=PINS, allow_unpinned=True)

    assert calls == ["base"]
    assert (dest / UNPINNED_MARKER).read_text().splitlines()[0] == "base"


def test_a_dest_holding_a_users_file_refuses_before_downloading(tmp_path):
    calls = []
    dest = tmp_path / "m"
    dest.mkdir()
    (dest / "thesis.docx").write_text("mine")

    with pytest.raises(ModelDestError, match="thesis.docx") as exc:
        fetch_model("small", dest, downloader=_recorder(calls), pinned_sha256=PINS)

    assert calls == []
    assert (dest / "thesis.docx").read_text() == "mine"
    assert "delete" not in str(exc.value).lower()


def test_a_dest_holding_model_files_plus_a_users_file_refuses(tmp_path):
    calls = []
    dest = write_fake_model(tmp_path / "m")
    (dest / "notes.txt").write_text("mine")

    with pytest.raises(ModelDestError, match="notes.txt"):
        fetch_model("small", dest, downloader=_recorder(calls), pinned_sha256=PINS)

    assert calls == [] and (dest / "notes.txt").exists()


def test_a_dest_already_holding_this_pinned_model_is_refetched(tmp_path):
    calls = []
    dest = write_fake_model(tmp_path / "m")
    (dest / ".cache" / "huggingface").mkdir(parents=True)  # the downloader's own

    assert fetch_model("small", dest, downloader=_recorder(calls), pinned_sha256=PINS) == dest
    assert calls == ["small"]


def test_a_dest_marked_for_this_unpinned_model_is_refetched(tmp_path):
    calls = []
    dest = tmp_path / "m"
    fake = _recorder(calls)
    fetch_model("base", dest, downloader=fake, pinned_sha256=PINS, allow_unpinned=True)

    fetch_model("base", dest, downloader=fake, pinned_sha256=PINS, allow_unpinned=True)

    assert calls == ["base", "base"]


def test_a_dest_marked_for_another_unpinned_model_refuses(tmp_path):
    calls = []
    dest = tmp_path / "m"
    fake = _recorder(calls)
    fetch_model("base", dest, downloader=fake, pinned_sha256=PINS, allow_unpinned=True)

    with pytest.raises(ModelDestError, match="'tiny'"):
        fetch_model("tiny", dest, downloader=fake, pinned_sha256=PINS, allow_unpinned=True)

    assert calls == ["base"]


def test_a_dest_that_is_a_file_refuses(tmp_path):
    calls = []
    dest = tmp_path / "m"
    dest.write_text("mine")

    with pytest.raises(ModelDestError, match="not a directory"):
        fetch_model("small", dest, downloader=_recorder(calls), pinned_sha256=PINS)

    assert calls == [] and dest.read_text() == "mine"
