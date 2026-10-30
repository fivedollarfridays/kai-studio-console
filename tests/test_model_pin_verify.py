"""Red-first: fetch_model's revision pin is enforced, never assumed.

Every branch uses a fake downloader (no network):

- ``revision=`` is passed ONLY to a downloader whose signature names that
  parameter explicitly (not ``**kwargs``, not an un-introspectable callable);
- when the pinned revision has shipped sha256 values
  (:data:`kai_slice.model_pins.PINNED_SHA256`), the fetched files are
  verified against them -- on the FIRST fetch too, so nothing is trusted on
  first use;
- otherwise the downloader must leave evidence (the snapshot path, or the
  Hugging Face local-dir metadata) naming the pinned commit, or the fetch
  refuses by name.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from conftest import FAKE_MODEL_FILES, fake_pins, write_fake_model

from kai_slice import model_pins
from kai_slice.model import (
    DEFAULT_MODEL_REVISION,
    ModelIntegrityError,
    ModelRevisionError,
    fetch_model,
)

REV = "cafe" * 10  # an unpinned 40-hex commit


def _metadata(output_dir, commit: str, name: str = "model.bin") -> None:
    meta = Path(output_dir) / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(f"{commit}\netag-{name}\n1761500000.0\n")


def test_kwargs_only_downloader_is_not_handed_revision(tmp_path):
    seen = []

    def fake(size_or_id, **kwargs):
        seen.append(kwargs)
        return write_fake_model(kwargs["output_dir"])

    fetch_model(
        "small", tmp_path / "m", downloader=fake, pinned_sha256=fake_pins(REV), revision=REV
    )

    assert "revision" not in seen[0]


def test_uninspectable_downloader_is_not_handed_revision(tmp_path):
    seen = []

    class Opaque:
        __signature__ = "not a signature"  # inspect.signature raises TypeError

        def __call__(self, size_or_id, **kwargs):
            seen.append(kwargs)
            return write_fake_model(kwargs["output_dir"])

    pins = fake_pins(REV)
    fetch_model("small", tmp_path / "m", downloader=Opaque(), pinned_sha256=pins, revision=REV)

    assert "revision" not in seen[0]


def test_unpinnable_downloader_without_shipped_hashes_refuses_before_downloading(tmp_path):
    calls = []

    def fake(size_or_id, output_dir=None, local_files_only=False):
        calls.append(size_or_id)
        return write_fake_model(output_dir)

    with pytest.raises(ModelRevisionError, match=REV):
        fetch_model(
            "small",
            tmp_path / "m",
            downloader=fake,
            pinned_sha256={},
            revision=REV,
            allow_unpinned=True,
        )

    assert calls == []


def test_named_revision_with_no_evidence_it_was_honoured_refuses(tmp_path):
    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        return write_fake_model(output_dir)  # ignores revision, says nothing

    with pytest.raises(ModelRevisionError, match=REV):
        fetch_model(
            "small",
            tmp_path / "m",
            downloader=fake,
            pinned_sha256={},
            revision=REV,
            allow_unpinned=True,
        )


def test_named_revision_honoured_per_download_metadata_is_accepted(tmp_path):
    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        _metadata(output_dir, revision)
        return write_fake_model(output_dir)

    dest = fetch_model(
        "small",
        tmp_path / "m",
        downloader=fake,
        pinned_sha256={},
        revision=REV,
        allow_unpinned=True,
    )

    assert (dest / "model.bin").is_file()


def test_download_metadata_naming_another_commit_refuses(tmp_path):
    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        _metadata(output_dir, "0" * 40)
        return write_fake_model(output_dir)

    with pytest.raises(ModelRevisionError, match=REV):
        fetch_model(
            "small",
            tmp_path / "m",
            downloader=fake,
            pinned_sha256={},
            revision=REV,
            allow_unpinned=True,
        )


def test_named_revision_honoured_per_snapshot_path_is_accepted(tmp_path):
    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        return str(write_fake_model(output_dir))

    dest = tmp_path / "snapshots" / REV
    assert (
        fetch_model(
            "small", dest, downloader=fake, pinned_sha256={}, revision=REV, allow_unpinned=True
        )
        == dest
    )


def test_hash_mismatch_deletes_the_model_files_and_says_so(tmp_path):
    dest = tmp_path / "m"

    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        return write_fake_model(output_dir, {**FAKE_MODEL_FILES, "model.bin": b"tampered"})

    with pytest.raises(ModelIntegrityError) as exc:
        fetch_model("small", dest, downloader=fake, pinned_sha256=fake_pins(REV), revision=REV)

    assert "model.bin" in str(exc.value) and str(dest) in str(exc.value)
    assert not (dest / "model.bin").exists()
    assert not (dest / "config.json").exists()  # nothing loadable is left behind


def test_missing_pinned_file_refuses(tmp_path):
    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        return write_fake_model(output_dir, {"model.bin": FAKE_MODEL_FILES["model.bin"]})

    with pytest.raises(ModelIntegrityError, match="config.json"):
        fetch_model(
            "small", tmp_path / "m", downloader=fake, pinned_sha256=fake_pins(REV), revision=REV
        )


def test_unexpected_file_refuses_and_is_deleted_never_left_for_the_user(tmp_path):
    # dest was proven empty or model-only before the download (model_dest),
    # so an unexpected file is the downloader's: it goes too, and the user is
    # never told to delete files themselves.
    dest = tmp_path / "m"

    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        return write_fake_model(output_dir, {**FAKE_MODEL_FILES, "preprocessor_config.json": b"{}"})

    with pytest.raises(ModelIntegrityError, match="preprocessor_config.json") as exc:
        fetch_model("small", dest, downloader=fake, pinned_sha256=fake_pins(REV), revision=REV)

    assert "yourself" not in str(exc.value)
    assert not (dest / "model.bin").exists()
    assert not (dest / "preprocessor_config.json").exists()


def test_downloader_dot_cache_metadata_is_not_hashed(tmp_path):
    dest = tmp_path / "m"
    stamps = iter(["1.0", "2.0"])

    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        meta = Path(output_dir) / ".cache" / "huggingface" / "download" / "model.bin.metadata"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(f"{revision}\netag\n{next(stamps)}\n")  # changes on every fetch
        return write_fake_model(output_dir)

    pins = fake_pins(REV)
    fetch_model("small", dest, downloader=fake, pinned_sha256=pins, revision=REV)
    fetch_model("small", dest, downloader=fake, pinned_sha256=pins, revision=REV)


def test_hashing_streams_the_file_rather_than_reading_it_whole(tmp_path, monkeypatch):
    def fake(size_or_id, output_dir=None, local_files_only=False, revision=None):
        return write_fake_model(output_dir)

    def no_whole_read(self):
        raise AssertionError(f"read_bytes() on {self} -- hash in chunks")

    monkeypatch.setattr(Path, "read_bytes", no_whole_read)
    fetch_model(
        "small", tmp_path / "m", downloader=fake, pinned_sha256=fake_pins(REV), revision=REV
    )


def test_shipped_pins_cover_the_default_revision():
    files = model_pins.PINNED_SHA256[DEFAULT_MODEL_REVISION]

    assert set(files) == {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in files.values())


def test_first_default_fetch_is_verified_against_shipped_pins_not_trusted(tmp_path):
    """No seam: a first-ever default fetch whose bytes are not the pinned
    upstream ones refuses -- there is no trust-on-first-use path left."""

    def fake(size_or_id, output_dir=None, local_files_only=False):  # cannot pin
        return write_fake_model(output_dir)

    with pytest.raises(ModelIntegrityError):
        fetch_model("small", tmp_path / "m", downloader=fake)

    assert hashlib.sha256(b"fake-weights").hexdigest() not in str(
        model_pins.PINNED_SHA256[DEFAULT_MODEL_REVISION]
    )
