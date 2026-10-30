"""Red-first: kai_slice.model.fetch_model, the pre-fetch used by scripts/fetch-model.

The unit tests inject a fake downloader and a matching ``pinned_sha256``
table (no network); the pin-enforcement branches live in
``test_model_pin_verify.py``. A cache-only
integration test exercises the real faster-whisper download path with
``local_files_only=True`` against a model already in the local Hugging Face
cache, so it proves the real wiring without ever downloading over the
network. Its skip guard probes that same cache, also with
``local_files_only=True``, so the probe itself never touches the network
either: when the `small` model is not already cached, the probe raises and
the test is SKIPPED with a reason naming the missing cache, instead of
silently falling through to a live download.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from conftest import fake_pins, write_fake_model
from faster_whisper.utils import download_model

from kai_slice.model import DEFAULT_MODEL_REVISION, fetch_model

PINS = fake_pins(DEFAULT_MODEL_REVISION)


def _small_model_is_cached() -> bool:
    """True when the `small` faster-whisper model is already in the local
    Hugging Face cache. Checked with ``local_files_only=True``, which makes
    huggingface_hub raise instead of downloading when the model is absent --
    so this probe itself never makes a network call."""
    try:
        download_model("small", local_files_only=True)
    except OSError:
        return False
    return True


requires_cached_small_model = pytest.mark.skipif(
    not _small_model_is_cached(),
    reason="the `small` faster-whisper model is not in the local Hugging Face cache",
)


def test_fetch_model_calls_the_injected_downloader_with_dest_dir(tmp_path):
    calls = []

    def fake_download(size_or_id, output_dir=None, local_files_only=False, revision=None):
        calls.append((size_or_id, output_dir, local_files_only))
        return write_fake_model(output_dir)

    dest = tmp_path / "models" / "small"
    result = fetch_model("small", dest, downloader=fake_download, pinned_sha256=PINS)

    assert calls == [("small", str(dest), False)]
    assert result == dest


def test_fetch_model_pins_the_default_revision_to_the_injected_downloader(tmp_path):
    """The default ("small") model's fetch pins a Hugging Face revision, so a
    later push to the upstream repo cannot silently swap the weights."""
    calls = []

    def fake_download(size_or_id, output_dir=None, local_files_only=False, revision=None):
        calls.append((size_or_id, revision))
        return write_fake_model(output_dir)

    fetch_model("small", tmp_path / "small", downloader=fake_download, pinned_sha256=PINS)

    assert calls == [("small", DEFAULT_MODEL_REVISION)]


def test_fetch_model_explicit_revision_overrides_the_default_pin(tmp_path):
    calls = []

    def fake_download(size_or_id, output_dir=None, local_files_only=False, revision=None):
        calls.append(revision)
        return write_fake_model(output_dir)

    pins = fake_pins("deadbeef")
    fetch_model(
        "small",
        tmp_path / "small",
        downloader=fake_download,
        revision="deadbeef",
        pinned_sha256=pins,
    )

    assert calls == ["deadbeef"]


def test_fetch_model_passes_local_files_only_through(tmp_path):
    calls = []

    def fake_download(size_or_id, output_dir=None, local_files_only=False, revision=None):
        calls.append(local_files_only)
        return write_fake_model(output_dir)

    fetch_model(
        "small",
        tmp_path / "small",
        downloader=fake_download,
        local_files_only=True,
        pinned_sha256=PINS,
    )

    assert calls == [True]


@requires_cached_small_model
def test_fetch_model_real_download_model_reuses_local_cache():
    """The real faster-whisper downloader, pointed at the model's own cached
    snapshot directory with ``local_files_only=True`` -- proves the wiring
    end to end by reusing what is already on disk, with no network call."""
    cached_dir = Path(download_model("small", local_files_only=True))
    result = fetch_model("small", cached_dir, local_files_only=True)

    assert result == cached_dir
    assert (result / "model.bin").is_file()


def test_real_download_test_passes_local_files_only():
    """Mutation guard: if `local_files_only=True` were dropped from the
    `fetch_model` call above, this must go red -- dropping it would silently
    restore a live network download in CI."""
    source = inspect.getsource(test_fetch_model_real_download_model_reuses_local_cache)
    tree = ast.parse(source)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fetch_model"
    )
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    local_files_only = kwargs.get("local_files_only")

    assert isinstance(local_files_only, ast.Constant)
    assert local_files_only.value is True


def test_fetch_model_silences_the_local_dir_use_symlinks_deprecation_warning(tmp_path):
    """huggingface_hub's real snapshot_download warns
    ``The `local_dir_use_symlinks` argument is deprecated and ignored`` for
    every call faster_whisper's download_model makes (it always passes
    ``output_dir``, which faster_whisper turns into that kwarg). The fetch
    step must not spam that at the user on every run."""
    import warnings

    def fake_download(size_or_id, output_dir=None, local_files_only=False, revision=None):
        warnings.warn(
            "The `local_dir_use_symlinks` argument is deprecated and ignored in "
            "`snapshot_download`. Downloading to a local directory does not use "
            "symlinks anymore.",
            UserWarning,
            stacklevel=2,
        )
        return write_fake_model(output_dir)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fetch_model("small", tmp_path / "small", downloader=fake_download, pinned_sha256=PINS)

    assert not any("local_dir_use_symlinks" in str(w.message) for w in caught)
