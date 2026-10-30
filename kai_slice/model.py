"""kai_slice.model — fetch a transcription model, with no raw network import.

This module imports no ``socket``/``urllib``/``requests``/``httpx``: the
actual outbound fetch is performed by faster-whisper's own downloader, the
one documented exception the structural test (``tests/test_structural.py``)
names explicitly rather than carrying this module in ``NETWORK_ALLOWED``.
Every other module in :mod:`kai_slice` is offline: it reads no secret and
opens no socket once the model is present (the only other network-reaching
code path is ``kai_slice serve``, which may LISTEN on
127.0.0.1). The CLI run path never falls back to faster-whisper's own cache
or an implicit download: :func:`resolve_model` requires the local directory
``scripts/fetch-model`` (via :func:`fetch_model`) already placed under
:data:`MODELS_ROOT`, and refuses with :data:`NO_MODEL_MESSAGE` otherwise.
``fetch_model`` itself is the explicit, air-gap-friendly pre-fetch to a
caller-named directory -- the only step that may touch the network.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kai_slice.model_dest import (
    UNPINNED_MARKER,
    ModelDestError,
    require_fetchable_dest,
    write_unpinned_marker,
)
from kai_slice.model_pins import PINNED_SHA256
from kai_slice.model_verify import (
    ModelIntegrityError,
    ModelRevisionError,
    names_revision_param,
    require_pin_evidence,
    verify_pinned_hashes,
)
from kai_slice.transcribe import DEFAULT_MODEL

#: Where ``scripts/fetch-model`` writes a fetched model by default (sibling
#: to ``scripts/``, regardless of caller cwd) -- the one place
#: :func:`resolve_model` looks when ``--model`` is a bare name, not a path.
MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

#: Printed by the CLI run path when ``--model`` resolves to nothing fetched.
NO_MODEL_MESSAGE = "no model: run ./scripts/fetch-model first"


class ModelNotFoundError(RuntimeError):
    """``--model`` named nothing :func:`resolve_model` could find locally --
    refused rather than letting a transcriber download one implicitly."""


#: The Hugging Face ``main``-branch commit this tree pins for the default
#: ("small") model (``Systran/faster-whisper-small``), so a later push to
#: that upstream repo cannot silently change the weights a fresh clone
#: downloads. Looked up once via ``GET /api/models/Systran/faster-whisper-
#: small/revision/main`` and documented in the README; bump it deliberately
#: (new lookup, new constant, new :data:`PINNED_SHA256` entry) to move to a
#: newer snapshot.
DEFAULT_MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"


def _download(downloader, size_or_id, dest_dir: Path, local_files_only, revision, pins) -> None:
    kwargs = {"output_dir": str(dest_dir), "local_files_only": local_files_only}
    with warnings.catch_warnings():
        # faster-whisper's download_model always passes `local_dir` (our
        # `output_dir`) to huggingface_hub's snapshot_download, which in turn
        # always sets `local_dir_use_symlinks=False` -- a kwarg recent
        # huggingface_hub versions deprecated and warn about on every call.
        warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")
        if names_revision_param(downloader):
            returned = downloader(size_or_id, revision=revision, **kwargs)
            if revision is not None and revision not in pins:
                require_pin_evidence(dest_dir, returned, revision)
            return
        if revision is not None and revision not in pins:
            raise ModelRevisionError(
                f"the installed downloader takes no `revision` parameter, and no sha256 is "
                f"shipped for revision {revision!r} -- refusing to fetch an unpinned model. "
                "Upgrade faster-whisper, or pin a revision listed in kai_slice/model_pins.py."
            )
        downloader(size_or_id, **kwargs)


def resolve_model(size_or_id: str, models_root: Path | None = None) -> Path:
    """Resolve a CLI ``--model`` value to a local, already-fetched directory
    -- NEVER a bare name a transcriber could resolve by downloading.

    An existing directory is returned as-is (a caller-supplied path, e.g. a
    custom ``--dest``). Otherwise looks for ``<models_root>/<size_or_id>`` --
    where :func:`fetch_model` (``scripts/fetch-model``) writes it -- and
    refuses with :class:`ModelNotFoundError` (:data:`NO_MODEL_MESSAGE`) when
    that is absent or empty. Touches no network itself; it only decides
    whether a transcriber may load a model with no risk of downloading one.
    """
    given = Path(size_or_id)
    if given.is_dir():
        return given
    root = MODELS_ROOT if models_root is None else Path(models_root)
    fetched = root / size_or_id
    if fetched.is_dir() and any(fetched.iterdir()):
        return fetched
    raise ModelNotFoundError(NO_MODEL_MESSAGE)


def _refuse_unpinned(size_or_id: str, revision: str | None) -> None:
    which = f"revision {revision!r}" if revision else "no revision"
    raise ModelRevisionError(
        f"no sha256 is shipped for model {size_or_id!r} ({which}) in kai_slice/model_pins.py "
        "-- refusing an unpinned fetch. Pass --allow-unpinned to fetch it anyway, unverified."
    )


def fetch_model(
    size_or_id: str,
    dest_dir: Path,
    *,
    local_files_only: bool = False,
    downloader: Callable[..., Any] | None = None,
    revision: str | None = None,
    pinned_sha256: dict[str, dict[str, str]] | None = None,
    allow_unpinned: bool = False,
) -> Path:
    """Fetch a CTranslate2 faster-whisper model into ``dest_dir`` once.

    ``downloader`` is the injection seam for tests (default: faster-whisper's
    own ``download_model``, which pulls from the Hugging Face Hub — the real
    network call ``scripts/fetch-model`` is the only caller of);
    ``pinned_sha256`` likewise defaults to the shipped
    :data:`kai_slice.model_pins.PINNED_SHA256`. After this call, ``dest_dir``
    is a local model directory :class:`WhisperTranscriber` loads with no
    further network access.

    ``revision`` defaults to :data:`DEFAULT_MODEL_REVISION` for the default
    ("small") model, whose sha256 values ship in the tree: the fetched files
    must match them (else :class:`ModelIntegrityError`, with the model files
    deleted). A fetch with no shipped sha256 (a non-default model, or a
    revision not in the table) refuses with :class:`ModelRevisionError`
    unless ``allow_unpinned``; even then a named revision must be proven by
    the downloader's own record. Before any download, a non-empty
    ``dest_dir`` that is not already this model's directory refuses with
    :class:`ModelDestError` (see :mod:`kai_slice.model_dest`).
    """
    pins = PINNED_SHA256 if pinned_sha256 is None else pinned_sha256
    if revision is None and size_or_id == DEFAULT_MODEL:
        revision = DEFAULT_MODEL_REVISION
    pinned = revision is not None and revision in pins
    if not pinned and not allow_unpinned:
        _refuse_unpinned(size_or_id, revision)
    pinned_hashes = pins[revision] if revision is not None and pinned else None

    dest_dir = Path(dest_dir)
    require_fetchable_dest(dest_dir, size_or_id, revision, pinned_hashes)
    if downloader is None:
        from faster_whisper.utils import download_model

        downloader = download_model
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    _download(downloader, size_or_id, dest_dir, local_files_only, revision, pins)
    if revision is not None and pinned_hashes is not None:
        verify_pinned_hashes(dest_dir, revision, pinned_hashes)
    else:
        write_unpinned_marker(dest_dir, size_or_id, revision)
    return dest_dir


__all__ = [
    "DEFAULT_MODEL_REVISION",
    "MODELS_ROOT",
    "NO_MODEL_MESSAGE",
    "UNPINNED_MARKER",
    "ModelDestError",
    "ModelIntegrityError",
    "ModelNotFoundError",
    "ModelRevisionError",
    "fetch_model",
    "resolve_model",
]
