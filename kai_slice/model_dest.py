"""kai_slice.model_dest — never download a model into a directory holding
the user's own files.

Checked by :func:`kai_slice.model.fetch_model` BEFORE the downloader runs.
``dest`` may be absent, empty, or already a model directory for THIS model:

- a pinned revision: every top-level entry is a file name the shipped
  sha256 table lists for it, or the downloader's own ``.cache`` metadata;
- an unpinned fetch (``--allow-unpinned``): ``dest`` carries
  :data:`UNPINNED_MARKER`, written by a previous fetch, naming this model
  and revision.

Anything else refuses by name, touching nothing.
"""

from __future__ import annotations

from pathlib import Path

#: Written into ``dest`` after an unpinned fetch, so a re-fetch of the same
#: model can recognise its own directory (a pinned one is recognised by its
#: file names; this file is a dot-file, so it is never hashed).
UNPINNED_MARKER = ".kai-unpinned-model"
_DOWNLOADER_METADATA = ".cache"


class ModelDestError(RuntimeError):
    """``dest`` is not safe to download into: a file, or a non-empty
    directory that is not already a model directory for this model."""


def _marker_text(size_or_id: str, revision: str | None) -> str:
    return f"{size_or_id}\n{revision or ''}\n"


def write_unpinned_marker(dest: Path, size_or_id: str, revision: str | None) -> None:
    (Path(dest) / UNPINNED_MARKER).write_text(_marker_text(size_or_id, revision))


def _marked_for(dest: Path, size_or_id: str, revision: str | None) -> bool:
    marker = dest / UNPINNED_MARKER
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        return marker.read_text() == _marker_text(size_or_id, revision)
    except OSError:
        return False


def require_fetchable_dest(dest: Path, size_or_id: str, revision: str | None, pinned_names) -> None:
    """Refuse a ``dest`` holding anything but this model's own files.

    ``pinned_names`` is the shipped file list for ``revision`` (``None`` when
    the fetch is unpinned)."""
    dest = Path(dest)
    if not dest.exists() and not dest.is_symlink():
        return
    if not dest.is_dir():
        raise ModelDestError(f"--dest {dest} is not a directory -- refusing to download there")
    entries = sorted(p.name for p in dest.iterdir())
    if not entries:
        return
    if pinned_names is None:
        if _marked_for(dest, size_or_id, revision):
            return
        foreign = entries
    else:
        allowed = {Path(n).parts[0] for n in pinned_names} | {_DOWNLOADER_METADATA}
        foreign = [e for e in entries if e not in allowed]
        if not foreign:
            return
    raise ModelDestError(
        f"--dest {dest} is not empty and is not a model directory for {size_or_id!r} "
        f"(it holds {foreign}) -- refusing to download into it. "
        "Pick an empty or non-existent --dest."
    )


__all__ = ["UNPINNED_MARKER", "ModelDestError", "require_fetchable_dest", "write_unpinned_marker"]
