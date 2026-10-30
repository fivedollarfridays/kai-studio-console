"""kai_slice.model_verify — prove a fetched model IS the pinned revision.

Two independent proofs, used by :func:`kai_slice.model.fetch_model`:

- **content**: every file matches the sha256 shipped in
  :data:`kai_slice.model_pins.PINNED_SHA256` for that revision (the strong
  one -- it holds whatever the downloader did);
- **provenance**, only when no hashes are shipped for the revision: the
  downloader's own record names the pinned commit -- the Hugging Face
  snapshot path it returned, or the local-dir download metadata it wrote.

Neither ever trusts what was downloaded first. Hashing is streamed, and skips
dot-directories (the downloader's ``.cache/`` metadata holds etags and
timestamps that legitimately change between fetches and are never loaded).
"""

from __future__ import annotations

import inspect
from pathlib import Path

from kai_slice.capture import sha256_file

#: Where ``huggingface_hub`` records, per file, the commit a local-dir
#: download came from (first line of ``<file>.metadata``).
_METADATA_DIR = Path(".cache") / "huggingface" / "download"
_NAMED = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class ModelIntegrityError(RuntimeError):
    """The fetched files do not match the sha256 shipped for the pinned
    revision. The model files were deleted, so nothing loadable is left."""


class ModelRevisionError(RuntimeError):
    """The pinned revision could not be enforced or verified -- refused
    rather than silently using whatever upstream serves today."""


def names_revision_param(downloader) -> bool:
    """True only when the signature names ``revision`` explicitly: a
    ``**kwargs`` catch-all or an un-introspectable callable may ignore it."""
    try:
        params = inspect.signature(downloader).parameters
    except (TypeError, ValueError):
        return False
    param = params.get("revision")
    return param is not None and param.kind in _NAMED


def file_sha256s(dest_dir: Path) -> dict[str, str]:
    """``{relative posix name: sha256}`` for every non-dot file, streamed."""
    out = {}
    for path in sorted(Path(dest_dir).rglob("*")):
        rel = path.relative_to(dest_dir)
        if not any(part.startswith(".") for part in rel.parts) and path.is_file():
            out[rel.as_posix()] = sha256_file(path)
    return out


def _delete_model_files(dest_dir: Path, names) -> list[str]:
    deleted = []
    for name in sorted(names):
        path = Path(dest_dir) / name
        if path.is_file() or path.is_symlink():
            path.unlink()
            deleted.append(name)
    return deleted


def verify_pinned_hashes(dest_dir: Path, revision: str, expected: dict[str, str]) -> None:
    """Refuse unless ``dest_dir`` holds exactly ``expected``; on refusal,
    delete every fetched non-dot file so a tampered or partial model cannot
    be loaded. Precondition: ``dest_dir`` held no user files before the
    download (:func:`kai_slice.model_dest.require_fetchable_dest`)."""
    actual = file_sha256s(dest_dir)
    bad = sorted(name for name, digest in expected.items() if actual.get(name) != digest)
    extra = sorted(set(actual) - set(expected))
    if not bad and not extra:
        return
    # fetch_model proved dest empty or model-only BEFORE the download
    # (kai_slice.model_dest), so every non-dot file here -- the unexpected
    # ones included -- is the downloader's, never the user's.
    deleted = _delete_model_files(dest_dir, set(expected) | set(extra))
    raise ModelIntegrityError(
        f"model files under {dest_dir} do not match the sha256 pinned for revision "
        f"{revision!r} (mismatched or missing: {bad}; unexpected: {extra}). Deleted "
        f"{deleted} so nothing loadable is left; re-run ./scripts/fetch-model."
    )


def _metadata_commits(dest_dir: Path) -> set[str]:
    meta_dir = Path(dest_dir) / _METADATA_DIR
    if not meta_dir.is_dir():
        return set()
    return {p.read_text().split("\n", 1)[0].strip() for p in meta_dir.rglob("*.metadata")}


def require_pin_evidence(dest_dir: Path, returned, revision: str) -> None:
    """Refuse unless the downloader's own record names ``revision``."""
    # A Hugging Face snapshot dir is ``.../snapshots/<commit>``: it names the
    # commit only when it is the very directory the model was fetched into.
    snapshot = Path(returned).resolve() if returned is not None else None
    in_place = snapshot is not None and snapshot == Path(dest_dir).resolve()
    if (
        in_place
        and snapshot is not None
        and snapshot.parent.name == "snapshots"
        and snapshot.name == revision
    ):
        return
    commits = _metadata_commits(dest_dir)
    if commits == {revision}:
        return
    seen = f"its download metadata names {sorted(commits)}" if commits else "nothing names it"
    raise ModelRevisionError(
        f"could not verify the downloader fetched revision {revision!r} into {dest_dir} "
        f"({seen}) -- refusing an unverified pin"
    )


__all__ = [
    "ModelIntegrityError",
    "ModelRevisionError",
    "file_sha256s",
    "names_revision_param",
    "require_pin_evidence",
    "verify_pinned_hashes",
]
