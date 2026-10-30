"""kai_slice.capture — verified camera-card offload (stage 1).

Re-homed verbatim from the source project (only imports and docstrings changed):
a verified copy (sha256 source vs. destination, never trust a completed
``cp``) of every file on a card into ``<out>/ingest/``, with a manifest
recording exactly what landed. A destination byte that does not hash back to
its source refuses by name and writes nothing further.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copy2

from kai_slice.media import local_path

MANIFEST_NAME = "manifest.json"
_HASH_CHUNK = 1024 * 1024


class CardOffloadError(RuntimeError):
    """The source card path is missing or unsafe (a symlinked entry), or a
    copy fails post-copy verification."""


@dataclass(frozen=True)
class FileRecord:
    name: str
    bytes: int
    sha256: str


def sha256_file(path: Path) -> str:
    """The sha256 hex digest of ``path``'s contents, streamed (no full read)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(
    src_dir: Path, dest_dir: Path, *, only: Sequence[str] | None = None
) -> list[FileRecord]:
    """Verified copy of every file in ``src_dir`` into ``dest_dir``.

    A destination file already present whose hash matches the source is left
    untouched (the idempotent re-run path); a missing or mismatched file is
    (re-)copied and re-verified before being accepted, and a copy that still
    does not verify raises rather than recording a bad hash. A card never
    contains symlinks, so one is a refusal (checked across the whole card,
    from the SAME ``entries`` snapshot the copy loop reads) rather than
    something to follow. ``only`` restricts the copy to named files.
    """
    try:
        entries = sorted(src_dir.iterdir())
    except FileNotFoundError as exc:
        raise CardOffloadError(f"card source vanished before copy: {src_dir}") from exc
    for entry in entries:
        if entry.is_symlink():
            raise CardOffloadError(f"symlinked card entry: {entry.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    records = []
    wanted = None if only is None else set(only)
    for src_file in sorted(p for p in entries if p.is_file()):
        if wanted is not None and src_file.name not in wanted:
            continue
        dest_file = dest_dir / src_file.name
        try:
            src_hash = sha256_file(src_file)
            if not dest_file.exists() or sha256_file(dest_file) != src_hash:
                copy2(src_file, dest_file)
                if sha256_file(dest_file) != src_hash:
                    raise CardOffloadError(f"post-copy verify failed: {src_file.name}")
        except FileNotFoundError as exc:
            raise CardOffloadError(f"card file vanished during copy: {src_file.name}") from exc
        records.append(
            FileRecord(name=src_file.name, bytes=dest_file.stat().st_size, sha256=src_hash)
        )
    return records


def _manifest(src_dir: str, dest_dir: str, records: list[FileRecord]) -> dict:
    return {
        "source": src_dir,
        "dest": dest_dir,
        "count": len(records),
        "bytes": sum(r.bytes for r in records),
        "files": [asdict(r) for r in records],
    }


def capture_stage(card_dir: Path, out_dir: Path) -> dict:
    """Verified copy of ``card_dir`` into ``<out_dir>/ingest/`` + manifest.

    Returns the manifest dict (with ``manifest_path`` and the ingest dir).
    """
    src_dir = Path(local_path(str(card_dir)))
    if not src_dir.is_dir():
        raise CardOffloadError(f"card source not found: {src_dir}")
    ingest_dir = Path(out_dir) / "ingest"
    records = copy_verified(src_dir, ingest_dir)
    manifest = _manifest(str(card_dir), str(ingest_dir), records)
    manifest_path = ingest_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "manifest_path": str(manifest_path), "ingest_dir": str(ingest_dir)}


__all__ = [
    "CardOffloadError",
    "FileRecord",
    "capture_stage",
    "copy_verified",
    "sha256_file",
]
