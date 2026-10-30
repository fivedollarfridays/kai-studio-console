"""kai_slice.model_pins — the expected sha256 of every file in each pinned
model revision, shipped in the tree so a fetch is checked against KNOWN
values rather than against whatever it downloaded first.

Provenance of the ``Systran/faster-whisper-small`` entry: hashed from that
revision's snapshot in a local Hugging Face cache, and cross-checked against
the identifiers the Hub itself assigns -- ``model.bin``'s value IS its Git LFS
object id (the cache's blob name), and each small file's git blob sha1
matched its blob name. To pin a new revision, add its table here in the same
change that bumps :data:`kai_slice.model.DEFAULT_MODEL_REVISION`.
"""

from __future__ import annotations

#: ``{revision: {relative file name: sha256 hex}}``.
PINNED_SHA256: dict[str, dict[str, str]] = {
    "536b0662742c02347bc0e980a01041f333bce120": {
        "config.json": "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828",
        "model.bin": "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
        "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
        "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
    },
}

__all__ = ["PINNED_SHA256"]
