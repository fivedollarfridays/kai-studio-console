"""kai_slice.console — the two console fixture-shaped payloads.

The run writes ``<out>/console/ingest.json`` and ``<out>/console/build.json``
in the console's typed shapes (``IngestResponse`` / ``BuildResponse`` from the
console's ``types.ts``), so the console can show the judge's own run. Both are
validated here against the schemas VENDORED beside this module
(``schemas/{ingest,build}.schema.json``), which this tree's tests pin to
``types.ts`` field for field.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

_CAMERAS = ("A", "B", "C")
MAX_CAMERAS = len(_CAMERAS)
# The four stages this README documents, in run order -- the console shows
# these names verbatim, never the private operator pipeline's vocabulary.
_STAGES = ("capture", "transcribe", "decide", "act")


class TooManyCamerasError(RuntimeError):
    """A card produced more media files than the console's camera model
    (A-C) supports -- a named refusal raised at capture, before any of the
    heavier transcribe/decide/act stages (let alone a render) run."""


def check_camera_count(media_files) -> None:
    count = len(media_files)
    if count > MAX_CAMERAS:
        raise TooManyCamerasError(
            f"kai_slice supports at most {MAX_CAMERAS} cameras (A-{_CAMERAS[-1]}); "
            f"the card produced {count} media files"
        )


def schema_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "schemas" / f"{name}.schema.json"


def _card(index: int, record: dict) -> dict:
    # Capture verifies the card's own bytes only (sha256 source vs. dest);
    # there is no second, independent NAS copy in this pipeline to compare
    # against, and no audio-stream probe of the ingested file -- so audio
    # presence, the NAS hash and the match flag are all genuinely unknown,
    # never hard-coded True or copied from sha256_card.
    gb = round(record["bytes"] / 1e9, 4) if record.get("bytes") else None
    return {
        "camera": _CAMERAS[index],
        "role": None,
        "state": "ingested",
        "take": None,
        "plate_seconds": None,
        "approx_gb": gb,
        "bytes": record.get("bytes"),
        "audio": {"source": "in mp4", "present": None},
        "sha256_card": record["sha256"],
        "sha256_nas": None,
        "sha256_match": None,
        "nas_archive": {"verified": None, "archived_at": None},
    }


def build_ingest(manifest: dict, *, slug: str, verified_at: str) -> dict:
    files = manifest["files"]
    check_camera_count(files)
    return {
        "endpoint": "GET /ingest/{slug}",
        "source": "live",
        "slug": slug,
        "brand": "kai",
        "title": slug,
        "status": "ingested",
        "taped_on": None,
        "taping_at": None,
        "set": None,
        "take": None,
        "ingest_verified_at": verified_at,
        "cards": [_card(i, r) for i, r in enumerate(files)],
    }


def _step(stage: str, rows_by_stage: dict[str, dict]) -> dict:
    row = rows_by_stage.get(stage)
    if row is None:
        return _blank_step(stage)
    ins = row["inputs"][0]["sha256"] if row["inputs"] else None
    outs = row["outputs"][0]["sha256"] if row["outputs"] else None
    return {
        "name": stage,
        "state": "ran" if row["rc"] == 0 else "refused",
        "rc": row["rc"],
        "wall_seconds": row["wall_seconds"],
        "input_sha256": ins,
        "output_sha256": outs,
        "refusal_reason": row.get("error"),
        "refusal_ruling": None,
        "gates": None,
        "output": None,
    }


def _blank_step(name: str) -> dict:
    return {
        "name": name,
        "state": "not run yet",
        "rc": None,
        "wall_seconds": None,
        "input_sha256": None,
        "output_sha256": None,
        "refusal_reason": None,
        "refusal_ruling": None,
        "gates": None,
        "output": None,
    }


def build_build(rows: list[dict], *, slug: str) -> dict:
    """The build payload from run-record rows alone -- it runs no checks of
    its own, so it is safe to call while recording a refusal. Capture is a
    step like any other now, so a capture refusal needs no special-casing:
    it is simply the first step in the refused state."""
    by_stage = {r["stage"]: r for r in rows}
    steps = [_step(s, by_stage) for s in _STAGES]
    refused = next((s for s in steps if s["state"] == "refused"), None)
    done = all(s["state"] == "ran" for s in steps)
    return {
        "endpoint": "GET /build/{slug}",
        "source": "live",
        "slug": slug,
        "brand": "kai",
        "title": slug,
        "status": "refused" if refused else ("done" if done else "running"),
        "stopped_at": refused["name"] if refused else None,
        "steps": steps,
    }


def validate(payload: dict, schema_name: str) -> None:
    schema = json.loads(schema_path(schema_name).read_text())
    jsonschema.validate(payload, schema)


def _write_payload(out_dir: Path, payload: dict, name: str) -> str:
    validate(payload, name)
    console_dir = Path(out_dir) / "console"
    console_dir.mkdir(parents=True, exist_ok=True)
    path = console_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(path)


def write_build(out_dir: Path, build: dict) -> str:
    return _write_payload(out_dir, build, "build")


def write_ingest(out_dir: Path, ingest: dict) -> str:
    return _write_payload(out_dir, ingest, "ingest")


def write_console(out_dir: Path, ingest: dict, build: dict) -> dict:
    validate(ingest, "ingest")
    validate(build, "build")
    return {"ingest": write_ingest(out_dir, ingest), "build": write_build(out_dir, build)}


__all__ = [
    "MAX_CAMERAS",
    "TooManyCamerasError",
    "build_build",
    "build_ingest",
    "check_camera_count",
    "schema_path",
    "validate",
    "write_build",
    "write_console",
    "write_ingest",
]
