"""kai_slice.run — the four-stage pipeline and ``script-check``.

capture -> transcribe -> decide -> act, each appending one row to
``<out>/run-record.jsonl``. A refusal (a corrupt capture byte, a script line
no take matches) records its exact text and stops; no later stage runs and
nothing is rendered.

The fake-transcriber test hook is ``KAI_SLICE_STT_REPLAY_DIR`` (or the
in-process ``transcriber=`` argument): a directory of pre-made
``<stem>.stt.json`` docs shaped like ``transcribe.build_doc``'s own output,
that stand in for a real model. It is a path, not a secret, and the offline
guarantee is unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from kai_slice import console, decide, probe, transcribe
from kai_slice.capture import capture_stage
from kai_slice.clips import MEDIA_SUFFIXES, load_clips
from kai_slice.model import resolve_model
from kai_slice.record import RunRecord
from kai_slice.scrub import scrub, scrub_labels

REPLAY_ENV = "KAI_SLICE_STT_REPLAY_DIR"


def _media(ingest_dir: Path) -> list[Path]:
    return sorted(p for p in ingest_dir.iterdir() if p.suffix.lower() in MEDIA_SUFFIXES)


def _local_model(model, transcriber) -> str | None:
    """The fetched model's local path, or a :class:`ModelNotFoundError`.

    A bare ``--model`` name must never reach WhisperTranscriber, which would
    hand it to faster-whisper's own downloader. ``None`` for replay (no model
    needed) and an injected transcriber (the caller already decided what runs
    it). Callers resolve this BEFORE capture, so a run that cannot transcribe
    never copies the card first.
    """
    if os.environ.get(REPLAY_ENV) or transcriber:
        return None
    return str(resolve_model(model))


def _transcribe(
    ingest_dir, transcripts_dir, *, model, device, transcriber, local_model=None
) -> list[Path]:
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    replay = os.environ.get(REPLAY_ENV)
    if local_model is None:
        local_model = _local_model(model, transcriber)
    # Built ONCE for the whole run (on first use, not eagerly -- a card with
    # no media never touches the model at all): WhisperTranscriber loads the
    # model itself, so a 3-camera card would otherwise load it 3 times.
    built = transcriber
    out_paths = []
    for media in _media(ingest_dir):
        if replay:
            doc = transcribe.load_doc(Path(replay) / f"{media.stem}.stt.json")
            doc["file"] = str(media)
        else:
            if built is None:
                built = transcribe.WhisperTranscriber(model=local_model, device=device)
            doc = transcribe.transcribe_take(
                media, duration=probe.duration(media), transcriber=built, model=model
            )
        out_paths.append(transcribe.save_doc(doc, transcribe.sidecar_path(transcripts_dir, media)))
    return out_paths


def _takes(ingest_dir: Path) -> dict | None:
    takes = ingest_dir / "takes.json"
    return json.loads(takes.read_text()) if takes.is_file() else None


def _record_cut_bounds(spans: list[dict], ingest_dir: Path) -> None:
    """Write act's frame-snapped bounds onto each span as ``cut_start`` /
    ``cut_end`` -- the same function act renders with, so decision.json says
    exactly what is in the cut. No spans: act refuses them, by name."""
    if not spans:
        return
    from kai_slice.act import rendered_bounds

    name_to_path = {s["clip"]: str(ingest_dir / s["clip"]) for s in spans}
    for span, cut in zip(spans, rendered_bounds(spans, name_to_path), strict=True):
        span["cut_start"], span["cut_end"] = cut["start"], cut["end"]


def _decide(ingest_dir, transcripts_dir, out, *, script, cue) -> dict:
    clips = load_clips(ingest_dir, transcripts_dir, takes=_takes(ingest_dir))
    if script is not None:
        decision = decide.decide_script(clips, Path(script).read_text())
    else:
        decision = decide.decide_cue(clips, list(cue or []))
    _record_cut_bounds(decision["spans"], ingest_dir)
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def _ingest(manifest: dict, slug: str, verified_at: str) -> dict:
    media = [r for r in manifest["files"] if Path(r["name"]).suffix.lower() in MEDIA_SUFFIXES]
    return console.build_ingest({**manifest, "files": media}, slug=slug, verified_at=verified_at)


def _console_rows(record: RunRecord, out: Path, card: Path) -> list[dict]:
    labels = scrub_labels(out, card)
    return [{**r, "error": scrub(r.get("error"), labels)} for r in record.rows()]


def _write_console(out, card, manifest, record, slug, verified_at) -> None:
    rows = _console_rows(record, out, card)
    ingest = _ingest(manifest, slug, verified_at)
    console.write_console(out, ingest, console.build_build(rows, slug=slug))


def _record_refusal(out, card, manifest, record, slug, verified_at) -> None:
    """Write the console's view of a refused run from what is already known
    (the run-record rows), re-running no checks. NEVER raises: the caller
    re-raises the original refusal, which a record failure must not mask."""
    try:
        rows = _console_rows(record, out, card)
        console.write_build(out, console.build_build(rows, slug=slug))
    except Exception as exc:  # noqa: BLE001 -- the original refusal wins
        print(f"kai_slice: console/build.json not written: {exc}", file=sys.stderr)
        return
    if manifest is None or not any(r["stage"] == "capture" and r["rc"] == 0 for r in rows):
        return  # capture refused: there is no valid ingest to describe
    try:
        console.write_ingest(out, _ingest(manifest, slug, verified_at))
    except Exception as exc:  # noqa: BLE001 -- the original refusal wins
        print(f"kai_slice: console/ingest.json not written: {exc}", file=sys.stderr)


def run_pipeline(
    card_dir,
    out_dir,
    *,
    script=None,
    cue=None,
    model="small",
    device="cpu",
    transcriber=None,
    slug=None,
) -> dict:
    """Run all four stages; return the decision and the artifact paths."""
    local_model = _local_model(model, transcriber)  # refuse before capture copies the card
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = RunRecord(out / "run-record.jsonl")
    slug = slug or out.name or "kai-slice-run"
    verified_at = datetime.now(UTC).isoformat()
    card = Path(card_dir)
    manifest: dict | None = None

    try:
        with record.stage("capture", [card]) as box:
            manifest = capture_stage(card, out)
            console.check_camera_count(_media(Path(manifest["ingest_dir"])))
            box["outputs"] = [Path(manifest["manifest_path"])]
        ingest_dir = Path(manifest["ingest_dir"])
        transcripts_dir = out / "transcripts"

        with record.stage("transcribe", _media(ingest_dir)) as box:
            box["outputs"] = _transcribe(
                ingest_dir,
                transcripts_dir,
                model=model,
                device=device,
                transcriber=transcriber,
                local_model=local_model,
            )

        with record.stage("decide", sorted(transcripts_dir.glob("*.stt.json"))) as box:
            decision = _decide(ingest_dir, transcripts_dir, out, script=script, cue=cue)
            box["outputs"] = [out / "decision.json"]

        with record.stage("act", [out / "decision.json"]) as box:
            name_to_path = {s["clip"]: str(ingest_dir / s["clip"]) for s in decision["spans"]}
            from kai_slice.act import act_stage

            rendered = act_stage(decision["spans"], name_to_path, out)
            box["outputs"] = [Path(rendered["cut"]), Path(rendered["edit"])]
    except Exception:
        _record_refusal(out, card, manifest, record, slug, verified_at)
        raise

    _write_console(out, card, manifest, record, slug, verified_at)
    return {"decision": decision, "manifest": manifest, "out": str(out), **rendered}


def run_script_check(card_dir, *, script, out_dir, model="small", device="cpu", transcriber=None):
    """Capture + transcribe, then print every line's best score/clip/words.

    Renders nothing and writes no decision or cut -- a read-only report so a
    take that went off-script can be fixed by updating the script to what was
    actually said, not by re-shooting it.
    """
    local_model = _local_model(model, transcriber)  # refuse before capture copies the card
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = capture_stage(card_dir, out)
    ingest_dir = Path(manifest["ingest_dir"])
    transcripts_dir = out / "transcripts"
    _transcribe(
        ingest_dir,
        transcripts_dir,
        model=model,
        device=device,
        transcriber=transcriber,
        local_model=local_model,
    )
    clips = load_clips(ingest_dir, transcripts_dir, takes=_takes(ingest_dir))
    return decide.script_check(clips, Path(script).read_text())


__all__ = ["run_pipeline", "run_script_check"]
