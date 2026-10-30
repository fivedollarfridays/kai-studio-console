"""kai_slice.cli — ``python -m kai_slice {run,script-check,serve}``.

A refusal (a corrupt capture byte, a script line no take matches) prints its
exact text to stderr and exits non-zero; the run-record ledger already holds
the same text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from av.error import FFmpegError

from kai_slice.capture import CardOffloadError
from kai_slice.console import TooManyCamerasError
from kai_slice.decide import DecideError
from kai_slice.model import ModelNotFoundError
from kai_slice.record import RunRecord
from kai_slice.run import run_pipeline, run_script_check
from kai_slice.serve import DEFAULT_PORT, NonLoopbackHostError
from kai_slice.serve import serve as serve_console


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kai_slice", description="capture -> transcribe -> decide -> act"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run all four stages")
    run.add_argument("card_dir", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--cue", action="append", default=[], metavar="WORD")
    run.add_argument("--script", type=Path, default=None)
    run.add_argument("--model", default="small")
    run.add_argument("--device", default="cpu")
    run.add_argument("--slug", default=None)

    check = sub.add_parser("script-check", help="print per-line best scores; render nothing")
    check.add_argument("card_dir", type=Path)
    check.add_argument("--script", type=Path, required=True)
    check.add_argument("--out", type=Path, default=None)
    check.add_argument("--model", default="small")
    check.add_argument("--device", default="cpu")

    serve = sub.add_parser("serve", help="serve <out>/console/ read-only on 127.0.0.1")
    serve.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=None,
        help="a run's --out directory, whose console/*.json is served alongside the UI; "
        "omit for replay-only (UI build with no run data)",
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="loopback name/address to bind: 127.0.0.1, localhost, or ::1 (default: 127.0.0.1)",
    )
    serve.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"TCP port to bind (default: {DEFAULT_PORT})"
    )
    return p


def _run(args) -> int:
    result = run_pipeline(
        args.card_dir,
        args.out,
        script=args.script,
        cue=args.cue,
        model=args.model,
        device=args.device,
        slug=args.slug,
    )
    print(
        json.dumps(
            {"out": result["out"], "spans": len(result["decision"]["spans"]), "cut": result["cut"]}
        )
    )
    return 0


def _script_check(args) -> int:
    out = args.out or Path(tempfile.mkdtemp(prefix="kai_slice-check-"))
    rows = run_script_check(
        args.card_dir, script=args.script, out_dir=out, model=args.model, device=args.device
    )
    had_error = False
    for r in rows:
        if r.get("error"):
            had_error = True
            print(f"line {r['line']} [{r['speaker']}]: REFUSED {r['error']}")
        else:
            print(
                f"line {r['line']} [{r['speaker']}]: best {r['best_score']:.2f} in {r['best_clip']}"
                f" -> {' '.join(r['transcript_words'])}"
            )
    return 1 if had_error else 0


def _serve(args) -> int:
    try:
        serve_console(args.out_dir, host=args.host, port=args.port)
    except NonLoopbackHostError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _ledger(args) -> RunRecord | None:
    out = getattr(args, "out", None)
    return RunRecord(out / "run-record.jsonl") if out else None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ledger = _ledger(args)
    # Rows already in a reused --out belong to an earlier run: only rows
    # this invocation appends may name its failure.
    start = len(ledger.rows()) if ledger else 0
    try:
        if args.cmd == "run":
            return _run(args)
        if args.cmd == "script-check":
            return _script_check(args)
        return _serve(args)
    except (
        CardOffloadError,
        DecideError,
        ModelNotFoundError,
        TooManyCamerasError,
        subprocess.CalledProcessError,
        FFmpegError,
    ) as exc:
        message = ledger.last_failure_message(since=start) if ledger else None
        print(message or str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
