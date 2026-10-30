"""kai_slice.record — the per-run ledger (``run-record.jsonl``).

Each stage appends exactly one row: its wall time, inputs and outputs by
sha256, and return code. A failure or refusal records its exact text and the
run stops — the ledger always reflects where a run got to.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from kai_slice.capture import sha256_file


def _digest_list(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        p = Path(p)
        out.append({"path": str(p), "sha256": sha256_file(p) if p.is_file() else None})
    return out


class RunRecord:
    """Append-only JSONL ledger for one run."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    @contextmanager
    def stage(self, name: str, inputs: list[Path]):
        """Time ``name``; on success write rc 0, on failure rc 1 + exact text.

        Yields a mutable dict; set ``outputs`` (a list of paths) on it before
        the block exits and they are recorded by sha256.
        """
        started = time.monotonic()
        box: dict = {"outputs": []}
        try:
            yield box
        except Exception as exc:
            self._append(
                {
                    "stage": name,
                    "wall_seconds": round(time.monotonic() - started, 6),
                    "rc": 1,
                    "inputs": _digest_list(inputs),
                    "outputs": [],
                    "error": str(exc),
                }
            )
            raise
        self._append(
            {
                "stage": name,
                "wall_seconds": round(time.monotonic() - started, 6),
                "rc": 0,
                "inputs": _digest_list(inputs),
                "outputs": _digest_list(box["outputs"]),
                "error": None,
            }
        )

    def rows(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def last_failure_message(self, since: int = 0) -> str | None:
        """``step '<stage>' failed: <error>`` for the last ``rc=1`` row at or
        after row index ``since``, or ``None`` if there is none -- a caller's
        fallback for a crash :meth:`stage` never got to record (e.g. before
        the first stage even starts). The ledger is append-only and an
        ``--out`` may be reused, so a caller passes ``since=len(rows())``
        taken before its own run, never reading an earlier run's failure."""
        failed = [r for r in self.rows()[since:] if r.get("rc") == 1]
        if not failed:
            return None
        row = failed[-1]
        return f"step {row['stage']!r} failed: {row['error']}"


__all__ = ["RunRecord"]
