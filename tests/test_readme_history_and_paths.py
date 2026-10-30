"""The README's history and path claims stay honest.

Two regressions the 2026-10-27 security/code-review audits found in the
exported README, now guarded structurally: the "no git history" claim
(false once this repo carries its own export-snapshot commits --
ACTIVITY.md links the development history those snapshots came
from) and a path to a file absent from the export (`scripts/export_public.py`
is the source project's gate, never shipped here). The third check
is general on purpose: ANY backtick path-shaped token the README names must
resolve to a real file or directory in this tree, so a stale or renamed
reference fails loudly instead of silently going wrong, the way the first
two did.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_README = (_ROOT / "README.md").read_text()

# A backtick token shaped like a relative path: at least one `/`, no
# placeholder (`<out>`), no API route or CLI flag (`/jobs`, `--host`), no
# whitespace -- see test_path_token_regex_ignores_non_path_tokens below for
# the shapes this must NOT match.
_PATH_TOKEN = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_.\-]+)+/?)`")

# Only these first segments are unambiguously repo-relative in this README --
# "console/" also names a per-run `--out` subdirectory (`console/build.json`,
# `<out>/console/{ingest,build}.json`), so a bare `console/...` token is run-
# relative prose, not a claim about this tree, and is not checked here.
_UNAMBIGUOUS_ROOTS = frozenset({"kai_slice", "scripts", "tests", "samples", "docs"})


def _referenced_paths() -> list[str]:
    return [p for p in _PATH_TOKEN.findall(_README) if p.split("/", 1)[0] in _UNAMBIGUOUS_ROOTS]


def test_no_git_history_claim_is_absent():
    assert "no git history" not in _README


def test_export_public_script_is_not_claimed_to_live_in_this_tree():
    assert "scripts/export_public.py" not in _README


def test_every_referenced_path_exists_in_this_tree():
    paths = _referenced_paths()
    assert paths, "the path-token regex matched nothing -- it is probably broken"
    missing = [p for p in paths if not (_ROOT / p.removeprefix("./")).exists()]
    assert not missing, f"README references paths absent from this tree: {missing}"


def test_path_token_regex_ignores_non_path_tokens():
    sample = (
        "`<out>/console/build.json` `--host` `/jobs` `127.0.0.1` `Host` "
        "`*.json` `..` `scripts/export_public.py`"
    )
    assert _PATH_TOKEN.findall(sample) == ["scripts/export_public.py"]


def test_the_path_check_would_have_caught_the_original_bug():
    """seed-13 (guards-must-fail-for-their-own-reason): a README naming a
    file truly absent from this tree must fail the check above, not just
    pass because nothing was ever wrong with the regex."""
    planted = _README + "\nSee `scripts/export_public.py` for the gate.\n"
    paths = [p for p in _PATH_TOKEN.findall(planted) if p.split("/", 1)[0] in _UNAMBIGUOUS_ROOTS]
    missing = [p for p in paths if not (_ROOT / p.removeprefix("./")).exists()]
    assert missing == ["scripts/export_public.py"]
