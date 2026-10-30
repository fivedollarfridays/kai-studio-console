"""The README's top half is accurate for a reader who follows it literally.

* the Quickstart's first command creates the venv and installs the package —
  `./scripts/fetch-model` imports `faster_whisper`, so running it before the
  install crashes;
* `ffmpeg` is named as a requirement before the Quickstart, not only in the
  folded walkthrough;
* the ACTIVITY.md summary names every table that file carries;
* the screenshot names whose machine produced it and says it is a separate
  run from the sample timings block;
* the tree's own hygiene files: `.gitignore` covers `pip install -e .`'s
  `*.egg-info/`, and the CI pins mypy the way it pins ruff.
"""

from __future__ import annotations

import re
from pathlib import Path

_TREE = Path(__file__).resolve().parent.parent
_INSTALL = "python3 -m venv .venv && . .venv/bin/activate && pip install -e ."


def _readme() -> str:
    return (_TREE / "README.md").read_text(encoding="utf-8")


def _top() -> str:
    """Everything before the folded walkthrough."""
    return _readme().split("<details>", 1)[0]


def _quickstart_commands() -> list[str]:
    section = _readme().split("## Quickstart", 1)[1]
    block = section.split("```bash", 1)[1].split("```", 1)[0]
    lines = [line.split("#", 1)[0].strip() for line in block.splitlines()]
    return [line for line in lines if line]


def test_quickstart_first_command_is_the_install():
    assert _quickstart_commands()[0] == _INSTALL


def test_quickstart_installs_before_anything_imports_the_package():
    commands = _quickstart_commands()
    assert commands.index(_INSTALL) < commands.index("./scripts/fetch-model")


def test_ffmpeg_requirement_is_stated_before_the_quickstart():
    before = _readme().split("## Quickstart", 1)[0]
    assert re.search(r"`ffmpeg`.*required|requires? `ffmpeg`", before, re.DOTALL)


def test_activity_summary_names_every_table_activity_md_carries():
    activity = (_TREE / "ACTIVITY.md").read_text(encoding="utf-8")
    top = _top()
    assert "nothing else" not in top
    if "## Python test functions by month" in activity:
        assert "test functions" in top


def test_screenshot_says_whose_machine_and_which_run():
    top = _top()
    assert "on this machine" not in top
    assert "author's" in top
    # docs/console.png is captured from the SAME run as the walkthrough's
    # Measured Performance block, and the README says so
    assert "the same run as that block" in top
    assert "separate run" not in top


def test_industrial_usefulness_row_uses_public_wording():
    row = next(line for line in _top().splitlines() if "Industrial Usefulness" in line)
    assert "professional editor's round-trip" in row


def test_gitignore_covers_egg_info():
    lines = (_TREE / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*.egg-info/" in lines


def test_ci_pins_mypy_like_ruff():
    ci = (_TREE / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r'"ruff==[0-9.]+"', ci)
    assert re.search(r'"mypy==[0-9.]+"', ci)


def test_fetch_model_comment_does_not_claim_more_than_it_can():
    """`pip install -e .` opens connections too, so the fetch-model line may
    only claim to be the one network command AFTER the install."""
    section = _readme().split("## Quickstart", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    line = next(raw for raw in section.splitlines() if raw.startswith("./scripts/fetch-model"))
    comment = line.split("#", 1)[1]
    assert "after the install" in comment, comment


def test_ci_installs_ffmpeg_before_pytest():
    """The `requires_ffmpeg` tests skip when ffmpeg is missing, so CI must
    install it explicitly or the end-to-end path could go green unrun."""
    ci = (_TREE / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = ci.split("\n  pytest:", 1)[1]
    assert "apt-get install -y ffmpeg" in job
    assert job.index("apt-get install -y ffmpeg") < job.index("run: pytest")
