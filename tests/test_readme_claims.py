"""The README's dependency claim matches
``pyproject.toml``: it may say "pins" only if every dependency is an exact
``==`` pin; with ranges (the truth today) it says "bounds"."""

from __future__ import annotations

import tomllib
from pathlib import Path

_TREE = Path(__file__).resolve().parent.parent


def _deps() -> list[str]:
    data = tomllib.loads((_TREE / "pyproject.toml").read_text())
    return data["project"]["dependencies"]


def test_the_readme_dependency_claim_matches_pyproject():
    readme = (_TREE / "README.md").read_text()
    exact = all("==" in dep for dep in _deps())
    if not exact:
        assert "pins dependencies" not in readme
        assert "bounds dependencies" in readme
    else:
        assert "pins dependencies" in readme
