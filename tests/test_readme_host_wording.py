"""The README's `--host` wording matches `serve`'s actual
refusal behaviour. The 2026-10-27 audit found the README claiming "any
`--host`/`--port` works", directly contradicted ten lines later by its own
"refuses any other `--host`" sentence; this cross-checks the surviving claim
against `check_loopback_host` itself rather than another hand-copied string.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kai_slice.serve import LOOPBACK_HOSTS, NonLoopbackHostError, check_loopback_host

_README = (Path(__file__).resolve().parent.parent / "README.md").read_text()


def test_readme_does_not_claim_any_host_works():
    assert "any `--host`" not in _README
    assert "any --host" not in _README


@pytest.mark.parametrize("host", sorted(LOOPBACK_HOSTS))
def test_every_loopback_host_the_readme_names_is_actually_accepted(host):
    assert host in _README
    check_loopback_host(host)  # does not raise


def test_a_host_the_readme_does_not_claim_works_is_actually_refused():
    with pytest.raises(NonLoopbackHostError):
        check_loopback_host("0.0.0.0")
