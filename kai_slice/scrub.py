"""kai_slice.scrub — strip this machine's absolute paths from a refusal text.

The publishable ``console/build.json`` carries every refusal's exact text, so
it must never carry the host's paths. Only ABSOLUTE, resolved paths are
replaced, and only at a path boundary: the match must be preceded by the
start of the text, whitespace, a quote, ``(``, ``[`` or ``=``, and followed
by ``/``, a quote, whitespace, ``:``, ``,``, ``)``, ``;``, ``]`` or the end
of the text (error messages usually put a path before ``:``). A bare relative string
(``--out o``) is never a replacement key, so ordinary words stay intact.

Besides ``<out>`` and ``<card>``, the repo root and its models root are
always replaced -- a clone outside ``$HOME`` (``/srv``, a Windows drive
mounted under WSL) must not publish its path either.
"""

from __future__ import annotations

import re
from pathlib import Path

#: The public tree's root (the directory holding ``kai_slice/``).
ROOT = Path(__file__).resolve().parent.parent
#: Where ``scripts/fetch-model`` writes models by default.
MODELS_ROOT = ROOT / "models"

_BEFORE = r"(?<![^\s\"'(=\[])"
_AFTER = r"(?=[/\"'\s:,);\]]|\Z)"


def _forms(path: Path) -> set[str]:
    """The absolute spellings of ``path``: as given (when absolute) and
    resolved. A relative spelling is never returned."""
    forms = {str(Path(path).resolve())}
    if Path(path).is_absolute():
        forms.add(str(path))
    return {f for f in forms if f != Path(f).anchor}


def scrub_labels(out: Path, card: Path) -> dict[str, str]:
    """``{absolute path: label}`` for one run."""
    labels: dict[str, str] = {}
    for path, label in (
        (Path.home(), "~"),
        (ROOT, "<root>"),
        (MODELS_ROOT, "<models>"),
        (card, "<card>"),
        (out, "<out>"),
    ):
        labels |= dict.fromkeys(_forms(path), label)
    return labels


def scrub(text: str | None, labels: dict[str, str]) -> str | None:
    """Replace every boundary-delimited occurrence of a label key in one
    pass, longest key first (so ``<out>`` under ``<root>`` wins)."""
    if text is None or not labels:
        return text
    keys = sorted(labels, key=len, reverse=True)
    pattern = re.compile(_BEFORE + "(" + "|".join(map(re.escape, keys)) + ")" + _AFTER)
    return pattern.sub(lambda m: labels[m.group(1)], text)


__all__ = ["MODELS_ROOT", "ROOT", "scrub", "scrub_labels"]
