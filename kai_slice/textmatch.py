"""kai_slice.textmatch — word normalisation and in-order (LCS) scoring.

The cut decider matches a script line against a take's transcript by how much
of the line's words appear IN ORDER in the transcript (longest common
subsequence ÷ line length). Words are normalised first: lowercased, with all
punctuation stripped, so "Don't!" and "dont" match. A matched run defines a
TIGHT span — from the first matched transcript word to the last — so the cut
opens and closes on a whole word, never inside one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kai_slice.clips import Clip


def normalise(token: str) -> str:
    """Lowercase ``token`` and strip every non-alphanumeric character."""
    return "".join(c for c in token.lower() if c.isalnum())


def normalise_words(text: str) -> list[str]:
    """``text`` split on whitespace and normalised, dropping empty tokens."""
    return [n for n in (normalise(t) for t in text.split()) if n]


def _lcs_table(a: list[str], b: list[str]) -> list[list[int]]:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def _lcs_length(line: list[str], trans: list[str]) -> int:
    """Length of the longest common subsequence of ``line`` and ``trans``."""
    if not line or not trans:
        return 0
    return _lcs_table(line, trans)[len(line)][len(trans)]


def lcs_match_indices(line: list[str], trans: list[str]) -> list[int]:
    """Indices into ``trans`` of a canonical LCS alignment with ``line``.

    Backtracks preferring diagonal (a match) and, on a tie, the row move, so
    the result is deterministic. Returned indices are strictly increasing.
    """
    if not line or not trans:
        return []
    dp = _lcs_table(line, trans)
    i, j = len(line), len(trans)
    out: list[int] = []
    while i > 0 and j > 0:
        if line[i - 1] == trans[j - 1]:
            out.append(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return out


def score_line_in_clip(line_words: list[str], clip: Clip) -> tuple[float, float, float, list[str]]:
    """Best in-order match of ``line_words`` against ``clip``'s transcript.

    Returns ``(score, span_start, span_end, span_surface_words)`` where score
    is ``lcs ÷ len(line_words)`` and the span opens on the first matched word's
    start and closes on the last matched word's end. An empty line or no match
    scores 0 with a zero-length span.
    """
    if not line_words:
        return 0.0, 0.0, 0.0, []
    trans = [w["norm"] for w in clip.words]
    idx = lcs_match_indices(line_words, trans)
    score = len(idx) / len(line_words)
    if not idx:
        return score, 0.0, 0.0, []
    first, last = idx[0], idx[-1]
    span_words = [w["word"] for w in clip.words[first : last + 1]]
    return score, clip.words[first]["start"], clip.words[last]["end"], span_words


__all__ = [
    "lcs_match_indices",
    "normalise",
    "normalise_words",
    "score_line_in_clip",
]
