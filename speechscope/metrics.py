"""Original, deterministic dynamic-programming text alignment and ASR rates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import InvalidReferenceError


@dataclass(frozen=True)
class Edit:
    tag: str
    ref: int | None
    hyp: int | None


def edit_path(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[Edit, ...]:
    """Optimal edit path; ties prefer match, substitution, deletion, insertion."""
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    result = []
    i, j = n, m
    while i or j:
        if i and j and reference[i - 1] == hypothesis[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            result.append(Edit("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and j and cost[i][j] == cost[i - 1][j - 1] + 1:
            result.append(Edit("substitution", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            result.append(Edit("deletion", i - 1, None))
            i -= 1
        else:
            result.append(Edit("insertion", None, j - 1))
            j -= 1
    return tuple(reversed(result))


def word_rates(edits: Sequence[Edit], ref_length: int, hyp_length: int) -> dict:
    if not ref_length:
        raise InvalidReferenceError("Reference contains no words after normalization")
    counts = Counter(e.tag for e in edits)
    h = counts["match"]
    errors = counts["substitution"] + counts["deletion"] + counts["insertion"]
    preserved = (h / ref_length) * (h / hyp_length) if hyp_length else 0.0
    return {
        "word_matches": h,
        "word_substitutions": counts["substitution"],
        "word_deletions": counts["deletion"],
        "word_insertions": counts["insertion"],
        "wer": errors / ref_length,
        "mer": errors / (h + errors),
        "wip": preserved,
        "wil": 1.0 - preserved,
    }


def char_rates(reference: str, hypothesis: str) -> dict:
    ref_chars, hyp_chars = list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
    if not ref_chars:
        raise InvalidReferenceError("Reference contains no characters after normalization")
    counts = Counter(edit.tag for edit in edit_path(ref_chars, hyp_chars))
    return {
        "reference_character_count": len(ref_chars),
        "character_substitutions": counts["substitution"],
        "character_deletions": counts["deletion"],
        "character_insertions": counts["insertion"],
        "cer": (counts["substitution"] + counts["deletion"] + counts["insertion"]) / len(ref_chars),
    }
