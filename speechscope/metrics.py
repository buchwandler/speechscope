"""Original, deterministic dynamic-programming text alignment and ASR rates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import AlignmentTooLargeError, InvalidReferenceError

MAX_EDIT_PATH_CELLS = 1_000_000
MAX_CHARACTER_CELLS = 5_000_000


@dataclass(frozen=True)
class Edit:
    tag: str
    ref: int | None
    hyp: int | None


def edit_path(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[Edit, ...]:
    """Optimal edit path; ties prefer match, substitution, deletion, insertion.

    The full backtrace matrix is limited to ``MAX_EDIT_PATH_CELLS`` to prevent
    unbounded memory use on unexpectedly long transcripts.
    """
    n, m = len(reference), len(hypothesis)
    cells = (n + 1) * (m + 1)
    if cells > MAX_EDIT_PATH_CELLS:
        raise AlignmentTooLargeError(
            f"Word alignment needs {cells:,} dynamic-programming cells; "
            f"the limit is {MAX_EDIT_PATH_CELLS:,}. Shorten the reference or transcript."
        )
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


def _character_edit_counts(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[int, int, int]:
    """Count the deterministic optimal path using O(len(hypothesis)) memory."""
    n, m = len(reference), len(hypothesis)
    cells = (n + 1) * (m + 1)
    if cells > MAX_CHARACTER_CELLS:
        raise AlignmentTooLargeError(
            f"Character comparison needs {cells:,} dynamic-programming cells; "
            f"the limit is {MAX_CHARACTER_CELLS:,}. Shorten the reference or transcript."
        )

    previous_cost = list(range(m + 1))
    previous_substitutions = [0] * (m + 1)
    previous_deletions = [0] * (m + 1)
    previous_insertions = list(range(m + 1))
    for i in range(1, n + 1):
        current_cost = [i] + [0] * m
        current_substitutions = [0] * (m + 1)
        current_deletions = [0] * (m + 1)
        current_deletions[0] = i
        current_insertions = [0] * (m + 1)
        for j in range(1, m + 1):
            diagonal = previous_cost[j - 1]
            deletion = previous_cost[j]
            insertion = current_cost[j - 1]
            best = min(
                deletion + 1,
                insertion + 1,
                diagonal + (reference[i - 1] != hypothesis[j - 1]),
            )
            current_cost[j] = best
            if reference[i - 1] == hypothesis[j - 1] and diagonal == best:
                current_substitutions[j] = previous_substitutions[j - 1]
                current_deletions[j] = previous_deletions[j - 1]
                current_insertions[j] = previous_insertions[j - 1]
            elif diagonal + 1 == best:
                current_substitutions[j] = previous_substitutions[j - 1] + 1
                current_deletions[j] = previous_deletions[j - 1]
                current_insertions[j] = previous_insertions[j - 1]
            elif deletion + 1 == best:
                current_substitutions[j] = previous_substitutions[j]
                current_deletions[j] = previous_deletions[j] + 1
                current_insertions[j] = previous_insertions[j]
            else:
                current_substitutions[j] = current_substitutions[j - 1]
                current_deletions[j] = current_deletions[j - 1]
                current_insertions[j] = current_insertions[j - 1] + 1
        previous_cost = current_cost
        previous_substitutions = current_substitutions
        previous_deletions = current_deletions
        previous_insertions = current_insertions
    return (
        previous_substitutions[m],
        previous_deletions[m],
        previous_insertions[m],
    )


def char_rates(reference: str, hypothesis: str) -> dict:
    ref_chars, hyp_chars = reference.replace(" ", ""), hypothesis.replace(" ", "")
    if not ref_chars:
        raise InvalidReferenceError("Reference contains no characters after normalization")
    substitutions, deletions, insertions = _character_edit_counts(ref_chars, hyp_chars)
    return {
        "reference_character_count": len(ref_chars),
        "character_substitutions": substitutions,
        "character_deletions": deletions,
        "character_insertions": insertions,
        "cer": (substitutions + deletions + insertions) / len(ref_chars),
    }
