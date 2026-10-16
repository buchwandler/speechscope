"""Original, deterministic dynamic-programming text alignment and ASR rates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from .errors import AlignmentTooLargeError, InvalidReferenceError

MAX_EDIT_PATH_CELLS = 1_000_000
MAX_CHARACTER_CELLS = 5_000_000

MAX_COMPOUND_PARTS = 3


@dataclass(frozen=True)
class Edit:
    tag: Literal["match", "substitution", "deletion", "insertion"]
    ref: int | None
    hyp: int | None


@dataclass(frozen=True)
class CompoundEdit:
    """Grouped word edit with half-open reference and hypothesis token spans."""

    tag: Literal["match", "boundary_match", "substitution", "deletion", "insertion"]
    ref_start: int
    ref_end: int
    hyp_start: int
    hyp_end: int


class CompoundWordRates(TypedDict):
    compound_word_matches: int
    compound_word_substitutions: int
    compound_word_deletions: int
    compound_word_insertions: int
    compound_wer: float
    word_boundary_match_count: int
    content_match: bool


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


def compound_edit_path(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    *,
    max_parts: int = MAX_COMPOUND_PARTS,
) -> tuple[CompoundEdit, ...]:
    """Align words while allowing bounded adjacent split/join equivalences.

    A boundary match consumes one token on one side and two through
    ``max_parts`` adjacent tokens on the other. The multi-token side must
    concatenate exactly to the single token. Only lexical edit count is
    optimized; ties prefer exact matches, then the boundary match consuming
    the fewest parts, substitution, deletion, and insertion.

    The same cell budget as :func:`edit_path` is checked before allocating the
    dynamic-programming matrix. ``max_parts`` may narrow the v0.1.0 maximum,
    but cannot exceed it.
    """
    if type(max_parts) is not int or not 1 <= max_parts <= MAX_COMPOUND_PARTS:
        raise ValueError(f"max_parts must be an integer from 1 to {MAX_COMPOUND_PARTS}")

    n, m = len(reference), len(hypothesis)
    cells = (n + 1) * (m + 1)
    if cells > MAX_EDIT_PATH_CELLS:
        raise AlignmentTooLargeError(
            f"Word alignment needs {cells:,} dynamic-programming cells; "
            f"the limit is {MAX_EDIT_PATH_CELLS:,}. Shorten the reference or transcript."
        )

    k_limit = min(max_parts, max(n, m))
    reference_joins = [
        {k: "".join(reference[i : i + k]) for k in range(2, min(k_limit, n - i) + 1)}
        for i in range(n)
    ]
    hypothesis_joins = [
        {k: "".join(hypothesis[j : j + k]) for k in range(2, min(k_limit, m - j) + 1)}
        for j in range(m)
    ]

    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
            if reference[i - 1] == hypothesis[j - 1]:
                best = min(best, cost[i - 1][j - 1])
            for parts in range(2, min(k_limit, j) + 1):
                if i and reference[i - 1] == hypothesis_joins[j - parts][parts]:
                    best = min(best, cost[i - 1][j - parts])
            for parts in range(2, min(k_limit, i) + 1):
                if j and reference_joins[i - parts][parts] == hypothesis[j - 1]:
                    best = min(best, cost[i - parts][j - 1])
            cost[i][j] = best

    edits = []
    i, j = n, m
    while i or j:
        current = cost[i][j]
        if i and j and reference[i - 1] == hypothesis[j - 1] and current == cost[i - 1][j - 1]:
            edits.append(CompoundEdit("match", i - 1, i, j - 1, j))
            i -= 1
            j -= 1
            continue

        boundary = []
        for parts in range(2, min(k_limit, j) + 1):
            if (
                i
                and reference[i - 1] == hypothesis_joins[j - parts][parts]
                and current == cost[i - 1][j - parts]
            ):
                boundary.append((parts, "reference_joined", parts))
        for parts in range(2, min(k_limit, i) + 1):
            if (
                j
                and reference_joins[i - parts][parts] == hypothesis[j - 1]
                and current == cost[i - parts][j - 1]
            ):
                boundary.append((parts, "hypothesis_joined", parts))
        if boundary:
            _, direction, parts = min(boundary)
            if direction == "reference_joined":
                edits.append(CompoundEdit("boundary_match", i - 1, i, j - parts, j))
                i -= 1
                j -= parts
            else:
                edits.append(CompoundEdit("boundary_match", i - parts, i, j - 1, j))
                i -= parts
                j -= 1
            continue

        if i and j and current == cost[i - 1][j - 1] + 1:
            edits.append(CompoundEdit("substitution", i - 1, i, j - 1, j))
            i -= 1
            j -= 1
        elif i and current == cost[i - 1][j] + 1:
            edits.append(CompoundEdit("deletion", i - 1, i, j, j))
            i -= 1
        else:
            edits.append(CompoundEdit("insertion", i, i, j - 1, j))
            j -= 1
    return tuple(reversed(edits))


def compound_word_rates(edits: Sequence[CompoundEdit], ref_length: int) -> CompoundWordRates:
    """Count compound-protocol word errors over the strict reference length.

    Boundary matches count all reference tokens in their span as matched;
    ``word_boundary_match_count`` separately counts the explicit variants.
    """
    if not ref_length:
        raise InvalidReferenceError("Reference contains no words after normalization")
    counts = Counter(edit.tag for edit in edits)
    matches = sum(
        edit.ref_end - edit.ref_start for edit in edits if edit.tag in ("match", "boundary_match")
    )
    substitutions = counts["substitution"]
    deletions = counts["deletion"]
    insertions = counts["insertion"]
    return {
        "compound_word_matches": matches,
        "compound_word_substitutions": substitutions,
        "compound_word_deletions": deletions,
        "compound_word_insertions": insertions,
        "compound_wer": (substitutions + deletions + insertions) / ref_length,
        "word_boundary_match_count": counts["boundary_match"],
        "content_match": substitutions == deletions == insertions == 0,
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
