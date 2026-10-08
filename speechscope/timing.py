"""Timing validity and conservative token-to-word projection."""

from __future__ import annotations

import math

from .metrics import Edit
from .normalize import normalize
from .types import AlignedWord, TimedWord, TranscriptionResult


def caption_timestamp_error(words: tuple[TimedWord, ...], duration_s: float) -> str | None:
    """Return why word timestamps cannot be exported as chronological captions.

    Starts must be non-decreasing; overlapping words and equal/zero-duration times
    remain valid. Individual intervals must be finite, ordered, nonnegative, and
    within the source audio (with the same 50 ms tolerance used by inspection).
    """
    if not words:
        return "Cannot export captions: word timestamps are unavailable"
    previous_start = None
    for index, word in enumerate(words):
        start, end = word.start_s, word.end_s
        valid = (
            start is not None
            and end is not None
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in (start, end)
            )
            and start >= 0
            and start <= end
            and end <= duration_s + 0.05
        )
        if not valid:
            return f"Cannot export captions: invalid timestamp for word {index}"
        if previous_start is not None and start < previous_start:
            return f"Cannot export captions: non-monotonic timestamp at word {index}"
        previous_start = start
    return None


def inspected_words(transcript: TranscriptionResult, duration_s: float):
    """Return words with invalid timestamps nulled and structured warnings."""
    cleaned: list[TimedWord] = []
    warnings = []
    invalid = nonmono = 0
    prev_start = None
    prev_end = None
    for i, word in enumerate(transcript.words):
        start, end = word.start_s, word.end_s
        valid = (
            start is not None
            and end is not None
            and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in (start, end)
            )
            and start >= 0
            and start <= end
            and end <= duration_s + 0.05
        )
        if (start is not None or end is not None) and not valid:
            invalid += 1
            warnings.append({"code": "invalid_word_timestamp", "word_index": i})
        if valid and prev_start is not None and start < prev_start:
            nonmono += 1
            warnings.append({"code": "non_monotonic_timestamp", "word_index": i})
        if valid:
            if prev_end is not None:
                if start < prev_end - 0.01:
                    warnings.append(
                        {
                            "code": "word_overlap",
                            "word_index": i,
                            "seconds": round(prev_end - start, 6),
                        }
                    )
                elif start - prev_end > 0.5:
                    warnings.append(
                        {"code": "word_gap", "word_index": i, "seconds": round(start - prev_end, 6)}
                    )
            prev_start = start
            prev_end = end
        cleaned.append(
            TimedWord(
                word.text,
                start if valid else None,
                end if valid else None,
                word.confidence
                if word.confidence is None
                or (isinstance(word.confidence, (int, float)) and math.isfinite(word.confidence))
                else None,
            )
        )
    return tuple(cleaned), warnings, invalid, nonmono


def token_timing_map(
    text: str, words: tuple[TimedWord, ...], profile: str
) -> dict[int, TimedWord | None]:
    """Map only exact 1-to-1 normalized token correspondences; never positional guessing."""
    tokens = normalize(text, profile).split()
    candidates = []
    for word in words:
        normalized = normalize(word.text, profile).split()
        candidates.extend([(part, word if len(normalized) == 1 else None) for part in normalized])
    if [part for part, _ in candidates] != tokens:
        return {i: None for i in range(len(tokens))}
    return {i: word for i, (_, word) in enumerate(candidates)}


def project_alignment(
    edits: tuple[Edit, ...],
    reference: list[str],
    hypothesis: list[str],
    words: dict[int, TimedWord | None],
    timing_source: str,
) -> tuple[AlignedWord, ...]:
    aligned = []
    for edit in edits:
        word = words.get(edit.hyp) if edit.hyp is not None else None
        aligned.append(
            AlignedWord(
                edit.tag,
                edit.ref,
                reference[edit.ref] if edit.ref is not None else None,
                edit.hyp,
                hypothesis[edit.hyp] if edit.hyp is not None else None,
                word.start_s if word else None,
                word.end_s if word else None,
                timing_source
                if word and word.start_s is not None and word.end_s is not None
                else "unavailable",
                word.confidence if word else None,
            )
        )
    return tuple(aligned)
