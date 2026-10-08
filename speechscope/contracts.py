"""Runtime validation for values returned by pluggable transcription backends."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

from .errors import TranscriptionError
from .types import TimedWord, TranscriptionResult

_TIMING_SOURCES = {"native_word", "derived_segment", "unavailable"}


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _timestamp(value: Any) -> Any:
    """Normalize supported numeric scalars; inspection diagnoses other values."""
    value = _python_scalar(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    return value


def _confidence(value: Any, index: int) -> float | None:
    if value is None:
        return None
    value = _python_scalar(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TranscriptionError(f"Backend word {index} confidence must be a real number or null")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise TranscriptionError(f"Backend word {index} confidence must be finite and in [0, 1]")
    return confidence


def _optional_string(value: Any, field: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TranscriptionError(f"Backend result {field} must be a string or null")


def validate_transcription_result(value: Any) -> TranscriptionResult:
    """Validate a backend result and normalize safe scalar values.

    Malformed result/word structures and confidence values fail inference with a
    ``TranscriptionError``. Timestamp coordinates are left for timing inspection,
    which records invalid coordinates and nulls them before any report is emitted.
    """
    if not isinstance(value, TranscriptionResult):
        raise TranscriptionError("Backend must return a TranscriptionResult")
    if not isinstance(value.text, str):
        raise TranscriptionError("Backend result text must be a string")
    if not isinstance(value.backend, str) or not value.backend:
        raise TranscriptionError("Backend result backend must be a nonempty string")
    for field, item in (
        ("language", value.language),
        ("model", value.model),
        ("backend_version", value.backend_version),
        ("device", value.device),
    ):
        _optional_string(item, field)
    if value.timing_source not in _TIMING_SOURCES:
        raise TranscriptionError("Backend result has an unsupported timing_source")

    duration = _python_scalar(value.duration_s)
    if isinstance(duration, bool) or not isinstance(duration, Real):
        raise TranscriptionError("Backend result duration_s must be a finite nonnegative number")
    duration = float(duration)
    if not math.isfinite(duration) or duration < 0:
        raise TranscriptionError("Backend result duration_s must be a finite nonnegative number")

    if not isinstance(value.words, Sequence) or isinstance(value.words, (str, bytes, bytearray)):
        raise TranscriptionError("Backend result words must be a sequence of TimedWord values")
    words = []
    for index, word in enumerate(value.words):
        if not isinstance(word, TimedWord):
            raise TranscriptionError(f"Backend word {index} must be a TimedWord")
        if not isinstance(word.text, str) or not word.text.strip():
            raise TranscriptionError(f"Backend word {index} text must be a nonempty string")
        words.append(
            TimedWord(
                text=word.text,
                start_s=_timestamp(word.start_s),
                end_s=_timestamp(word.end_s),
                confidence=_confidence(word.confidence, index),
            )
        )
    return TranscriptionResult(
        text=value.text,
        words=tuple(words),
        language=value.language,
        backend=value.backend,
        model=value.model,
        duration_s=duration,
        timing_source=value.timing_source,
        backend_version=value.backend_version,
        device=value.device,
    )
