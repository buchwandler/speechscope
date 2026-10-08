"""Atomic CSV/SRT/VTT transcript exports; never infer missing timings."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .errors import ExportError
from .timing import caption_timestamp_error
from .types import VerificationResult, atomic_write


def write_words(result: VerificationResult, path: Path | str) -> None:
    out = io.StringIO()
    keys = [
        "operation",
        "reference_index",
        "reference_text",
        "hypothesis_index",
        "hypothesis_text",
        "start_s",
        "end_s",
        "timing_source",
        "confidence",
    ]
    writer = csv.DictWriter(out, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    if result.reference_text is None:
        for index, word in enumerate(result.transcription.words):
            writer.writerow(
                {
                    "operation": "transcribed",
                    "reference_index": None,
                    "reference_text": None,
                    "hypothesis_index": index,
                    "hypothesis_text": word.text,
                    "start_s": word.start_s,
                    "end_s": word.end_s,
                    "timing_source": result.transcription.timing_source
                    if word.start_s is not None
                    else "unavailable",
                    "confidence": word.confidence,
                }
            )
    else:
        for word in result.aligned_words:
            writer.writerow({key: getattr(word, key) for key in keys})
    atomic_write(Path(path), out.getvalue())


def _timestamp(seconds: float, *, srt: bool) -> str:
    millis = round(seconds * 1000)
    h, rem = divmod(millis, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{',' if srt else '.'}{ms:03d}"


def write_captions(
    result: VerificationResult,
    path: Path | str,
    *,
    kind: str = "srt",
    max_seconds: float = 4.0,
    max_chars: int = 62,
) -> None:
    if kind not in ("srt", "vtt"):
        raise ExportError(f"Unknown caption format {kind}")
    words = result.transcription.words
    error = caption_timestamp_error(words, result.audio_duration_s)
    if error:
        raise ExportError(error)
    captions = []
    current = []
    for word in words:
        if current and (
            word.end_s - current[0].start_s > max_seconds
            or len(" ".join(w.text for w in [*current, word])) > max_chars
        ):
            captions.append(current)
            current = []
        current.append(word)
    if current:
        captions.append(current)
    lines = ["WEBVTT", ""] if kind == "vtt" else []
    for i, group in enumerate(captions, 1):
        if kind == "srt":
            lines.append(str(i))
        lines += [
            (
                f"{_timestamp(group[0].start_s, srt=kind == 'srt')} --> "
                f"{_timestamp(group[-1].end_s, srt=kind == 'srt')}"
            ),
            " ".join(word.text for word in group).strip(),
            "",
        ]
    atomic_write(Path(path), "\n".join(lines) + "\n")
