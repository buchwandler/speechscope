"""Stable value objects and results, independent of backends."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioInput:
    """Mono float64 samples at original sample rate; original WAV path is an optimization."""

    samples: np.ndarray = field(repr=False, compare=False)
    sample_rate: int
    channels: int
    duration_s: float
    path: Path
    transforms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TimedWord:
    text: str
    start_s: float | None
    end_s: float | None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class BackendInfo:
    name: str
    version: str | None
    supports_word_timestamps: bool
    supports_segment_timestamps: bool
    supports_language_hint: bool
    supported_devices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    words: tuple[TimedWord, ...]
    language: str | None
    backend: str
    model: str | None
    duration_s: float
    timing_source: Literal["native_word", "derived_segment", "unavailable"] = "unavailable"
    backend_version: str | None = None
    device: str | None = None


@dataclass(frozen=True, slots=True)
class AlignedWord:
    operation: Literal["match", "substitution", "deletion", "insertion"]
    reference_index: int | None
    reference_text: str | None
    hypothesis_index: int | None
    hypothesis_text: str | None
    start_s: float | None
    end_s: float | None
    timing_source: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class VerificationMetrics:
    reference_word_count: int
    hypothesis_word_count: int
    word_matches: int
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    wer: float
    mer: float
    wil: float
    wip: float
    reference_character_count: int
    character_substitutions: int
    character_deletions: int
    character_insertions: int
    cer: float
    aligned_reference_words_with_timestamps: int
    reference_timestamp_coverage_ratio: float
    hypothesis_timestamp_coverage_ratio: float
    invalid_timestamp_count: int
    non_monotonic_timestamp_count: int
    max_word_end_s: float | None
    trailing_audio_s: float | None
    processing_time_s: float
    realtime_factor: float


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    max_wer: float | None = None
    max_cer: float | None = None
    max_mer: float | None = None
    min_reference_timestamp_coverage: float | None = None
    max_invalid_timestamp_count: int | None = None
    missing_timing: Literal["review", "fail"] = "review"
    schema_version: str = "1"


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    audio: Path | str
    reference_text: str
    backend: str = "redux"
    language: str | None = None
    device: str = "cpu"
    model: str | None = None
    cache_dir: Path | str | None = None
    normalization: str = "basic-v1"
    thresholds: ThresholdPolicy | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    audio: Path
    audio_duration_s: float
    audio_sample_rate: int
    audio_channels: int
    audio_transforms: tuple[str, ...]
    transcription: TranscriptionResult
    reference_text: str | None
    normalized_reference: str | None
    normalized_transcript: str
    normalization: str | None
    aligned_words: tuple[AlignedWord, ...]
    metrics: VerificationMetrics | None
    evaluation: dict
    diagnostics: tuple[dict, ...]
    processing_time_s: float
    realtime_factor: float

    def to_dict(self) -> dict:
        from . import __version__

        result = {
            "schema_version": "1",
            "tool": {"name": "speechscope", "version": __version__},
            "input": {
                "audio_path": str(self.audio),
                "duration_s": self.audio_duration_s,
                "sample_rate_hz": self.audio_sample_rate,
                "channels": self.audio_channels,
            },
            "backend": {
                "name": self.transcription.backend,
                "version": self.transcription.backend_version,
                "model": self.transcription.model,
                "device": self.transcription.device,
            },
            "reference": None
            if self.reference_text is None
            else {
                "raw": self.reference_text,
                "normalized": self.normalized_reference,
                "profile": self.normalization,
            },
            "transcript": {
                "raw": self.transcription.text,
                "normalized": self.normalized_transcript,
                "language": self.transcription.language,
                "timing_source": self.transcription.timing_source,
                "words": [asdict(word) for word in self.transcription.words],
            },
            "alignment": [asdict(word) for word in self.aligned_words],
            "metrics": None if self.metrics is None else asdict(self.metrics),
            "evaluation": self.evaluation,
            "diagnostics": list(self.diagnostics),
            "provenance": {
                "processing_time_s": self.processing_time_s,
                "realtime_factor": self.realtime_factor,
                "audio_transformations": list(self.audio_transforms),
            },
        }
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    def write_json(self, path: Path | str) -> None:
        atomic_write(Path(path), self.to_json())


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix="." + path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as fh:
            name = fh.name
            fh.write(content)
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)
