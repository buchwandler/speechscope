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
    """ASR word timing in seconds from the start of the original audio.

    ``start_s`` and ``end_s`` are nullable estimates; ``confidence`` is an
    optional backend-provided score and is not a SpeechScope correctness value.
    """

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
class BackendAudioInput:
    """Provider-neutral description of the audio representation consumed by ASR."""

    kind: Literal["encoded_wav_path", "generated_wav_path", "audio_input"]
    selection: Literal["encoded_path", "canonicalized_wav", "provider_defined"]
    sample_rate_hz: int
    channels: int
    sample_format: str | None
    transformations: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Transcript plus ASR timing and provenance from one backend call.

    ``timing_source`` distinguishes native word timestamps, conservatively
    derived segment timing, and unavailable timing. Times are seconds.
    """

    text: str
    words: tuple[TimedWord, ...]
    language: str | None
    backend: str
    model: str | None
    duration_s: float
    timing_source: Literal["native_word", "derived_segment", "unavailable"] = "unavailable"
    backend_version: str | None = None
    device: str | None = None

    backend_audio_input: BackendAudioInput | None = None
    diagnostics: tuple[dict[str, object], ...] = ()


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
    """Strict and compound-protocol text metrics plus timing coverage.

    ``wer`` and the ``word_*`` counts use ordinary normalized whitespace
    tokens. ``compound_wer`` additionally accepts bounded adjacent split/join
    boundary variants and is divided by the strict reference token count;
    ``content_match`` means no compound substitution, deletion, or insertion.
    Timestamp coverage describes available ASR timing, not forced alignment.
    """

    reference_word_count: int
    hypothesis_word_count: int
    word_matches: int
    word_substitutions: int
    word_deletions: int
    word_insertions: int
    compound_word_matches: int
    compound_word_substitutions: int
    compound_word_deletions: int
    compound_word_insertions: int
    compound_wer: float
    word_boundary_match_count: int
    content_match: bool
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
    """Optional conjunctive thresholds for text and timestamp metrics.

    ``max_wer`` gates strict whitespace-tokenized WER. Use
    ``max_compound_wer`` when adjacent compound-boundary variants should be
    tolerated; setting both requires both protocols to pass. Missing timing
    follows ``missing_timing`` when a coverage rule is configured.
    """

    max_wer: float | None = None
    max_compound_wer: float | None = None
    max_cer: float | None = None
    max_mer: float | None = None
    min_reference_timestamp_coverage: float | None = None
    max_invalid_timestamp_count: int | None = None
    missing_timing: Literal["review", "fail"] = "review"
    schema_version: str = "1"


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    """Request transcription and reference comparison for one audio file.

    ``audio`` must be a WAV path. ``normalization`` selects a stable versioned
    comparison profile; it does not change the compound matching algorithm.
    """

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
    """Immutable transcription, comparison, evaluation, and provenance report.

    ``reference_text`` and ``metrics`` are ``None`` for transcription-only
    reports. ``to_dict`` and ``to_json`` preserve normalized inputs and emit
    report schema v1; timing values are seconds and remain nullable when unknown.
    """

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

        backend_audio = self.transcription.backend_audio_input
        if backend_audio is None:
            backend_audio_input = {
                "kind": "audio_input",
                "selection": "provider_defined",
                "source_path": str(self.audio),
                "sample_rate_hz": self.audio_sample_rate,
                "channels": self.audio_channels,
                "sample_format": None,
                "loader_sample_transformations": list(self.audio_transforms),
                "transformations_on_backend_input": None,
            }
        else:
            backend_audio_input = {
                "kind": backend_audio.kind,
                "selection": backend_audio.selection,
                "source_path": str(self.audio),
                "sample_rate_hz": backend_audio.sample_rate_hz,
                "channels": backend_audio.channels,
                "sample_format": backend_audio.sample_format,
                "loader_sample_transformations": list(self.audio_transforms),
                "transformations_on_backend_input": (
                    None
                    if backend_audio.transformations is None
                    else list(backend_audio.transformations)
                ),
            }

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
                "backend_audio_input": backend_audio_input,
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
