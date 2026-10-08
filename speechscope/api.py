"""High-level analyze / transcribe workflow with reusable backend session."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from time import perf_counter
from typing import Self

from .audio import load_wav
from .contracts import validate_transcription_result
from .errors import (
    BackendUnavailableError,
    InvalidReferenceError,
    InvalidThresholdPolicyError,
    SpeechScopeError,
    TranscriptionError,
    UnsupportedBackendOptionError,
)
from .evaluation import evaluate, parse_policy
from .metrics import char_rates, edit_path, word_rates
from .normalize import normalize
from .registry import open_backend
from .timing import inspected_words, project_alignment, token_timing_map
from .types import (
    ThresholdPolicy,
    TranscriptionResult,
    VerificationMetrics,
    VerificationRequest,
    VerificationResult,
)


class Verifier:
    """Reusable session; opens one model for any number of WAVs."""

    def __init__(
        self,
        *,
        backend: str = "redux",
        device: str = "cpu",
        model: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.backend, self.device, self.model, self.cache_dir = backend, device, model, cache_dir
        self._backend = None

    def __enter__(self) -> Self:
        self._backend = open_backend(
            self.backend, device=self.device, model=self.model, cache_dir=self.cache_dir
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False

    def close(self) -> None:
        if self._backend is not None:
            backend, self._backend = self._backend, None
            try:
                backend.close()
            except SpeechScopeError:
                raise
            except Exception as exc:
                raise TranscriptionError(f"Backend {self.backend} failed to close: {exc}") from exc

    def _transcribe(self, audio, language: str | None):
        if self._backend is None:
            raise RuntimeError("Use Verifier as a context manager")
        # Backend capability is checked on live instance, so explicit plugins can advertise language.
        try:
            info = self._backend.info()
        except SpeechScopeError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                f"Backend {self.backend} capability inspection failed: {exc}"
            ) from exc
        if language is not None and not info.supports_language_hint:
            raise UnsupportedBackendOptionError(
                f"Backend {self.backend} does not support --language"
            )
        if (
            self.device != "auto"
            and info.supported_devices
            and self.device not in info.supported_devices
        ):
            raise UnsupportedBackendOptionError(
                f"Backend {self.backend} does not support device {self.device}"
            )
        try:
            outcome = self._backend.transcribe(audio, language=language)
        except SpeechScopeError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Backend {self.backend} transcription failed: {exc}") from exc
        outcome = validate_transcription_result(outcome)
        words, warnings, invalid, nonmono = inspected_words(outcome, audio.duration_s)
        normalized = replace(outcome, words=words, duration_s=audio.duration_s)
        if not outcome.text.strip():
            warnings.append({"code": "empty_transcript"})
        return normalized, warnings, invalid, nonmono

    def transcribe(self, audio: Path | str, *, language: str | None = None) -> TranscriptionResult:
        loaded = load_wav(audio)
        return self._transcribe_loaded(loaded, language)

    def _transcribe_loaded(self, loaded, language: str | None) -> TranscriptionResult:
        transcript, _, _, _ = self._transcribe(loaded, language)
        return transcript

    def analyze(self, request: VerificationRequest) -> VerificationResult:
        loaded = load_wav(request.audio)
        normalized_reference = _validate_analysis_inputs(request)
        return self._analyze_loaded(request, loaded, normalized_reference)

    def _analyze_loaded(
        self, request: VerificationRequest, loaded, normalized_reference: str
    ) -> VerificationResult:
        started = perf_counter()
        ref_tokens = normalized_reference.split()
        transcript, diagnostics, invalid, nonmono = self._transcribe(loaded, request.language)
        normalized_transcript = normalize(transcript.text, request.normalization)
        hyp_tokens = normalized_transcript.split()
        edits = edit_path(ref_tokens, hyp_tokens)
        wm = word_rates(edits, len(ref_tokens), len(hyp_tokens))
        cm = char_rates(normalized_reference, normalized_transcript)
        word_map = token_timing_map(transcript.text, transcript.words, request.normalization)
        if transcript.words and not any(w is not None for w in word_map.values()):
            diagnostics.append(
                {
                    "code": "word_tokenization_mismatch",
                    "message": "Cannot safely map ASR word timing to transcript tokens",
                }
            )
        alignment = project_alignment(
            edits, ref_tokens, hyp_tokens, word_map, transcript.timing_source
        )
        timed_ref = sum(
            1
            for a in alignment
            if a.reference_index is not None and a.end_s is not None and a.start_s is not None
        )
        timed_hyp = sum(
            1
            for a in alignment
            if a.hypothesis_index is not None and a.end_s is not None and a.start_s is not None
        )
        ends = [a.end_s for a in alignment if a.end_s is not None]
        max_end = max(ends) if ends else None
        elapsed = perf_counter() - started
        ratio = elapsed / loaded.duration_s
        metrics = VerificationMetrics(
            reference_word_count=len(ref_tokens),
            hypothesis_word_count=len(hyp_tokens),
            **wm,
            **cm,
            aligned_reference_words_with_timestamps=timed_ref,
            reference_timestamp_coverage_ratio=timed_ref / len(ref_tokens),
            hypothesis_timestamp_coverage_ratio=timed_hyp / len(hyp_tokens) if hyp_tokens else 0.0,
            invalid_timestamp_count=invalid,
            non_monotonic_timestamp_count=nonmono,
            max_word_end_s=max_end,
            trailing_audio_s=max(0.0, loaded.duration_s - max_end) if max_end is not None else None,
            processing_time_s=elapsed,
            realtime_factor=ratio,
        )
        outcome = evaluate(metrics, request.thresholds)
        return VerificationResult(
            audio=loaded.path,
            audio_duration_s=loaded.duration_s,
            audio_sample_rate=loaded.sample_rate,
            audio_channels=loaded.channels,
            audio_transforms=loaded.transforms,
            transcription=transcript,
            reference_text=request.reference_text,
            normalized_reference=normalized_reference,
            normalized_transcript=normalized_transcript,
            normalization=request.normalization,
            aligned_words=alignment,
            metrics=metrics,
            evaluation=outcome,
            diagnostics=tuple(diagnostics),
            processing_time_s=elapsed,
            realtime_factor=ratio,
        )


def _validate_analysis_inputs(request: VerificationRequest) -> str:
    if not isinstance(request.reference_text, str):
        raise InvalidReferenceError("Reference text must be a string")
    try:
        normalized_reference = normalize(request.reference_text, request.normalization)
    except TypeError as exc:
        raise InvalidReferenceError("Reference text must be a string") from exc
    if not normalized_reference:
        raise InvalidReferenceError("Reference is empty after normalization")
    if request.thresholds is not None:
        if not isinstance(request.thresholds, ThresholdPolicy):
            raise InvalidThresholdPolicyError("Thresholds must be a ThresholdPolicy")
        values = {
            field.name: getattr(request.thresholds, field.name) for field in fields(ThresholdPolicy)
        }
        parse_policy(values)
    return normalized_reference


def analyze(request: VerificationRequest) -> VerificationResult:
    loaded = load_wav(request.audio)
    normalized_reference = _validate_analysis_inputs(request)
    with Verifier(
        backend=request.backend,
        device=request.device,
        model=request.model,
        cache_dir=request.cache_dir,
    ) as session:
        return session._analyze_loaded(request, loaded, normalized_reference)


def verify(request: VerificationRequest) -> VerificationResult:
    """Compatibility alias: analysis does not necessarily yield a pass/fail status."""
    return analyze(request)


def transcribe(
    audio: Path | str,
    *,
    backend: str = "redux",
    language: str | None = None,
    device: str = "cpu",
    model: str | None = None,
    cache_dir=None,
) -> TranscriptionResult:
    loaded = load_wav(audio)
    with Verifier(backend=backend, device=device, model=model, cache_dir=cache_dir) as session:
        return session._transcribe_loaded(loaded, language)


def transcribe_report(
    audio: Path | str,
    *,
    backend: str = "redux",
    language: str | None = None,
    device: str = "cpu",
    model: str | None = None,
    cache_dir=None,
) -> VerificationResult:
    loaded = load_wav(audio)
    with Verifier(backend=backend, device=device, model=model, cache_dir=cache_dir) as session:
        started = perf_counter()
        transcript, warnings, _, _ = session._transcribe(loaded, language)
        normalized_transcript = normalize(transcript.text)
        elapsed = perf_counter() - started
    return VerificationResult(
        audio=loaded.path,
        audio_duration_s=loaded.duration_s,
        audio_sample_rate=loaded.sample_rate,
        audio_channels=loaded.channels,
        audio_transforms=loaded.transforms,
        transcription=transcript,
        reference_text=None,
        normalized_reference=None,
        normalized_transcript=normalized_transcript,
        normalization=None,
        aligned_words=(),
        metrics=None,
        evaluation={"status": "not_evaluated", "rules": []},
        diagnostics=tuple(warnings),
        processing_time_s=elapsed,
        realtime_factor=elapsed / loaded.duration_s,
    )
