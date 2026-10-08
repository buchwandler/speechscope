"""Lazy Photon/Parakeet Redux adapter. No network or model access at import/inspect."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from contextlib import suppress

from ..errors import BackendUnavailableError, TranscriptionError, UnsupportedBackendOptionError
from ..types import AudioInput, BackendInfo, TimedWord, TranscriptionResult

DEFAULT_MODEL = "moondream/parakeet-redux"


def _get(item: object, key: str, default=None):
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def info() -> BackendInfo:
    try:
        version = importlib.metadata.version("moondream")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return BackendInfo(
        name="redux",
        version=version,
        supports_word_timestamps=True,
        supports_segment_timestamps=True,
        supports_language_hint=False,
        supported_devices=("cpu", "cuda", "mps", "auto"),
    )


class ReduxTranscriber:
    def __init__(self, *, device: str = "cpu", model: str | None = None, cache_dir=None):
        if model is not None and model != DEFAULT_MODEL:
            raise UnsupportedBackendOptionError(
                f"Redux supports only model {DEFAULT_MODEL!r}, not {model!r}"
            )
        if device not in ("cpu", "cuda", "mps", "auto"):
            raise UnsupportedBackendOptionError(f"Redux unsupported device {device!r}")
        if cache_dir is not None:
            raise UnsupportedBackendOptionError(
                "Redux does not expose a cache-dir option via Photon; configure its cache externally"
            )
        try:
            import moondream
        except ImportError as exc:
            raise BackendUnavailableError(
                "Install Redux with: pip install 'speechscope[redux]'"
            ) from exc
        self._device = device
        self._model = DEFAULT_MODEL if model is None else model
        manager = None
        try:
            if device == "auto":
                manager = moondream.photon(self._model)
            else:
                manager = moondream.photon(self._model, device=device)
            self._manager = manager
            self._speech = manager.__enter__()
        except Exception as exc:
            if manager is not None:
                # Preserve initialization failure if best-effort cleanup also fails.
                with suppress(Exception):
                    manager.__exit__(type(exc), exc, exc.__traceback__)
            self._manager = None
            raise BackendUnavailableError(
                f"Could not initialize Redux on device {device}: {exc}"
            ) from exc

    def info(self) -> BackendInfo:
        return info()

    def transcribe(self, audio: AudioInput, *, language: str | None = None) -> TranscriptionResult:
        if language is not None:
            raise UnsupportedBackendOptionError(
                "Redux detects language automatically; --language is unsupported"
            )
        try:
            raw = self._speech.transcribe(audio=audio.path, timestamps="word")
            if not isinstance(_get(raw, "text", ""), str):
                raise TypeError("Transcript text must be a string")
            text = _get(raw, "text", "")
            words = []
            segments = _get(raw, "segments", []) or []
            for segment in segments:
                for word in _get(segment, "words", []) or []:
                    token = _get(word, "word", _get(word, "text", None))
                    if not isinstance(token, str) or not token.strip():
                        raise TypeError("Invalid word text returned by Redux")
                    start, end = _get(word, "start"), _get(word, "end")
                    # Preserve malformed coordinates for post-inference diagnostics when numeric.
                    words.append(
                        TimedWord(
                            token,
                            start,
                            end,
                            _get(word, "probability", None),
                        )
                    )
            if not words:
                for word in _get(raw, "words", []) or []:
                    token = _get(word, "word", _get(word, "text", None))
                    if not isinstance(token, str):
                        raise TypeError("Invalid word text returned by Redux")
                    start, end = _get(word, "start"), _get(word, "end")
                    words.append(
                        TimedWord(
                            token,
                            start,
                            end,
                        )
                    )
            return TranscriptionResult(
                text=text,
                words=tuple(words),
                language=_get(raw, "language"),
                backend="redux",
                model=self._model,
                duration_s=audio.duration_s,
                timing_source="native_word" if words else "unavailable",
                backend_version=info().version,
                device=self._device,
            )
        except Exception as exc:
            raise TranscriptionError(f"Redux inference failed: {exc}") from exc

    def close(self) -> None:
        if getattr(self, "_manager", None) is not None:
            manager, self._manager = self._manager, None
            self._speech = None
            manager.__exit__(None, None, None)
