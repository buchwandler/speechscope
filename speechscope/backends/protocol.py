"""Backend-neutral ASR contract."""

from typing import Protocol

from ..types import AudioInput, BackendInfo, TranscriptionResult


class SpeechTranscriber(Protocol):
    def info(self) -> BackendInfo: ...
    def transcribe(
        self, audio: AudioInput, *, language: str | None = None
    ) -> TranscriptionResult: ...
    def close(self) -> None: ...
