"""Demonstrate the minimum provider protocol with an offline fake transcript."""

import argparse
from pathlib import Path

from speechscope import (
    AudioInput,
    BackendInfo,
    TranscriptionResult,
    VerificationRequest,
    analyze,
    register_backend,
)


class ExampleBackend:
    def __init__(self, text: str):
        self.text = text

    def info(self) -> BackendInfo:
        return BackendInfo("example-text", "1.0", False, False, False, ("cpu",))

    def transcribe(self, audio: AudioInput, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(
            text=self.text,
            words=(),
            language=language,
            backend="example-text",
            model=None,
            duration_s=audio.duration_s,
            timing_source="unavailable",
            backend_version="1.0",
            device="cpu",
        )

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--transcript", required=True)
    args = parser.parse_args()

    register_backend("example-text", lambda **_options: ExampleBackend(args.transcript))
    result = analyze(
        VerificationRequest(
            audio=args.audio,
            reference_text=args.reference,
            backend="example-text",
        )
    )
    assert result.metrics is not None
    print(f"fake-backend WER: {result.metrics.wer:.4f}")


if __name__ == "__main__":
    main()
