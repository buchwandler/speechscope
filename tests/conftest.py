from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speechscope import BackendInfo, TranscriptionResult


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    sf.write(path, np.zeros(24000, dtype=np.float32), 24000)
    return path


@pytest.fixture
def register_fake():
    from speechscope.registry import _FACTORIES

    def add(text: str, words=(), *, name: str = "ci-fake", language_hint: bool = False):
        class Fake:
            def __init__(self, **options):
                self.options = options
                self.closed = False
                self.calls = 0

            def info(self):
                return BackendInfo(name, "test", True, True, language_hint, ("cpu",))

            def transcribe(self, audio, *, language=None):
                self.calls += 1
                return TranscriptionResult(
                    text=text,
                    words=tuple(words),
                    language=language,
                    backend=name,
                    model=None,
                    duration_s=audio.duration_s,
                    timing_source="native_word" if words else "unavailable",
                    device="cpu",
                )

            def close(self):
                self.closed = True

        instances = []

        def factory(**kwargs):
            instance = Fake(**kwargs)
            instances.append(instance)
            return instance

        _FACTORIES[name] = factory
        return instances

    yield add
    _FACTORIES.pop("ci-fake", None)
