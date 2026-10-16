"""Deterministic WAV materialization for provider inputs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .errors import TranscriptionError
from .types import AudioInput


@dataclass(frozen=True, slots=True)
class PreparedWav:
    """A temporary mono PCM16 WAV and the transformations used to create it."""

    path: Path
    sample_rate: int
    channels: int
    subtype: str
    transformations: tuple[str, ...]
    clipped_sample_count: int

    @property
    def diagnostics(self) -> tuple[dict[str, object], ...]:
        if not self.clipped_sample_count:
            return ()
        return (
            {
                "code": "backend_audio_clipped_for_pcm16",
                "sample_count": self.clipped_sample_count,
            },
        )


@contextmanager
def materialize_pcm16_wav(
    audio: AudioInput,
    *,
    sample_rate: int,
    filter_width: int = 32,
    rolloff: float = 0.945,
    length_mode: str = "round",
) -> Iterator[PreparedWav]:
    """Write an ephemeral mono PCM16 WAV from already-loaded mono samples.

    The resampler runs only when the source rate differs. This helper does not
    alter source duration semantics; callers continue to report ``audio.duration_s``.
    """
    try:
        import soundfile as sf

        source_rate = audio.sample_rate
        if source_rate <= 0 or sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        samples = np.asarray(audio.samples)
        if samples.ndim != 1:
            raise ValueError("backend audio samples must be one-dimensional mono data")
        if samples.size == 0:
            raise ValueError("backend audio samples must be non-empty")
        if samples.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("backend audio samples must be float32 or float64")
        if not np.isfinite(samples).all():
            raise ValueError("backend audio samples must be finite")

        transformations = list(audio.transforms)
        if source_rate != sample_rate:
            from audiosig import resample

            samples = resample(
                samples,
                source_rate=source_rate,
                target_rate=sample_rate,
                filter_width=filter_width,
                rolloff=rolloff,
                length_mode=length_mode,
            )
            transformations.append(
                f"resample:{source_rate}->{sample_rate}:windowed_sinc-"
                f"fw{filter_width}-r{rolloff:g}-{length_mode}"
            )
        else:
            samples = np.ascontiguousarray(samples)

        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError("resampler output must be one-dimensional mono data")
        if samples.size == 0:
            raise ValueError("resampler output must be non-empty")
        if samples.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("resampler output must be float32 or float64")
        if not np.isfinite(samples).all():
            raise ValueError("resampler output must be finite")
        clipped_sample_count = int(np.count_nonzero((samples < -1.0) | (samples > 1.0)))
        if clipped_sample_count:
            samples = np.clip(samples, -1.0, 1.0)
            transformations.append("clip_to_pcm16_range")
        transformations.append("encode:WAV/PCM_16")
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"Failed to prepare backend WAV samples: {exc}") from exc

    try:
        temporary_directory = TemporaryDirectory(prefix="speechscope-asr-")
    except Exception as exc:
        raise TranscriptionError(
            f"Failed to create temporary backend WAV directory: {exc}"
        ) from exc

    with temporary_directory as directory:
        path = Path(directory) / "input.wav"
        try:
            sf.write(
                path,
                samples,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
            info = sf.info(path)
            if (
                info.format != "WAV"
                or info.subtype != "PCM_16"
                or info.channels != 1
                or info.samplerate != sample_rate
                or info.frames <= 0
            ):
                raise TranscriptionError("Materialized backend WAV metadata is not canonical")
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Failed to write canonical backend WAV: {exc}") from exc

        yield PreparedWav(
            path=path,
            sample_rate=sample_rate,
            channels=1,
            subtype="PCM_16",
            transformations=tuple(transformations),
            clipped_sample_count=clipped_sample_count,
        )
