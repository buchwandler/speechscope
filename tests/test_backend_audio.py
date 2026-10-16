from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from speechscope.audio import load_wav
from speechscope.backend_audio import materialize_pcm16_wav
from speechscope.errors import TranscriptionError
from speechscope.types import AudioInput


def write_float_wav(tmp_path, samples, sample_rate):
    path = tmp_path / "source.wav"
    sf.write(path, samples, sample_rate, format="WAV", subtype="FLOAT")
    return path


def assert_pcm16_wav(path, sample_rate):
    info = sf.info(path)
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.channels == 1
    assert info.samplerate == sample_rate
    assert info.frames > 0
    return info


def test_48khz_mono_resamples_to_pcm16_and_cleans_up(tmp_path):
    source_path = write_float_wav(
        tmp_path,
        np.sin(np.linspace(0, 20, 4800, dtype=np.float64)) * 0.5,
        48_000,
    )
    audio = load_wav(source_path)

    with materialize_pcm16_wav(audio, sample_rate=16_000) as prepared:
        info = assert_pcm16_wav(prepared.path, 16_000)
        assert info.frames == round(4800 * 16_000 / 48_000)
        assert abs(info.frames / 16_000 - audio.duration_s) <= 1 / 32_000
        assert prepared.transformations == (
            "resample:48000->16000:windowed_sinc-fw32-r0.945-round",
            "encode:WAV/PCM_16",
        )
        assert prepared.clipped_sample_count == 0
        temp_path = prepared.path
        assert temp_path.is_file()

    assert not temp_path.exists()


def test_44100hz_stereo_is_averaged_once_then_resampled(tmp_path):
    left = np.linspace(-0.4, 0.4, 4410, dtype=np.float64)
    right = np.linspace(0.2, -0.2, 4410, dtype=np.float64)
    source_path = write_float_wav(tmp_path, np.column_stack((left, right)), 44_100)
    audio = load_wav(source_path)
    assert audio.channels == 2
    assert audio.transforms == ("average_channels",)
    assert np.allclose(audio.samples, (left + right) / 2)

    with materialize_pcm16_wav(audio, sample_rate=16_000) as prepared:
        info = assert_pcm16_wav(prepared.path, 16_000)
        assert info.frames == round(4410 * 16_000 / 44_100)
        assert prepared.transformations == (
            "average_channels",
            "resample:44100->16000:windowed_sinc-fw32-r0.945-round",
            "encode:WAV/PCM_16",
        )


def test_16khz_mono_does_not_resample(tmp_path):
    source_path = write_float_wav(
        tmp_path,
        np.linspace(-0.5, 0.5, 1600, dtype=np.float64),
        16_000,
    )
    audio = load_wav(source_path)

    with materialize_pcm16_wav(audio, sample_rate=16_000) as prepared:
        info = assert_pcm16_wav(prepared.path, 16_000)
        assert info.frames == 1600
        assert prepared.transformations == ("encode:WAV/PCM_16",)
        assert prepared.diagnostics == ()


def test_out_of_pcm_range_samples_are_clipped_and_diagnosed(tmp_path):
    source_path = write_float_wav(tmp_path, np.array([-1.2, 0.25, 1.5]), 16_000)
    audio = load_wav(source_path)

    with materialize_pcm16_wav(audio, sample_rate=16_000) as prepared:
        assert_pcm16_wav(prepared.path, 16_000)
        encoded, rate = sf.read(prepared.path, dtype="float64")
        assert rate == 16_000
        assert encoded.min() >= -1.0
        assert encoded.max() <= 1.0
        assert prepared.clipped_sample_count == 2
        assert prepared.transformations == (
            "clip_to_pcm16_range",
            "encode:WAV/PCM_16",
        )
        assert prepared.diagnostics == (
            {"code": "backend_audio_clipped_for_pcm16", "sample_count": 2},
        )


def test_temp_wav_is_removed_when_consumer_raises(tmp_path):
    source_path = write_float_wav(
        tmp_path,
        np.ones(160, dtype=np.float64) * 0.1,
        16_000,
    )
    audio = load_wav(source_path)
    temp_path = None

    with (
        pytest.raises(RuntimeError, match="consumer failed"),
        materialize_pcm16_wav(audio, sample_rate=16_000) as prepared,
    ):
        temp_path = prepared.path
        assert temp_path.is_file()
        raise RuntimeError("consumer failed")

    assert temp_path is not None
    assert not temp_path.exists()


def test_invalid_backend_samples_fail_with_controlled_error(tmp_path):
    audio = AudioInput(
        samples=np.array([0.0, np.nan], dtype=np.float64),
        sample_rate=16_000,
        channels=1,
        duration_s=2 / 16_000,
        path=tmp_path / "invalid.wav",
    )

    with (
        pytest.raises(TranscriptionError, match="must be finite"),
        materialize_pcm16_wav(audio, sample_rate=16_000),
    ):
        pytest.fail("invalid audio must not be materialized")
