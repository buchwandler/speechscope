from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from jsonschema import validate

from speechscope import TimedWord, transcribe
from speechscope.api import transcribe_report
from speechscope.audio import load_wav
from speechscope.backends.redux import DEFAULT_MODEL, ReduxTranscriber
from speechscope.errors import UnsupportedBackendOptionError
from speechscope.types import AudioInput


def audio_input(path: Path) -> AudioInput:
    return AudioInput(
        samples=np.zeros(24000, dtype=np.float64),
        sample_rate=24000,
        channels=1,
        duration_s=1.0,
        path=path,
    )


class FakeSpeech:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeManager:
    def __init__(self, speech):
        self.speech = speech
        self.closed = False

    def __enter__(self):
        return self.speech

    def __exit__(self, *args):
        self.closed = True


def install_sdk(monkeypatch, response):
    calls = []
    speech = FakeSpeech(response)
    manager = FakeManager(speech)
    sdk = types.ModuleType("moondream")

    def photon(*args, **kwargs):
        calls.append((args, kwargs))
        return manager

    sdk.photon = photon
    monkeypatch.setitem(sys.modules, "moondream", sdk)
    return calls, speech, manager


def test_redux_photon_sdk_contract_nested_segments_and_cpu_cleanup(monkeypatch, tmp_path):
    response = {
        "text": "Hello there",
        "language": "en",
        "segments": [
            {
                "start": 0.1,
                "end": 0.8,
                "text": "Hello there",
                "words": [
                    {"word": "Hello", "start": 0.1, "end": 0.35, "probability": 0.91},
                    {"word": "there", "start": 0.4, "end": 0.8},
                ],
            }
        ],
    }
    calls, speech, manager = install_sdk(monkeypatch, response)
    path = tmp_path / "speech.wav"
    backend = ReduxTranscriber(device="cpu")
    result = backend.transcribe(audio_input(path))

    assert calls == [((DEFAULT_MODEL,), {"device": "cpu"})]
    assert speech.calls == [{"audio": path, "timestamps": "word"}]
    assert result.text == "Hello there"
    assert result.language == "en"
    assert result.words == (
        TimedWord("Hello", 0.1, 0.35, 0.91),
        TimedWord("there", 0.4, 0.8),
    )
    assert result.timing_source == "native_word"
    backend.close()
    assert manager.closed


def test_redux_auto_device_omits_device_override(monkeypatch):
    calls, _, manager = install_sdk(monkeypatch, {"text": "hello", "words": []})
    backend = ReduxTranscriber(device="auto")
    assert calls == [((DEFAULT_MODEL,), {})]
    backend.close()
    assert manager.closed


def test_redux_model_and_device_options_fail_before_sdk_import(monkeypatch, wav, capsys):
    from speechscope.cli import main

    monkeypatch.setitem(sys.modules, "moondream", None)
    with pytest.raises(UnsupportedBackendOptionError, match="supports only model"):
        ReduxTranscriber(model="moondream/parakeet-ultra")
    with pytest.raises(UnsupportedBackendOptionError, match="unsupported device"):
        ReduxTranscriber(device="tpu")

    audio = wav
    assert (
        main(
            [
                "transcribe",
                str(audio),
                "--backend",
                "redux",
                "--model",
                "moondream/parakeet-ultra",
            ]
        )
        == 3
    )
    assert "supports only model" in capsys.readouterr().err
    assert sys.modules["moondream"] is None


def test_configured_moondream_release_exposes_lazy_photon_api():
    try:
        version = importlib.metadata.version("moondream")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Optional Moondream SDK is not installed")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (2, 4, 1):
        pytest.skip(f"Installed Moondream {version} is below the supported SDK minimum")
    expected = os.environ.get("SPEECHSCOPE_EXPECTED_MOONDREAM_VERSION")
    if expected:
        assert version == expected
    sdk = importlib.import_module("moondream")
    assert callable(getattr(sdk, "photon", None))


def test_redux_stereo_path_and_loader_provenance_agree(monkeypatch, tmp_path):
    source = np.linspace(-0.5, 0.5, 2400, dtype=np.float64)
    path = tmp_path / "opposite-phase.wav"
    sf.write(path, np.column_stack((source, -source)), 24000, subtype="FLOAT")
    loaded = load_wav(path)
    assert loaded.channels == 2
    assert loaded.transforms == ("average_channels",)
    assert np.allclose(loaded.samples, 0.0)

    calls, speech, _ = install_sdk(
        monkeypatch,
        {"text": "hello", "segments": [{"words": [{"word": "hello", "start": 0.1, "end": 0.4}]}]},
    )
    report = transcribe_report(path, backend="redux", device="cpu")
    assert calls[0][1] == {"device": "cpu"}
    assert calls[0][0] == (DEFAULT_MODEL,)
    assert speech.calls == [{"audio": path, "timestamps": "word"}]
    # The actual SDK argument is the original encoded file, not the silent mono samples.
    assert report.transcription.words[0].text == "hello"
    provenance = report.to_dict()["provenance"]
    assert provenance["audio_transformations"] == ["average_channels"]
    assert provenance["backend_audio_input"] == {
        "kind": "encoded_wav_path",
        "selection": "encoded_path",
        "path": str(path),
        "sample_rate_hz": 24000,
        "source_channels": 2,
        "loader_sample_transformations": ["average_channels"],
        "transformations_on_backend_input": [],
    }
    assert report.to_dict()["provenance"]["backend_audio_input"]["path"] == str(path)
    schema = json.loads(
        (
            Path(__file__).parents[1] / "speechscope" / "schemas" / "report-v1.schema.json"
        ).read_text()
    )
    validate(report.to_dict(), schema)


def test_redux_import_is_lazy_and_does_not_import_sdk():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'moondream':
        raise AssertionError('moondream imported during SpeechScope import')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import speechscope
print(speechscope.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "speechscope"
    assert completed.stderr == ""


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SPEECHSCOPE_RUN_REDUX_INTEGRATION") != "1",
    reason="Set SPEECHSCOPE_RUN_REDUX_INTEGRATION=1 to download/use model weights.",
)
def test_real_redux_cpu_cli_integration(capsys):
    audio = Path(os.environ["SPEECHSCOPE_REDUX_FIXTURE"])
    reference = os.environ["SPEECHSCOPE_REDUX_REFERENCE"]
    license_evidence = os.environ["SPEECHSCOPE_REDUX_FIXTURE_LICENSE"]
    assert audio.is_file(), "SPEECHSCOPE_REDUX_FIXTURE must name an authorized WAV file"
    assert reference.strip(), "Provide the fixture's known transcript"
    assert license_evidence.strip(), "Record fixture rights/license evidence before running"
    sdk_version = importlib.metadata.version("moondream")
    print(f"Redux integration: moondream={sdk_version}; fixture_license={license_evidence}")

    doctor_code, doctor_output = main_for_test(["doctor", "--backend", "redux"], capsys)
    assert doctor_code == 0
    assert "dependency available" in doctor_output.out
    transcript = transcribe(audio, backend="redux", device="cpu")
    assert transcript.text.strip()
    assert transcript.words
    starts = []
    for word in transcript.words:
        assert word.start_s is not None and word.end_s is not None
        assert 0 <= word.start_s <= word.end_s <= transcript.duration_s + 0.05
        starts.append(word.start_s)
    assert starts == sorted(starts)

    transcribe_code, transcribe_output = main_for_test(
        ["transcribe", str(audio), "--backend", "redux", "--device", "cpu", "--format", "json"],
        capsys,
    )
    assert transcribe_code == 0
    transcribe_report = json.loads(transcribe_output.out)
    assert transcribe_report["transcript"]["raw"].strip()
    assert transcribe_report["transcript"]["words"]

    analysis_code, analysis_output = main_for_test(
        [
            "analyze",
            str(audio),
            "--backend",
            "redux",
            "--device",
            "cpu",
            "--text",
            reference,
            "--format",
            "json",
        ],
        capsys,
    )
    assert analysis_code == 0
    analysis = json.loads(analysis_output.out)
    schema = json.loads(
        (
            Path(__file__).parents[1] / "speechscope" / "schemas" / "report-v1.schema.json"
        ).read_text()
    )
    validate(analysis, schema)
    assert analysis["transcript"]["raw"].strip()


def main_for_test(argv, capsys):
    from speechscope.cli import main

    code = main(argv)
    captured = capsys.readouterr()
    assert captured.err == ""
    return code, captured
