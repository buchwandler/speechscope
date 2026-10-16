from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from jsonschema import validate

from speechscope import VerificationRequest, analyze, available_backends, register_backend
from speechscope.api import transcribe_report
from speechscope.audio import load_wav
from speechscope.backends.whispercpp import (
    EXECUTABLE_ENV,
    MODEL_ENV,
    WhisperCppTranscriber,
    info,
)
from speechscope.cli import main
from speechscope.errors import (
    BackendUnavailableError,
    TranscriptionError,
    UnsupportedBackendOptionError,
)

FAKE_CLI = r"""#!/usr/bin/env python3
import json
import os
import sys
import soundfile as sf

args = sys.argv[1:]
if args == ["--version"]:
    if os.environ.get("FAKE_MODE") == "help-only":
        print("version unavailable", file=sys.stderr)
        raise SystemExit(1)
    print("whisper.cpp version: fake-1.0")
    raise SystemExit(0)
if args == ["--help"]:
    print("whisper-cli fake help")
    raise SystemExit(0)

def option(name):
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None

input_path = option("--file")
model_path = option("--model")
prefix = option("--output-file")
with open(os.environ["FAKE_ARGS_FILE"], "w", encoding="utf-8") as stream:
    json.dump({"argv": args, "input_path": input_path, "model_path": model_path}, stream)
if not input_path or not model_path or not prefix:
    print("missing required argument", file=sys.stderr)
    raise SystemExit(8)
info = sf.info(input_path)
if info.format != "WAV" or info.subtype != "PCM_16" or info.channels != 1 or info.samplerate != 16000:
    print("unexpected provider WAV format", file=sys.stderr)
    raise SystemExit(9)
mode = os.environ.get("FAKE_MODE", "success")
if mode == "nonzero":
    print("deterministic fake failure", file=sys.stderr)
    raise SystemExit(7)
if mode == "missing":
    raise SystemExit(0)
result_path = prefix + ".json"
if mode == "malformed":
    with open(result_path, "w", encoding="utf-8") as stream:
        stream.write("{not json")
    raise SystemExit(0)
if mode == "segment":
    payload = {
        "result": {"text": "hello world", "language": "en"},
        "transcription": [{"text": " hello world", "offsets": {"from": 0, "to": 800}}],
    }
else:
    payload = {
        "result": {"text": "Hallo, café 世界!", "language": "de"},
        "transcription": [
            {"text": " Hallo,", "offsets": {"from": 100, "to": 400}},
            {"text": " café", "timestamps": {"from": "00:00:00,400", "to": "00:00:00,700"}},
            {"text": " 世界!", "start": 0.7, "end": 1.0},
        ],
    }
with open(result_path, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False)
"""


def fake_cli(tmp_path, monkeypatch, *, mode="success"):
    executable = tmp_path / "whisper-cli-fake"
    executable.write_text(
        "#!" + sys.executable + "\n" + FAKE_CLI.split("\n", 1)[1], encoding="utf-8"
    )
    executable.chmod(0o755)
    model = tmp_path / "model.ggml"
    model.write_bytes(b"test model placeholder")
    args_file = tmp_path / "args.json"
    monkeypatch.setenv(EXECUTABLE_ENV, str(executable))
    monkeypatch.setenv("FAKE_ARGS_FILE", str(args_file))
    monkeypatch.setenv("FAKE_MODE", mode)
    return executable, model, args_file


def source_wav(tmp_path, *, rate=44_100, frames=4410, channels=2, peak=0.4):
    path = tmp_path / "source.wav"
    left = np.linspace(-peak, peak, frames, dtype=np.float64)
    samples = left if channels == 1 else np.column_stack((left, -left / 2))
    sf.write(path, samples, rate, format="WAV", subtype="FLOAT")
    return path


def run_backend(path, *, model=None, device="cpu", language=None):
    audio = load_wav(path)
    backend = WhisperCppTranscriber(device=device, model=model)
    try:
        return backend.transcribe(audio, language=language), audio
    finally:
        backend.close()


def test_fake_cli_receives_canonical_wav_and_result_has_backend_provenance(tmp_path, monkeypatch):
    _, model, args_file = fake_cli(tmp_path, monkeypatch)
    source = source_wav(tmp_path)
    result, audio = run_backend(source, model=str(model), device="cpu", language="de")

    assert result.duration_s == audio.duration_s == pytest.approx(0.1)
    assert result.model == str(model.resolve())
    assert result.text == "Hallo, café 世界!"
    assert result.language == "de"
    assert result.timing_source == "native_word"
    assert [word.start_s for word in result.words] == [0.1, 0.4, 0.7]
    assert result.backend_audio_input is not None
    assert result.backend_audio_input.kind == "generated_wav_path"
    assert result.backend_audio_input.selection == "canonicalized_wav"
    assert result.backend_audio_input.sample_rate_hz == 16_000
    assert result.backend_audio_input.channels == 1
    assert result.backend_audio_input.sample_format == "PCM_16"
    assert result.backend_audio_input.transformations == (
        "average_channels",
        "resample:44100->16000:windowed_sinc-fw32-r0.945-round",
        "encode:WAV/PCM_16",
    )
    invocation = json.loads(args_file.read_text(encoding="utf-8"))
    argv = invocation["argv"]
    assert invocation["model_path"] == str(model.resolve())
    assert argv[argv.index("--language") + 1] == "de"
    assert "--no-gpu" in argv
    assert "--output-json-full" in argv
    assert "--no-prints" in argv
    assert "--split-on-word" in argv
    assert invocation["input_path"] not in repr(result)
    assert not Path(invocation["input_path"]).exists()


def test_auto_device_omits_no_gpu_and_explicit_model_overrides_environment(tmp_path, monkeypatch):
    _, explicit_model, args_file = fake_cli(tmp_path, monkeypatch)
    env_model = tmp_path / "environment-model.ggml"
    env_model.write_bytes(b"environment model")
    monkeypatch.setenv(MODEL_ENV, str(env_model))
    source = source_wav(tmp_path, rate=16_000, frames=1600, channels=1)

    result, _ = run_backend(source, model=str(explicit_model), device="auto")
    invocation = json.loads(args_file.read_text(encoding="utf-8"))
    assert invocation["model_path"] == str(explicit_model.resolve())
    assert "--no-gpu" not in invocation["argv"]
    assert result.backend_audio_input.transformations == ("encode:WAV/PCM_16",)


def test_environment_model_is_used_when_explicit_model_is_absent(tmp_path, monkeypatch):
    _, model, args_file = fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv(MODEL_ENV, str(model))
    source = source_wav(tmp_path, rate=16_000, frames=1600, channels=1)
    run_backend(source)
    invocation = json.loads(args_file.read_text(encoding="utf-8"))
    assert invocation["model_path"] == str(model.resolve())
    assert invocation["argv"][invocation["argv"].index("--language") + 1] == "auto"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("nonzero", "exited with status 7"),
        ("malformed", "Could not read whisper-cli JSON"),
        ("missing", "did not create its JSON result"),
    ],
)
def test_cli_failures_are_controlled_and_cleanup_temp_wav(tmp_path, monkeypatch, mode, message):
    _, model, args_file = fake_cli(tmp_path, monkeypatch, mode=mode)
    source = source_wav(tmp_path, rate=16_000, frames=1600, channels=1)
    audio = load_wav(source)
    backend = WhisperCppTranscriber(model=str(model))
    with pytest.raises(TranscriptionError, match=message):
        backend.transcribe(audio)
    input_path = json.loads(args_file.read_text(encoding="utf-8"))["input_path"]
    assert not Path(input_path).exists()


def test_segment_timing_is_not_promoted_to_word_timing(tmp_path, monkeypatch):
    _, model, _ = fake_cli(tmp_path, monkeypatch, mode="segment")
    source = source_wav(tmp_path, rate=16_000, frames=1600, channels=1)
    result, _ = run_backend(source, model=str(model))
    assert result.timing_source == "derived_segment"
    assert len(result.words) == 1
    assert result.words[0].text == "hello world"


def test_pcm16_clipping_diagnostic_is_reported(tmp_path, monkeypatch):
    _, model, _ = fake_cli(tmp_path, monkeypatch)
    source = source_wav(tmp_path, rate=16_000, frames=1600, channels=1, peak=1.5)
    result, audio = run_backend(source, model=str(model))
    expected_count = int(np.count_nonzero((audio.samples < -1.0) | (audio.samples > 1.0)))
    assert result.diagnostics == (
        {"code": "backend_audio_clipped_for_pcm16", "sample_count": expected_count},
    )


def test_options_fail_before_process_or_audio_materialization(tmp_path, monkeypatch):
    executable, model, _ = fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGS_FILE", str(tmp_path / "unused-args.json"))
    with pytest.raises(UnsupportedBackendOptionError, match="supports only cpu and auto"):
        WhisperCppTranscriber(device="cuda", model=str(model))
    with pytest.raises(UnsupportedBackendOptionError, match="does not support cache_dir"):
        WhisperCppTranscriber(device="cpu", model=str(model), cache_dir=tmp_path)
    with pytest.raises(BackendUnavailableError, match="model file does not exist"):
        WhisperCppTranscriber(device="cpu", model=str(tmp_path / "missing.ggml"))
    assert executable.is_file()
    assert not (tmp_path / "unused-args.json").exists()


def test_metadata_discovery_does_not_execute_cli(tmp_path, monkeypatch):
    fake_cli(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGS_FILE", str(tmp_path / "should-not-exist.json"))
    assert info().name == "whispercpp"
    assert info().version is None
    assert not (tmp_path / "should-not-exist.json").exists()


def test_executable_must_resolve_and_model_environment_is_required(tmp_path, monkeypatch):
    missing = tmp_path / "not-an-executable"
    model = tmp_path / "model.ggml"
    model.write_bytes(b"placeholder")
    monkeypatch.setenv(EXECUTABLE_ENV, str(missing))
    with pytest.raises(BackendUnavailableError, match="not found or is not runnable"):
        WhisperCppTranscriber(model=str(model))

    executable, _, _ = fake_cli(tmp_path, monkeypatch)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    with pytest.raises(BackendUnavailableError, match="Provide a local whisper.cpp model"):
        WhisperCppTranscriber()
    assert executable.is_file()


def test_report_provenance_preserves_original_input_and_excludes_temp_path(tmp_path, monkeypatch):
    _, model, args_file = fake_cli(tmp_path, monkeypatch)
    source = source_wav(tmp_path)
    report = transcribe_report(source, backend="whispercpp", model=str(model), language="de")
    data = report.to_dict()
    assert data["input"]["audio_path"] == str(source)
    assert data["input"]["duration_s"] == pytest.approx(0.1)
    assert data["provenance"]["backend_audio_input"] == {
        "kind": "generated_wav_path",
        "selection": "canonicalized_wav",
        "source_path": str(source),
        "sample_rate_hz": 16_000,
        "channels": 1,
        "sample_format": "PCM_16",
        "loader_sample_transformations": ["average_channels"],
        "transformations_on_backend_input": [
            "average_channels",
            "resample:44100->16000:windowed_sinc-fw32-r0.945-round",
            "encode:WAV/PCM_16",
        ],
    }
    invocation = json.loads(args_file.read_text(encoding="utf-8"))
    assert invocation["input_path"] not in report.to_json()
    schema_path = Path(__file__).parents[1] / "speechscope" / "schemas" / "report-v1.schema.json"
    validate(data, json.loads(schema_path.read_text(encoding="utf-8")))


def test_pcm16_clipping_diagnostic_survives_in_public_report(tmp_path, monkeypatch):
    _, model, _ = fake_cli(tmp_path, monkeypatch)
    source = source_wav(tmp_path, rate=16_000, frames=1600, channels=1, peak=1.5)
    report = transcribe_report(source, backend="whispercpp", model=str(model))
    diagnostics = report.to_dict()["diagnostics"]
    clipping = next(
        item for item in diagnostics if item["code"] == "backend_audio_clipped_for_pcm16"
    )
    assert clipping["sample_count"] > 0


def test_builtin_discovery_is_lazy_and_names_are_reserved(monkeypatch):
    from speechscope import registry

    def unexpected_process(*args, **kwargs):
        raise AssertionError("discovery must not execute whisper-cli")

    monkeypatch.setattr("speechscope.backends.whispercpp.subprocess.run", unexpected_process)
    names = {backend.name for backend in available_backends()}
    assert {"redux", "whispercpp"} <= names
    with pytest.raises(ValueError, match="already reserved"):
        register_backend("whispercpp", lambda **options: None)
    with pytest.raises(ValueError, match="already reserved"):
        register_backend("redux", lambda **options: None)

    class EntryPoint:
        name = "whispercpp"

    monkeypatch.setattr(
        registry.importlib.metadata, "entry_points", lambda **kwargs: (EntryPoint(),)
    )
    with pytest.raises(BackendUnavailableError, match="Duplicate backend entry-point names"):
        available_backends()


def test_whispercpp_doctor_checks_cli_without_a_model(tmp_path, monkeypatch, capsys):
    executable, _, _ = fake_cli(tmp_path, monkeypatch)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    assert main(["doctor", "--backend", "whispercpp"]) == 0
    output = capsys.readouterr().out
    assert "fake-1.0" in output
    assert str(executable.resolve()) in output
    assert "Model weights are not accessed" in output


def test_whispercpp_doctor_falls_back_to_help(tmp_path, monkeypatch, capsys):
    fake_cli(tmp_path, monkeypatch, mode="help-only")
    monkeypatch.delenv(MODEL_ENV, raising=False)
    assert main(["doctor", "--backend", "whispercpp"]) == 0
    assert "version unavailable" in capsys.readouterr().out


def test_whispercpp_doctor_fails_when_executable_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(EXECUTABLE_ENV, str(tmp_path / "missing-whisper-cli"))
    assert main(["doctor", "--backend", "whispercpp"]) == 3
    assert "not found or is not runnable" in capsys.readouterr().err


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SPEECHSCOPE_RUN_WHISPERCPP_INTEGRATION") != "1",
    reason="Set SPEECHSCOPE_RUN_WHISPERCPP_INTEGRATION=1 for real whisper.cpp inference.",
)
def test_real_whispercpp_cpu_integration():
    from speechscope.backends.whispercpp import probe_executable

    fixture = Path(os.environ["SPEECHSCOPE_WHISPERCPP_FIXTURE"])
    reference = os.environ["SPEECHSCOPE_WHISPERCPP_REFERENCE"]
    rights_evidence = os.environ["SPEECHSCOPE_WHISPERCPP_FIXTURE_LICENSE"]
    model = os.environ[MODEL_ENV]
    assert fixture.is_file(), "SPEECHSCOPE_WHISPERCPP_FIXTURE must name an authorized WAV"
    assert reference.strip(), "Provide the fixture's reference transcript"
    assert rights_evidence.strip(), "Record fixture rights evidence before running"
    executable, version = probe_executable()
    report = analyze(
        VerificationRequest(
            audio=fixture,
            reference_text=reference,
            backend="whispercpp",
            device="cpu",
            model=model,
        )
    )
    data = report.to_dict()
    assert data["transcript"]["raw"].strip()
    assert data["provenance"]["backend_audio_input"]["sample_rate_hz"] == 16_000
    assert data["provenance"]["backend_audio_input"]["sample_format"] == "PCM_16"
    schema_path = Path(__file__).parents[1] / "speechscope" / "schemas" / "report-v1.schema.json"
    validate(data, json.loads(schema_path.read_text(encoding="utf-8")))
    print(
        f"whispercpp integration: executable={executable}; version={version}; "
        f"model={model}; device=cpu; fixture_rights={rights_evidence}"
    )
