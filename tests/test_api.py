import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from jsonschema import validate

from speechscope import (
    BackendInfo,
    ThresholdPolicy,
    TimedWord,
    TranscriptionResult,
    VerificationRequest,
    Verifier,
    analyze,
    available_backends,
    register_backend,
    transcribe,
)
from speechscope.api import transcribe_report
from speechscope.cli import main
from speechscope.errors import (
    AlignmentTooLargeError,
    BackendUnavailableError,
    ExportError,
    InvalidAudioError,
    InvalidReferenceError,
    InvalidThresholdPolicyError,
    TranscriptionError,
)
from speechscope.exports import write_captions, write_words


def test_analyze_alignment_and_json_schema(register_fake, wav: Path):
    register_fake("Hello world", (TimedWord("Hello", 0.1, 0.35), TimedWord("world", 0.4, 0.8)))
    result = analyze(VerificationRequest(wav, "Hello world", backend="ci-fake"))
    assert result.metrics.wer == 0 and result.metrics.cer == 0
    assert result.metrics.reference_timestamp_coverage_ratio == 1
    assert [word.operation for word in result.aligned_words] == ["match", "match"]
    schema = json.loads(
        (
            Path(__file__).parents[1] / "speechscope" / "schemas" / "report-v1.schema.json"
        ).read_text()
    )
    validate(result.to_dict(), schema)
    assert result.evaluation["status"] == "not_evaluated"
    assert result.to_dict()["tool"]["name"] == "speechscope"
    result.write_json(wav.parent / "report.json")
    assert json.loads((wav.parent / "report.json").read_text())["metrics"]["wer"] == 0
    write_words(result, wav.parent / "words.csv")
    write_captions(result, wav.parent / "captions.srt")
    write_captions(result, wav.parent / "captions.vtt", kind="vtt")
    assert "00:00:00,100" in (wav.parent / "captions.srt").read_text()
    assert (wav.parent / "captions.vtt").read_text().startswith("WEBVTT")


def test_missing_timing_review_policy(register_fake, wav):
    register_fake("hello")
    result = analyze(
        VerificationRequest(
            wav,
            "hello",
            backend="ci-fake",
            thresholds=ThresholdPolicy(min_reference_timestamp_coverage=0.9),
        )
    )
    assert result.evaluation["status"] == "review"
    assert result.metrics.trailing_audio_s is None
    with pytest.raises(ExportError):
        write_captions(result, wav.with_suffix(".srt"))


@pytest.mark.parametrize(
    "reference, transcript, words, expected_coverage, expected_wer, expected_status",
    [
        ("dog", "dog", (TimedWord("dog", 0.1, 0.4),), 1.0, 0.0, "pass"),
        ("dog", "cat", (TimedWord("cat", 0.1, 0.4),), 1.0, 1.0, "pass"),
        (
            "dog",
            "dog extra",
            (TimedWord("dog", 0.1, 0.4), TimedWord("extra", 0.5, 0.7)),
            1.0,
            1.0,
            "pass",
        ),
        ("dog cat", "cat", (TimedWord("cat", 0.1, 0.4),), 0.5, 0.5, "fail"),
        ("dog", "dog", (), 0.0, 0.0, "review"),
    ],
    ids=["match", "substitution-is-position-coverage", "insertion", "deletion", "missing-timing"],
)
def test_timestamp_coverage_means_alignment_positions(
    register_fake,
    wav,
    reference,
    transcript,
    words,
    expected_coverage,
    expected_wer,
    expected_status,
):
    register_fake(transcript, words)
    result = analyze(
        VerificationRequest(
            wav,
            reference,
            backend="ci-fake",
            thresholds=ThresholdPolicy(min_reference_timestamp_coverage=0.9),
        )
    )
    assert result.metrics.reference_timestamp_coverage_ratio == expected_coverage
    assert result.metrics.wer == expected_wer
    assert result.evaluation["status"] == expected_status


def test_empty_threshold_policy_is_not_evaluated_in_api_and_cli(
    register_fake, wav, tmp_path, capsys
):
    register_fake("hello", (TimedWord("hello", 0.1, 0.4),))
    result = analyze(
        VerificationRequest(wav, "hello", backend="ci-fake", thresholds=ThresholdPolicy())
    )
    assert result.evaluation == {"status": "not_evaluated", "rules": []}

    policy = tmp_path / "empty-policy.json"
    policy.write_text("{}")
    assert (
        main(
            [
                "analyze",
                str(wav),
                "--backend",
                "ci-fake",
                "--text",
                "hello",
                "--thresholds",
                str(policy),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["evaluation"] == {
        "status": "not_evaluated",
        "rules": [],
    }


def test_invalid_timing_reported(register_fake, wav):
    register_fake("hello", (TimedWord("hello", float("nan"), 1.0),))
    result = analyze(VerificationRequest(wav, "hello", backend="ci-fake"))
    assert result.metrics.invalid_timestamp_count == 1
    assert result.aligned_words[0].start_s is None
    json.loads(result.to_json())


def test_token_mapping_not_guessed(register_fake, wav):
    register_fake("a b", (TimedWord("ab", 0.0, 0.5),))
    result = analyze(VerificationRequest(wav, "a b", backend="ci-fake"))
    assert result.metrics.hypothesis_timestamp_coverage_ratio == 0
    assert any(x["code"] == "word_tokenization_mismatch" for x in result.diagnostics)


def test_session_reuse(register_fake, wav):
    instances = register_fake("test")
    with Verifier(backend="ci-fake") as sess:
        sess.transcribe(wav)
        sess.transcribe(wav)
        assert instances[0].calls == 2
    assert instances[0].closed
    assert len(instances) == 1


def test_cli_clean_json_and_threshold_exit(register_fake, wav, capsys):
    register_fake("hello", (TimedWord("hello", 0, 0.4),))
    assert (
        main(["analyze", str(wav), "--backend", "ci-fake", "--text", "hello", "--format", "json"])
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["metrics"]["wer"] == 0
    assert captured.err == ""
    p = wav.parent / "policy.json"
    p.write_text('{"schema_version":"1","max_wer":0}')
    assert (
        main(
            [
                "analyze",
                str(wav),
                "--backend",
                "ci-fake",
                "--text",
                "goodbye",
                "--thresholds",
                str(p),
            ]
        )
        == 1
    )
    assert "Status: fail" in capsys.readouterr().out


def test_transcribe_no_reference_metrics(register_fake, wav, capsys):
    register_fake("hello")
    assert main(["transcribe", str(wav), "--backend", "ci-fake", "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reference"] is None and result["metrics"] is None
    schema = json.loads(
        (
            Path(__file__).parents[1] / "speechscope" / "schemas" / "report-v1.schema.json"
        ).read_text()
    )
    validate(result, schema)


def test_invalid_audio(wav):
    sf.write(wav, np.array([], dtype=np.float64), 24000)
    with pytest.raises(InvalidAudioError):
        from speechscope.audio import load_wav

        load_wav(wav)


def test_registry_discovery_no_download_and_duplicate():
    assert "redux" in [b.name for b in available_backends()]
    with pytest.raises(ValueError):
        register_backend("redux", lambda **kw: None)
    completed = subprocess.run(
        [sys.executable, "-m", "speechscope", "backends", "list", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout)[0]["name"] in ("redux",)


def test_transcribe_words_csv(register_fake, wav):
    register_fake("hello", (TimedWord("hello", 0.1, 0.9),))
    path = wav.with_suffix(".csv")
    assert main(["transcribe", str(wav), "--backend", "ci-fake", "--words", str(path)]) == 0
    assert "transcribed" in path.read_text()


def test_stereo_downmix_and_audio_provenance(tmp_path, register_fake):
    wav = tmp_path / "stereo.wav"
    stereo = np.column_stack((np.ones(3200), -np.ones(3200)))
    sf.write(wav, stereo, 16000)
    register_fake("hi")
    result = analyze(VerificationRequest(wav, "hi", backend="ci-fake"))
    assert result.audio_channels == 2
    assert result.audio_transforms == ("average_channels",)
    assert result.audio_duration_s == pytest.approx(0.2)


def test_redux_preflight_options_no_model_download(wav, capsys):
    assert main(["transcribe", str(wav), "--language", "en", "--format", "json"]) == 3
    assert "language" in capsys.readouterr().err
    assert main(["transcribe", str(wav), "--cache-dir", "cache"]) == 3
    assert "cache" in capsys.readouterr().err


def test_timestamp_overlap_and_gap_diagnostics(register_fake, wav):
    register_fake(
        "a b c", (TimedWord("a", 0.1, 0.5), TimedWord("b", 0.3, 0.4), TimedWord("c", 0.95, 1.0))
    )
    result = analyze(VerificationRequest(wav, "a b c", backend="ci-fake"))
    codes = {row["code"] for row in result.diagnostics}
    assert "word_overlap" in codes and "word_gap" in codes


@pytest.mark.parametrize("kind", ["srt", "vtt"])
def test_caption_export_rejects_backward_timestamps_without_writing(
    register_fake, wav, tmp_path, capsys, kind
):
    register_fake("first second", (TimedWord("first", 0.7, 0.8), TimedWord("second", 0.3, 0.4)))
    result = analyze(VerificationRequest(wav, "first second", backend="ci-fake"))
    output = tmp_path / f"invalid.{kind}"
    with pytest.raises(ExportError, match="non-monotonic"):
        write_captions(result, output, kind=kind)
    assert not output.exists()

    cli_output = tmp_path / f"cli-invalid.{kind}"
    assert (
        main(
            [
                "analyze",
                str(wav),
                "--backend",
                "ci-fake",
                "--text",
                "first second",
                f"--{kind}",
                str(cli_output),
            ]
        )
        == 2
    )
    assert not cli_output.exists()
    assert "non-monotonic" in capsys.readouterr().err


def test_caption_export_rejects_invalid_intervals_and_audio_bounds(register_fake, wav, tmp_path):
    for index, word in enumerate(
        (
            TimedWord("negative", -0.1, 0.2),
            TimedWord("inverted", 0.7, 0.3),
            TimedWord("past-end", 0.8, 1.1),
            TimedWord("missing", 0.2, None),
        )
    ):
        register_fake(word.text, (word,))
        result = analyze(VerificationRequest(wav, word.text, backend="ci-fake"))
        output = tmp_path / f"invalid-{index}.srt"
        with pytest.raises(ExportError, match="invalid timestamp"):
            write_captions(result, output)
        assert not output.exists()


def test_caption_export_allows_monotonic_overlaps_gaps_and_zero_duration(register_fake, wav):
    words = (
        TimedWord("a", 0.1, 0.3),
        TimedWord("b", 0.25, 0.5),
        TimedWord("c", 0.9, 0.9),
    )
    register_fake("a b c", words)
    result = analyze(VerificationRequest(wav, "a b c", backend="ci-fake"))
    output = wav.with_suffix(".srt")
    write_captions(result, output, max_chars=1)
    cues = [line for line in output.read_text().splitlines() if " --> " in line]
    assert cues == [
        "00:00:00,100 --> 00:00:00,300",
        "00:00:00,250 --> 00:00:00,500",
        "00:00:00,900 --> 00:00:00,900",
    ]
    assert all(line.split(" --> ")[1] >= line.split(" --> ")[0] for line in cues)


def test_caption_edge_cases_preserve_text_and_hour_rollover(register_fake, wav, tmp_path):
    original_words = (
        TimedWord("！？", 0.1, 0.2),
        TimedWord("你好", 0.2, 0.2),
        TimedWord("Supercalifragilisticexpialidocious", 0.2, 0.3),
        TimedWord("Done.", 0.3, 0.4),
        TimedWord("Next?", 0.4, 0.5),
    )
    register_fake("！？ 你好 Supercalifragilisticexpialidocious Done. Next?", original_words)
    result = analyze(VerificationRequest(wav, "caption text", backend="ci-fake"))
    shifted_words = (
        TimedWord("！？", 3599.9, 3600.0),
        TimedWord("你好", 3600.0, 3600.0),
        TimedWord("Supercalifragilisticexpialidocious", 3600.0, 3600.1),
        TimedWord("Done.", 3600.2, 3600.4),
        TimedWord("Next?", 3600.4, 3600.5),
    )
    result = replace(
        result,
        audio_duration_s=3601.0,
        transcription=replace(result.transcription, words=shifted_words),
    )

    vtt = tmp_path / "unicode.vtt"
    write_captions(result, vtt, kind="vtt", max_chars=100)
    content = vtt.read_text(encoding="utf-8")
    assert "00:59:59.900 --> 01:00:00.500" in content
    assert "！？ 你好 Supercalifragilisticexpialidocious Done. Next?" in content

    srt = tmp_path / "long-word.srt"
    write_captions(result, srt, max_chars=5)
    cues = srt.read_text(encoding="utf-8")
    assert "！？ 你好" in cues
    assert "\nSupercalifragilisticexpialidocious\n" in cues
    assert "Done." in cues and "Next?" in cues


def test_one_shot_preflight_rejects_bad_inputs_before_backend_creation(
    register_fake, wav, tmp_path
):
    instances = register_fake("hello")
    missing = tmp_path / "missing.wav"
    with pytest.raises(InvalidAudioError):
        analyze(VerificationRequest(missing, "hello", backend="ci-fake"))
    with pytest.raises(InvalidAudioError):
        transcribe(missing, backend="ci-fake")
    with pytest.raises(InvalidAudioError):
        transcribe_report(missing, backend="ci-fake")

    empty = tmp_path / "empty.wav"
    sf.write(empty, np.array([], dtype=np.float64), 24000)
    with pytest.raises(InvalidAudioError):
        transcribe(empty, backend="ci-fake")
    with pytest.raises(InvalidReferenceError):
        analyze(VerificationRequest(wav, "   ", backend="ci-fake"))
    with pytest.raises(InvalidReferenceError):
        analyze(VerificationRequest(wav, None, backend="ci-fake"))
    with pytest.raises(InvalidThresholdPolicyError):
        analyze(
            VerificationRequest(
                wav,
                "hello",
                backend="ci-fake",
                thresholds=ThresholdPolicy(max_wer=-1),
            )
        )
    assert instances == []


def test_cli_invalid_audio_does_not_initialize_backend(register_fake, tmp_path, capsys):
    instances = register_fake("hello")
    missing = tmp_path / "missing.wav"
    assert main(["transcribe", str(missing), "--backend", "ci-fake"]) == 2
    assert instances == []
    assert "Expected a readable WAV" in capsys.readouterr().err


def test_redux_context_manager_closes_after_enter_failure(monkeypatch):
    import sys
    import types

    from speechscope.backends.redux import ReduxTranscriber
    from speechscope.errors import BackendUnavailableError

    class BrokenManager:
        closed_with = None

        def __enter__(self):
            raise RuntimeError("context setup failed")

        def __exit__(self, exc_type, exc, traceback):
            self.closed_with = (exc_type, exc)

    manager = BrokenManager()
    sdk = types.ModuleType("moondream")
    sdk.photon = lambda *args, **kwargs: manager
    monkeypatch.setitem(sys.modules, "moondream", sdk)

    with pytest.raises(BackendUnavailableError, match="context setup failed"):
        ReduxTranscriber(device="cpu")
    assert manager.closed_with[0] is RuntimeError
    assert str(manager.closed_with[1]) == "context setup failed"


def install_contract_backend(monkeypatch, result):
    from speechscope import registry

    class Provider:
        closed = False

        def info(self):
            return BackendInfo("ci-contract", "test", True, True, False, ("cpu",))

        def transcribe(self, audio, *, language=None):
            return result

        def close(self):
            self.closed = True

    provider = Provider()
    monkeypatch.setitem(registry._FACTORIES, "ci-contract", lambda **kwargs: provider)
    return provider


@pytest.mark.parametrize(
    "result",
    [
        object(),
        TranscriptionResult(42, (), None, "ci-contract", None, 1.0),
        TranscriptionResult("hello", "hello", None, "ci-contract", None, 1.0),
        TranscriptionResult("hello", (object(),), None, "ci-contract", None, 1.0),
        TranscriptionResult("hello", (TimedWord("", 0.1, 0.2),), None, "ci-contract", None, 1.0),
    ],
    ids=[
        "wrong-result-type",
        "non-string-text",
        "non-sequence-words",
        "wrong-word-type",
        "empty-word",
    ],
)
def test_malformed_provider_contract_raises_controlled_error(monkeypatch, wav, result):
    provider = install_contract_backend(monkeypatch, result)
    with pytest.raises(TranscriptionError):
        transcribe(wav, backend="ci-contract")
    assert provider.closed


@pytest.mark.parametrize("confidence", [True, 1 + 0j, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_provider_confidence_raises_transcription_error(monkeypatch, wav, confidence):
    result = TranscriptionResult(
        "hello", (TimedWord("hello", 0.1, 0.4, confidence),), None, "ci-contract", None, 1.0
    )
    install_contract_backend(monkeypatch, result)
    with pytest.raises(TranscriptionError, match="confidence"):
        transcribe(wav, backend="ci-contract")


def test_provider_numeric_scalars_normalize_and_bad_timing_is_diagnosed(monkeypatch, wav):
    result = TranscriptionResult(
        "hello world",
        (
            TimedWord("hello", "not-a-time", np.float32(0.4), np.float32(0.75)),
            TimedWord("world", np.float32(0.5), np.float32(0.8), np.float32(0.5)),
        ),
        None,
        "ci-contract",
        None,
        np.float32(1.0),
    )
    install_contract_backend(monkeypatch, result)
    report = analyze(VerificationRequest(wav, "hello world", backend="ci-contract"))
    assert report.metrics.invalid_timestamp_count == 1
    assert report.transcription.words[0].start_s is None
    assert report.transcription.words[0].end_s is None
    assert type(report.transcription.words[1].start_s) is float
    assert type(report.transcription.words[1].end_s) is float
    assert type(report.transcription.words[1].confidence) is float
    schema = json.loads(
        (
            Path(__file__).parents[1] / "speechscope" / "schemas" / "report-v1.schema.json"
        ).read_text()
    )
    validate(report.to_dict(), schema)


def test_plugin_factory_exception_is_backend_unavailable(monkeypatch, wav):
    from speechscope import registry

    def factory(**kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setitem(registry._FACTORIES, "ci-factory-failure", factory)
    with pytest.raises(BackendUnavailableError, match="factory failed"):
        transcribe(wav, backend="ci-factory-failure")


def test_alignment_resource_error_is_a_controlled_cli_input_error(monkeypatch, capsys):
    def raise_alignment_limit(*args, **kwargs):
        raise AlignmentTooLargeError("test alignment limit")

    monkeypatch.setattr("speechscope.cli.analyze", raise_alignment_limit)
    assert main(["analyze", "unused.wav", "--text", "hello"]) == 2
    assert "test alignment limit" in capsys.readouterr().err


def test_report_timing_excludes_audio_loading_and_backend_initialization(
    register_fake, wav, monkeypatch
):
    register_fake("hello", (TimedWord("hello", 0.1, 0.4),))
    ticks = iter((100.0, 102.5, 200.0, 204.0))
    monkeypatch.setattr("speechscope.api.perf_counter", lambda: next(ticks))

    analyzed = analyze(VerificationRequest(wav, "hello", backend="ci-fake"))
    transcribed = transcribe_report(wav, backend="ci-fake")

    assert analyzed.processing_time_s == 2.5
    assert analyzed.realtime_factor == 2.5 / analyzed.audio_duration_s
    assert transcribed.processing_time_s == 4.0
    assert transcribed.realtime_factor == 4.0 / transcribed.audio_duration_s


def test_redux_doctor_is_metadata_preflight_not_model_readiness(monkeypatch, capsys):
    info = BackendInfo("redux", "2.6.1", True, False, False, ("cpu",))
    monkeypatch.setattr("speechscope.cli.backend_info", lambda name: info)

    assert main(["doctor", "--backend", "redux"]) == 0
    output = capsys.readouterr().out
    assert "metadata-only preflight" in output
    assert "not model/device readiness" in output
    assert "Photon is not initialized" in output
    assert "weights are not accessed" in output


def test_redux_doctor_rejects_sdk_below_supported_floor(monkeypatch, capsys):
    info = BackendInfo("redux", "2.0.0", True, False, False, ("cpu",))
    monkeypatch.setattr("speechscope.cli.backend_info", lambda name: info)

    assert main(["doctor", "--backend", "redux"]) == 3
    assert "does not satisfy Redux's >=2.4.1 minimum" in capsys.readouterr().err
