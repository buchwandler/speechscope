import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from jsonschema import validate

from speechscope import (
    ThresholdPolicy,
    TimedWord,
    VerificationRequest,
    Verifier,
    analyze,
    available_backends,
    register_backend,
)
from speechscope.cli import main
from speechscope.errors import ExportError, InvalidAudioError
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
