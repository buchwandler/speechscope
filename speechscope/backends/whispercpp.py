"""External whisper.cpp CLI adapter with canonical WAV provider input."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from ..backend_audio import materialize_pcm16_wav
from ..errors import BackendUnavailableError, TranscriptionError, UnsupportedBackendOptionError
from ..types import AudioInput, BackendAudioInput, BackendInfo, TranscriptionResult

BACKEND_NAME = "whispercpp"
DEFAULT_EXECUTABLE = "whisper-cli"
TARGET_SAMPLE_RATE = 16_000
WHISPERCPP_SAMPLE_RATE = TARGET_SAMPLE_RATE
WHISPERCPP_WAV_FORMAT = "WAV"
WHISPERCPP_WAV_SUBTYPE = "PCM_16"
WHISPERCPP_RESAMPLE_FILTER_WIDTH = 32
WHISPERCPP_RESAMPLE_ROLLOFF = 0.945
WHISPERCPP_RESAMPLE_LENGTH_MODE = "round"
EXECUTABLE_ENV = "SPEECHSCOPE_WHISPERCPP_BIN"
MODEL_ENV = "SPEECHSCOPE_WHISPERCPP_MODEL"
_VERSION_RE = re.compile(r"whisper\.cpp version:\s*(.+)", re.IGNORECASE)


def info() -> BackendInfo:
    """Return static adapter capabilities without executing a process or loading a model."""
    return BackendInfo(
        name=BACKEND_NAME,
        version=None,
        supports_word_timestamps=False,
        supports_segment_timestamps=False,
        supports_language_hint=True,
        supported_devices=("cpu", "auto"),
    )


def resolve_executable(executable: str | None = None) -> str:
    """Resolve an executable selector to a runnable absolute path."""
    selected = executable or os.environ.get(EXECUTABLE_ENV) or DEFAULT_EXECUTABLE
    found = shutil.which(selected)
    if found is None or not os.access(found, os.X_OK):
        raise BackendUnavailableError(
            f"whisper-cli executable {selected!r} was not found or is not runnable; "
            f"install whisper.cpp or set {EXECUTABLE_ENV}"
        )
    return str(Path(found).resolve())


def probe_executable(executable: str | None = None) -> tuple[str, str | None]:
    """Check the CLI version/help contract without requiring a model."""
    path = resolve_executable(executable)
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendUnavailableError(f"Could not run whisper-cli --version: {exc}") from exc

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode == 0:
        match = _VERSION_RE.search(output)
        return path, match.group(1).strip() if match else None

    try:
        help_result = subprocess.run(
            [path, "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendUnavailableError(f"Could not run whisper-cli --help: {exc}") from exc
    if help_result.returncode == 0:
        return path, None
    detail = (help_result.stderr or help_result.stdout or output).strip()
    raise BackendUnavailableError(
        f"whisper-cli failed its --version and --help checks (exit {help_result.returncode}): "
        f"{detail}"
    )


def _model_path(model: str | None, *, cache_dir=None) -> str:
    selected = model if model is not None else os.environ.get(MODEL_ENV)
    if not isinstance(selected, str) or not selected.strip():
        raise BackendUnavailableError(
            f"Provide a local whisper.cpp model with --model or set {MODEL_ENV}"
        )
    if selected.startswith("whispercpp:"):
        from ..models import ModelManager

        return str(ModelManager(cache_dir=cache_dir, packaged_catalog=True).path(selected))
    path = Path(selected).expanduser()
    if not path.is_file():
        raise BackendUnavailableError(f"whisper.cpp model file does not exist: {path}")
    try:
        with path.open("rb") as model_file:
            model_file.read(1)
    except OSError as exc:
        raise BackendUnavailableError(f"whisper.cpp model file is not readable: {path}") from exc
    return str(path.resolve())


def _parse_output(path: Path, audio: AudioInput, model: str, device: str) -> TranscriptionResult:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TranscriptionError("whisper-cli did not create its JSON result file") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TranscriptionError(f"Could not read whisper-cli JSON result: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TranscriptionError("whisper-cli JSON result must be an object")

    result = data.get("result", {})
    if not isinstance(result, Mapping):
        raise TranscriptionError("whisper-cli JSON result field must be an object")
    raw_segments = data.get("transcription", [])
    if not isinstance(raw_segments, list):
        raise TranscriptionError("whisper-cli JSON transcription field must be an array")
    text_parts: list[str] = []
    for segment in raw_segments:
        if not isinstance(segment, Mapping):
            raise TranscriptionError("whisper-cli JSON transcription entries must be objects")
        segment_text = segment.get("text")
        if not isinstance(segment_text, str):
            raise TranscriptionError("whisper-cli JSON transcription text must be a string")
        text_parts.append(segment_text)

    raw_text = result.get("text", data.get("text"))
    if raw_text is None:
        raw_text = "".join(text_parts).strip()
    if not isinstance(raw_text, str):
        raise TranscriptionError("whisper-cli JSON result text must be a string")
    language = result.get("language")
    if language is None:
        language = data.get("language")
    if language is not None and not isinstance(language, str):
        raise TranscriptionError("whisper-cli JSON language must be a string or null")

    return TranscriptionResult(
        text=raw_text,
        words=(),
        language=language,
        backend=BACKEND_NAME,
        model=model,
        duration_s=audio.duration_s,
        timing_source="unavailable",
        device=device,
    )


class WhisperCppTranscriber:
    """Run whisper.cpp as a local CLI process for each transcription call."""

    def __init__(self, *, device: str = "cpu", model: str | None = None, cache_dir=None):
        selected_model = model if model is not None else os.environ.get(MODEL_ENV)
        managed_ref = isinstance(selected_model, str) and selected_model.startswith("whispercpp:")
        if cache_dir is not None and not managed_ref:
            raise UnsupportedBackendOptionError(
                "whispercpp does not support cache_dir for explicit model paths"
            )
        if device not in ("cpu", "auto"):
            raise UnsupportedBackendOptionError(
                f"whispercpp supports only cpu and auto devices, not {device!r}"
            )
        self._device = device
        self._model = _model_path(selected_model, cache_dir=cache_dir)
        self._model_label = (
            selected_model if managed_ref and isinstance(selected_model, str) else self._model
        )
        self._executable = resolve_executable()

    def info(self) -> BackendInfo:
        return info()

    def transcribe(self, audio: AudioInput, *, language: str | None = None) -> TranscriptionResult:
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise UnsupportedBackendOptionError(
                "whispercpp language must be a nonempty language code"
            )

        with materialize_pcm16_wav(
            audio,
            sample_rate=WHISPERCPP_SAMPLE_RATE,
            filter_width=WHISPERCPP_RESAMPLE_FILTER_WIDTH,
            rolloff=WHISPERCPP_RESAMPLE_ROLLOFF,
            length_mode=WHISPERCPP_RESAMPLE_LENGTH_MODE,
        ) as prepared:
            output_prefix = prepared.path.parent / "transcription"
            output_json = output_prefix.with_suffix(".json")
            argv = [
                self._executable,
                "--model",
                self._model,
                "--file",
                str(prepared.path),
                "--output-json-full",
                "--output-file",
                str(output_prefix),
                "--no-prints",
                "--split-on-word",
            ]
            argv.extend(("--language", language if language is not None else "auto"))
            if self._device == "cpu":
                argv.append("--no-gpu")
            try:
                completed = subprocess.run(
                    argv,
                    shell=False,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as exc:
                raise BackendUnavailableError(f"Could not run whisper-cli: {exc}") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                suffix = f": {detail}" if detail else ""
                raise TranscriptionError(
                    f"whisper-cli exited with status {completed.returncode}{suffix}"
                )
            result = _parse_output(output_json, audio, self._model_label, self._device)
            return TranscriptionResult(
                text=result.text,
                words=result.words,
                language=result.language,
                backend=result.backend,
                model=result.model,
                duration_s=result.duration_s,
                timing_source=result.timing_source,
                backend_version=result.backend_version,
                device=result.device,
                backend_audio_input=BackendAudioInput(
                    kind="generated_wav_path",
                    selection="canonicalized_wav",
                    sample_rate_hz=prepared.sample_rate,
                    channels=prepared.channels,
                    sample_format=prepared.subtype,
                    transformations=prepared.transformations,
                ),
                diagnostics=prepared.diagnostics,
            )

    def close(self) -> None:
        """No persistent process or model session is held by this adapter."""


def create_backend(*, device: str = "cpu", model: str | None = None, cache_dir=None):
    return WhisperCppTranscriber(device=device, model=model, cache_dir=cache_dir)
