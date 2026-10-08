"""Shell interface. JSON mode emits exactly one object and no other stdout data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .api import analyze, transcribe_report
from .errors import (
    AlignmentTooLargeError,
    BackendNotFoundError,
    BackendUnavailableError,
    ExportError,
    InvalidAudioError,
    InvalidReferenceError,
    InvalidThresholdPolicyError,
    TranscriptionError,
    UnsupportedBackendOptionError,
)
from .evaluation import parse_policy
from .exports import write_captions, write_words
from .registry import available_backends, backend_info
from .timing import caption_timestamp_error
from .types import VerificationRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speechscope")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for cmd in ("analyze", "verify", "transcribe"):
        p = commands.add_parser(
            cmd,
            help="Analyze with reference text"
            if cmd != "transcribe"
            else "Timestamped speech transcription",
        )
        p.add_argument("audio", type=Path)
        if cmd != "transcribe":
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument("--text")
            group.add_argument("--text-file", type=Path)
            p.add_argument("--thresholds", type=Path)
            p.add_argument(
                "--normalization",
                choices=("basic-v1", "strict-v1", "readio-compat-v1"),
                default="basic-v1",
            )
        p.add_argument("--backend", default="redux")
        p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
        p.add_argument("--model")
        p.add_argument("--cache-dir", type=Path)
        p.add_argument("--language")
        p.add_argument("--format", choices=("json", "human"), default="human")
        p.add_argument("--json", type=Path)
        p.add_argument("--words", type=Path)
        p.add_argument("--srt", type=Path)
        p.add_argument("--vtt", type=Path)
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--quiet", action="store_true")
    backends = commands.add_parser("backends")
    sub = backends.add_subparsers(dest="action", required=True)
    l = sub.add_parser("list")
    l.add_argument("--format", choices=("human", "json"), default="human")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("name")
    inspect.add_argument("--format", choices=("human", "json"), default="human")
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--backend", default="redux")
    metrics = commands.add_parser("metrics")
    metrics.add_argument("action", choices=("list",))
    metrics.add_argument("--format", choices=("json", "human"), default="human")
    return parser


def _emit(data, fmt):
    if fmt == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(data)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "backends":
            if args.action == "list":
                data = [asdict(info) for info in available_backends()]
            else:
                data = asdict(backend_info(args.name))
            if args.format == "json":
                _emit(data, "json")
            else:
                for item in data if isinstance(data, list) else [data]:
                    print(item)
            return 0
        if args.command == "doctor":
            info = backend_info(args.backend)
            if info.name == "redux":
                if info.version is None:
                    raise BackendUnavailableError(
                        "Install Redux with: pip install 'speechscope[redux]'"
                    )
                match = re.match(r"^(\d+)\.(\d+)\.(\d+)", info.version)
                version = tuple(map(int, match.groups())) if match else ()
                if version < (2, 4, 1):
                    raise BackendUnavailableError(
                        f"Moondream {info.version!r} does not satisfy Redux's >=2.4.1 minimum; "
                        "upgrade with: pip install --upgrade 'speechscope[redux]'"
                    )
                print(
                    f"Backend redux: dependency available (Moondream {info.version}); "
                    "metadata-only preflight, not model/device readiness. "
                    "Photon is not initialized and model weights are not accessed."
                )
            else:
                print(
                    f"Backend {info.name}: dependency metadata preflight passed; "
                    "model readiness is not tested."
                )
            return 0
        if args.command == "metrics":
            # Audio-based quality metrics belong to audiosig and are not part of v0.1.
            names = ["wer", "cer", "mer", "wil", "wip"]
            _emit(names if args.format == "json" else "\n".join(names), args.format)
            return 0
        # Check capabilities before expensive model initialization.
        info = backend_info(args.backend)
        if args.language and not info.supports_language_hint and info.name == "redux":
            raise UnsupportedBackendOptionError(
                "Redux detects language automatically; --language is unsupported"
            )
        if args.cache_dir and info.name == "redux":
            raise UnsupportedBackendOptionError(
                "Redux does not expose --cache-dir; configure model cache externally"
            )
        if args.command in ("analyze", "verify"):
            try:
                text = (
                    args.text
                    if args.text is not None
                    else args.text_file.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError) as exc:
                raise InvalidReferenceError(f"Could not read text file: {exc}") from exc
            policy = None
            if args.thresholds:
                try:
                    policy = parse_policy(json.loads(args.thresholds.read_text(encoding="utf-8")))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise InvalidThresholdPolicyError(str(exc)) from exc
            result = analyze(
                VerificationRequest(
                    audio=args.audio,
                    reference_text=text,
                    backend=args.backend,
                    language=args.language,
                    device=args.device,
                    model=args.model,
                    cache_dir=args.cache_dir,
                    normalization=args.normalization,
                    thresholds=policy,
                )
            )
        else:
            result = transcribe_report(
                args.audio,
                backend=args.backend,
                language=args.language,
                device=args.device,
                model=args.model,
                cache_dir=args.cache_dir,
            )
        # Validate all export preconditions before writing any result artifact.
        if args.srt or args.vtt:
            error = caption_timestamp_error(result.transcription.words, result.audio_duration_s)
            if error:
                raise ExportError(error)
        if args.words:
            write_words(result, args.words)
        if args.srt:
            write_captions(result, args.srt, kind="srt")
        if args.vtt:
            write_captions(result, args.vtt, kind="vtt")
        if args.json:
            result.write_json(args.json)
        if args.format == "json":
            print(result.to_json(), end="")
        elif not args.quiet:
            print(
                f"Backend: {result.transcription.backend} ({result.transcription.model or 'default'})"
            )
            print(f"Duration: {result.audio_duration_s:.3f}s")
            print(f"Transcript: {result.transcription.text}")
            if result.metrics:
                m = result.metrics
                print(
                    f"WER: {m.wer:.4f}  CER: {m.cer:.4f}  MER: {m.mer:.4f}  WIL: {m.wil:.4f}  WIP: {m.wip:.4f}"
                )
                print(
                    f"Words: matches={m.word_matches}, substitutions={m.word_substitutions}, deletions={m.word_deletions}, insertions={m.word_insertions}"
                )
                print(f"Reference timestamp coverage: {m.reference_timestamp_coverage_ratio:.1%}")
            print(f"Status: {result.evaluation['status']}")
        if args.verbose and result.diagnostics:
            print(json.dumps(result.diagnostics, indent=2), file=sys.stderr)
        return 1 if result.evaluation["status"] == "fail" else 0
    except (
        AlignmentTooLargeError,
        InvalidAudioError,
        InvalidReferenceError,
        InvalidThresholdPolicyError,
        ExportError,
    ) as exc:
        print(f"speechscope: {exc}", file=sys.stderr)
        return 2
    except (BackendNotFoundError, BackendUnavailableError, UnsupportedBackendOptionError) as exc:
        print(f"speechscope: {exc}", file=sys.stderr)
        return 3
    except TranscriptionError as exc:
        print(f"speechscope: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001
        print(f"speechscope: internal error: {exc}", file=sys.stderr)
        return 5
