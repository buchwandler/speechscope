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
    ModelError,
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
    models = commands.add_parser("models", help="Discover and manage catalogued models")
    model_sub = models.add_subparsers(dest="models_action", required=True)

    def add_model_options(action, *, backend=False, force=False):
        if backend:
            action.add_argument("--backend")
        action.add_argument("--catalog", help="Local catalog path or HTTPS catalog URL")
        action.add_argument("--cache-dir", type=Path, help="SpeechScope model/catalog cache root")
        action.add_argument("--offline", action="store_true", help="Use only a cached catalog")
        action.add_argument("--refresh", action="store_true", help="Refresh the remote catalog")
        action.add_argument("--format", choices=("human", "json"), default="human")
        if force:
            action.add_argument(
                "--force", action="store_true", help="Replace a verified managed install"
            )

    list_models = model_sub.add_parser("list", help="List catalog models")
    add_model_options(list_models, backend=True)
    show_model = model_sub.add_parser("show", help="Show catalog metadata for a model")
    show_model.add_argument("ref")
    add_model_options(show_model, backend=True)
    install_model = model_sub.add_parser("install", help="Download and verify a managed model")
    install_model.add_argument("ref")
    add_model_options(install_model, backend=True, force=True)
    installed_models = model_sub.add_parser("installed", help="List SpeechScope-managed models")
    add_model_options(installed_models)
    for action_name, help_text in (
        ("path", "Print the primary installed model path"),
        ("verify", "Verify installed model bytes"),
        ("remove", "Remove a SpeechScope-managed model"),
    ):
        action = model_sub.add_parser(action_name, help=help_text)
        action.add_argument("ref")
        add_model_options(action, backend=True)
    return parser


def _emit(data, fmt):
    if fmt == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(data)


def _run_models(args) -> int:
    from .models import ModelManager

    manager = ModelManager(
        catalog=args.catalog,
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh=args.refresh,
    )
    action = args.models_action
    if action == "list":
        models = manager.list(backend=args.backend)
        if args.format == "json":
            _emit(models, "json")
        elif not models:
            print("No models found.")
        else:
            for model in models:
                mode = model["acquisition"]["mode"]
                print(f"{model['ref']}  {model['format']}  {mode}  {model['description']}")
        return 0
    if action == "show":
        model = manager.show(args.ref, backend=args.backend)
        if args.format == "json":
            _emit(model, "json")
        else:
            source = model["source"]
            print(f"Reference: {model['ref']}")
            print(f"Backend: {model['backend']}")
            print(f"Format: {model['format']}")
            print(f"Acquisition: {model['acquisition']['mode']}")
            print(f"Source: {source['repository']}@{source['revision']}")
            print(f"License: {source['license']}")
            print(f"Description: {model['description']}")
            for artifact in model["artifacts"]:
                role = "primary" if artifact["primary"] else artifact["role"]
                print(f"Artifact: {artifact['path']} ({role})")
        return 0
    if action == "install":
        path = manager.install(args.ref, backend=args.backend, force=args.force)
        if args.format == "json":
            _emit(
                {"ref": manager.show(args.ref, backend=args.backend)["ref"], "path": str(path)},
                "json",
            )
        else:
            print(f"Installed {args.ref} at {path}")
        return 0
    if action == "installed":
        installed = manager.installed()
        if args.format == "json":
            _emit(installed, "json")
        elif not installed:
            print("No SpeechScope-managed models installed.")
        else:
            for item in installed:
                primary = next(artifact for artifact in item["artifacts"] if artifact["primary"])
                print(
                    f"{item['ref']}  {manager.store.models_dir / item['backend'] / item['id'] / primary['path']}"
                )
        return 0
    if action == "path":
        path = manager.path(args.ref, backend=args.backend)
        if args.format == "json":
            _emit({"ref": args.ref, "path": str(path)}, "json")
        else:
            print(path)
        return 0
    if action == "verify":
        result = manager.verify(args.ref, backend=args.backend)
        if args.format == "json":
            _emit(result, "json")
        else:
            print(f"Verified {result['ref']} ({len(result['artifacts'])} artifact(s))")
        return 0
    if action == "remove":
        model = manager.show(args.ref, backend=args.backend)
        manager.remove(args.ref, backend=args.backend)
        if args.format == "json":
            _emit({"ref": model["ref"], "removed": True}, "json")
        else:
            print(f"Removed {model['ref']}")
        return 0
    raise AssertionError(f"Unknown models action: {action}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "models":
            return _run_models(args)
        if args.command == "backends":
            if args.action == "list":
                data: list[dict[str, object]] | dict[str, object] = [
                    asdict(info) for info in available_backends()
                ]
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
            elif info.name == "whispercpp":
                from .backends.whispercpp import probe_executable

                executable, whisper_version = probe_executable()
                version_note = (
                    f" (whisper.cpp {whisper_version})"
                    if whisper_version
                    else " (version unavailable)"
                )
                print(
                    f"Backend whispercpp: executable available{version_note} at {executable}; "
                    "metadata-only preflight. Model weights are not accessed."
                )
            else:
                print(
                    f"Backend {info.name}: dependency metadata preflight passed; "
                    "model readiness is not tested."
                )
            return 0
        if args.command == "metrics":
            # Audio-based quality metrics belong to audiosig and are not part of v0.1.
            names = ["wer", "compound_wer", "cer", "mer", "wil", "wip"]
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
                    f"WER: {m.wer:.4f}  Compound-WER: {m.compound_wer:.4f}  CER: {m.cer:.4f}  MER: {m.mer:.4f}  WIL: {m.wil:.4f}  WIP: {m.wip:.4f}"
                )
                print(
                    f"Words: matches={m.word_matches}, substitutions={m.word_substitutions}, deletions={m.word_deletions}, insertions={m.word_insertions}"
                )
                print(f"Boundary variants: {m.word_boundary_match_count}")
                print(f"Reference timestamp coverage: {m.reference_timestamp_coverage_ratio:.1%}")
            print(f"Status: {result.evaluation['status']}")
        if args.verbose and result.diagnostics:
            print(json.dumps(result.diagnostics, indent=2), file=sys.stderr)
        return 1 if result.evaluation["status"] == "fail" else 0
    except ModelError as exc:
        print(f"speechscope: {exc}", file=sys.stderr)
        return 3
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
