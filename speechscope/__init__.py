"""SpeechScope public API: transcript, alignment, and text-quality analysis."""

try:
    from ._version import __version__
except ImportError:  # source checkout, before SCM has generated version module
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("speechscope")
    except PackageNotFoundError:
        __version__ = "0.1.dev0"

from .api import Verifier, analyze, transcribe, verify
from .backends.protocol import SpeechTranscriber
from .errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    ExportError,
    InvalidAudioError,
    InvalidReferenceError,
    InvalidThresholdPolicyError,
    SpeechScopeError,
    TranscriptionError,
    UnsupportedBackendOptionError,
)
from .registry import available_backends, backend_info, register_backend
from .types import (
    AlignedWord,
    AudioInput,
    BackendInfo,
    ThresholdPolicy,
    TimedWord,
    TranscriptionResult,
    VerificationMetrics,
    VerificationRequest,
    VerificationResult,
)

__all__ = [
    "AlignedWord",
    "AudioInput",
    "BackendInfo",
    "BackendNotFoundError",
    "BackendUnavailableError",
    "ExportError",
    "InvalidAudioError",
    "InvalidReferenceError",
    "InvalidThresholdPolicyError",
    "SpeechScopeError",
    "SpeechTranscriber",
    "ThresholdPolicy",
    "TimedWord",
    "TranscriptionError",
    "TranscriptionResult",
    "UnsupportedBackendOptionError",
    "VerificationMetrics",
    "VerificationRequest",
    "VerificationResult",
    "Verifier",
    "__version__",
    "analyze",
    "available_backends",
    "backend_info",
    "register_backend",
    "transcribe",
    "verify",
]
