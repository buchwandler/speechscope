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
    AlignmentTooLargeError,
    BackendNotFoundError,
    BackendUnavailableError,
    ExportError,
    InvalidAudioError,
    InvalidReferenceError,
    InvalidThresholdPolicyError,
    ModelCatalogError,
    ModelError,
    ModelIntegrityError,
    ModelNotFoundError,
    ModelNotInstalledError,
    ModelNotManagedError,
    SpeechScopeError,
    TranscriptionError,
    UnsupportedBackendOptionError,
)
from .models import ModelCatalogClient, ModelManager, ModelStore
from .registry import available_backends, backend_info, register_backend
from .types import (
    AlignedWord,
    AudioInput,
    BackendAudioInput,
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
    "AlignmentTooLargeError",
    "AudioInput",
    "BackendAudioInput",
    "BackendInfo",
    "BackendNotFoundError",
    "BackendUnavailableError",
    "ExportError",
    "InvalidAudioError",
    "InvalidReferenceError",
    "InvalidThresholdPolicyError",
    "ModelCatalogClient",
    "ModelCatalogError",
    "ModelError",
    "ModelIntegrityError",
    "ModelManager",
    "ModelNotFoundError",
    "ModelNotInstalledError",
    "ModelNotManagedError",
    "ModelStore",
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
