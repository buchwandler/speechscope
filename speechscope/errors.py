"""Typed public failure classes; outcome/threshold failures are not exceptions."""


class SpeechScopeError(Exception):
    """Base class for expected failures."""


class InvalidAudioError(SpeechScopeError):
    pass


class InvalidReferenceError(SpeechScopeError):
    pass


class BackendNotFoundError(SpeechScopeError):
    pass


class BackendUnavailableError(SpeechScopeError):
    pass


class UnsupportedBackendOptionError(SpeechScopeError):
    pass


class TranscriptionError(SpeechScopeError):
    pass


class InvalidThresholdPolicyError(SpeechScopeError):
    pass


class ExportError(SpeechScopeError):
    pass
