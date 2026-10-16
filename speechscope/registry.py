"""Explicit + entry-point backend registry with deterministic collision policy."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable

from .errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    SpeechScopeError,
)
from .types import BackendInfo

# Explicit registration overrides neither another explicit registration nor a plugin.
_FACTORIES: dict[str, Callable[..., object]] = {}

_BUILTIN_NAMES = ("redux", "whispercpp")


def register_backend(name: str, factory: Callable[..., object]) -> None:
    """Register a local backend factory without initializing it.

    Factories receive ``device``, ``model`` and ``cache_dir`` keyword options
    and must return an object implementing ``info()``, ``transcribe(audio,
    language=None)`` and ``close()``. Duplicate names and reserved built-in names are
    rejected; use an entry point for an installed plugin.
    """
    if (
        not name
        or name in _BUILTIN_NAMES
        or name in _FACTORIES
        or any(ep.name == name for ep in _entries())
    ):
        raise ValueError(f"Backend name already reserved: {name!r}")
    _FACTORIES[name] = factory


def _entries():
    return tuple(importlib.metadata.entry_points(group="speechscope.backends"))


def _builtin_module(name: str):
    return importlib.import_module(f".backends.{name}", package=__package__)


def _names() -> tuple[str, ...]:
    names = [*_BUILTIN_NAMES, *_FACTORIES, *(ep.name for ep in _entries())]
    if len(names) != len(set(names)):
        raise BackendUnavailableError(
            "Duplicate backend entry-point names; remove conflicting plugins"
        )
    return tuple(sorted(names))


def available_backends() -> tuple[BackendInfo, ...]:
    """Return discoverable backend metadata without opening models or weights."""
    # Built-in adapters expose static info(); discovery never probes executables.
    return tuple(
        _builtin_module(name).info()
        if name in _BUILTIN_NAMES
        else BackendInfo(name, None, False, False, False, ())
        for name in _names()
    )


def backend_info(name: str) -> BackendInfo:
    for entry in available_backends():
        if entry.name == name:
            return entry
    raise BackendNotFoundError(f"Unknown backend {name!r}. Available: {', '.join(_names())}")


def open_backend(name: str, *, device: str, model: str | None, cache_dir=None):
    # Detect entry-point collisions with reserved built-ins before dispatch.
    _names()
    if name in _BUILTIN_NAMES:
        try:
            factory = _builtin_module(name).create_backend
            return factory(device=device, model=model, cache_dir=cache_dir)
        except SpeechScopeError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(f"Backend {name!r} factory failed: {exc}") from exc
    if name in _FACTORIES:
        try:
            return _FACTORIES[name](device=device, model=model, cache_dir=cache_dir)
        except SpeechScopeError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(f"Backend {name!r} factory failed: {exc}") from exc
    for entry in _entries():
        if entry.name == name:
            try:
                return entry.load()(device=device, model=model, cache_dir=cache_dir)
            except SpeechScopeError:
                raise
            except ImportError as exc:
                raise BackendUnavailableError(
                    f"Backend plugin {name!r} missing dependencies"
                ) from exc
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Backend plugin {name!r} factory failed: {exc}"
                ) from exc
    raise BackendNotFoundError(f"Unknown backend {name!r}")
