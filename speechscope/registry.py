"""Explicit + entry-point backend registry with deterministic collision policy."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable

from .errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    SpeechScopeError,
    UnsupportedBackendOptionError,
)
from .types import BackendInfo

# Explicit registration overrides neither another explicit registration nor a plugin.
_FACTORIES: dict[str, Callable[..., object]] = {}


def register_backend(name: str, factory: Callable[..., object]) -> None:
    if (
        not name
        or name == "redux"
        or name in _FACTORIES
        or any(ep.name == name for ep in _entries())
    ):
        raise ValueError(f"Backend name already reserved: {name!r}")
    _FACTORIES[name] = factory


def _entries():
    return tuple(importlib.metadata.entry_points(group="speechscope.backends"))


def _names() -> tuple[str, ...]:
    names = ["redux", *_FACTORIES, *(ep.name for ep in _entries())]
    if len(names) != len(set(names)):
        raise BackendUnavailableError(
            "Duplicate backend entry-point names; remove conflicting plugins"
        )
    return tuple(sorted(names))


def available_backends() -> tuple[BackendInfo, ...]:
    # Plugin capability inspection may require loading plugin metadata; never open a model.
    from .backends.redux import info

    return tuple(
        info() if name == "redux" else BackendInfo(name, None, False, False, False, ())
        for name in _names()
    )


def backend_info(name: str) -> BackendInfo:
    for entry in available_backends():
        if entry.name == name:
            return entry
    raise BackendNotFoundError(f"Unknown backend {name!r}. Available: {', '.join(_names())}")


def open_backend(name: str, *, device: str, model: str | None, cache_dir=None):
    if name == "redux":
        from .backends.redux import DEFAULT_MODEL, ReduxTranscriber

        if model is not None and model != DEFAULT_MODEL:
            raise UnsupportedBackendOptionError(
                f"Redux supports only model {DEFAULT_MODEL!r}, not {model!r}"
            )
        if device not in ("cpu", "cuda", "mps", "auto"):
            raise UnsupportedBackendOptionError(f"Redux unsupported device {device!r}")
        return ReduxTranscriber(device=device, model=model, cache_dir=cache_dir)
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
