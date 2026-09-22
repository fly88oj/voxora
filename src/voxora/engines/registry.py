"""Engine registry + lazy pool.

``register`` is called from ``voxora.engines.__init__`` for every built-in
engine; tests register stub engines the same way, so the API contract is
exercisable without any model weights.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import TypeVar

from .base import BaseEngine, EngineError

_EngineT = TypeVar("_EngineT", bound=type[BaseEngine])

_REGISTRY: dict[str, type[BaseEngine]] = {}
_POOL: dict[str, BaseEngine] = {}
_POOL_LOCK = threading.Lock()


def register(cls: _EngineT) -> _EngineT:
    """Class decorator: add an engine class to the global registry."""
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate engine name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def engine_classes() -> dict[str, type[BaseEngine]]:
    return dict(_REGISTRY)


def create_engine(name: str, models_dir: str, *, num_threads: int = 0) -> BaseEngine:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise EngineError(f"unknown engine '{name}'. Known engines: {known}") from None
    return cls(models_dir, num_threads=num_threads)


def get_engine(name: str, models_dir: str, *, num_threads: int = 0) -> BaseEngine:
    """Return a process-wide singleton instance (thread-safe, lazily loaded)."""
    with _POOL_LOCK:
        eng = _POOL.get(name)
        if eng is None:
            eng = create_engine(name, models_dir, num_threads=num_threads)
            _POOL[name] = eng
        return eng


def reset_pool() -> None:
    """Drop cached instances (used by tests)."""
    with _POOL_LOCK:
        _POOL.clear()


def iter_engines(models_dir: str, *, num_threads: int = 0) -> Iterator[BaseEngine]:
    for name in sorted(_REGISTRY):
        yield create_engine(name, models_dir, num_threads=num_threads)
