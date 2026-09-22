"""Engine abstraction: every ASR/TTS backend behind one interface.

Engines are declared in a registry (``registry.py``) and instantiated lazily
by the API/benchmark layers. Heavy ML dependencies are imported inside
``load()`` so that a default install can serve the ONNX-only engines without
PyTorch, and missing optional dependencies surface as ``EngineNotAvailable``.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class EngineKind(str, Enum):
    ASR = "asr"
    TTS = "tts"


class EngineError(RuntimeError):
    """Engine-level failure (missing weights, failed load, inference error)."""


class EngineNotAvailable(EngineError):
    """The engine's runtime dependency is not installed / weights not present."""


@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None
    audio_duration_s: float = 0.0
    processing_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        return round(self.processing_s / self.audio_duration_s, 4) if self.audio_duration_s else None


@dataclass
class SynthesisResult:
    audio: np.ndarray  # float32 mono
    sample_rate: int
    text: str = ""
    processing_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def rtf(self) -> float:
        return round(self.processing_s / self.duration_s, 4) if self.duration_s else None


class BaseEngine(ABC):
    """A single speech engine. Subclasses set metadata and implement one modality."""

    name: str = "base"
    kind: EngineKind = EngineKind.ASR
    description: str = ""
    license_note: str = ""  # short pointer; the authoritative table is docs/MODEL_LICENSES.md
    requires_extras: tuple[str, ...] = ()  # pip extras needed, e.g. ("onnx",)
    import_probe: str = ""  # module name used to test availability without loading
    model_subdir: str = ""  # expected location under the models dir

    def __init__(self, models_dir: str | Path, *, num_threads: int = 0):
        self.models_dir = Path(models_dir)
        self.num_threads = num_threads
        self._loaded = False
        self._load_lock = threading.Lock()
        self.load_s: float | None = None

    # -- lifecycle ----------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Load once, thread-safely; map dependency/weight problems to
        ``EngineNotAvailable``/``EngineError`` so the API can answer 503."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            try:
                self.load()
            except ImportError as e:  # optional dependency missing
                raise EngineNotAvailable(
                    f"{self.name}: missing dependency ({e}); "
                    f"install extras {self.requires_extras or 'see docs'}"
                ) from e
            except FileNotFoundError as e:
                raise EngineNotAvailable(f"{self.name}: {e}") from e
            self.load_s = round(time.perf_counter() - t0, 3)
            self._loaded = True

    @abstractmethod
    def load(self) -> None:
        """Load weights. May raise ImportError/FileNotFoundError (mapped) or EngineError."""

    def weights_present(self) -> bool:
        """True if the model files appear to exist (engines without weights, e.g.
        test stubs, are considered present)."""
        if not self.model_subdir:
            return True
        return (self.models_dir / self.model_subdir).exists()

    # -- inference ----------------------------------------------------------
    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000,
                   *, language: str | None = None, **kwargs: Any) -> TranscriptionResult:
        self.ensure_loaded()
        dur = len(samples) / sample_rate
        t0 = time.perf_counter()
        result = self._transcribe(samples, sample_rate, language=language, **kwargs)
        result.audio_duration_s = round(dur, 3)
        result.processing_s = round(time.perf_counter() - t0, 3)
        return result

    @abstractmethod
    def _transcribe(self, samples: np.ndarray, sample_rate: int,
                    *, language: str | None, **kwargs: Any) -> TranscriptionResult:
        ...

    def synthesize(self, text: str, *, language: str | None = None,
                   voice: str | None = None, **kwargs: Any) -> SynthesisResult:
        self.ensure_loaded()
        t0 = time.perf_counter()
        result = self._synthesize(text, language=language, voice=voice, **kwargs)
        result.text = text
        result.processing_s = round(time.perf_counter() - t0, 3)
        return result

    @abstractmethod
    def _synthesize(self, text: str, *, language: str | None,
                    voice: str | None, **kwargs: Any) -> SynthesisResult:
        ...

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
            "license": self.license_note,
            "requires_extras": list(self.requires_extras),
            "model_subdir": self.model_subdir,
            "weights_present": self.weights_present(),
            "loaded": self._loaded,
        }


class AsrEngine(BaseEngine):
    kind = EngineKind.ASR

    def _synthesize(self, text: str, *, language: str | None,
                    voice: str | None, **kwargs: Any) -> SynthesisResult:
        raise EngineError(f"{self.name} is an ASR engine and cannot synthesize speech")


class TtsEngine(BaseEngine):
    kind = EngineKind.TTS

    def _transcribe(self, samples: np.ndarray, sample_rate: int,
                    *, language: str | None, **kwargs: Any) -> TranscriptionResult:
        raise EngineError(f"{self.name} is a TTS engine and cannot transcribe audio")
