"""Shared fixtures: deterministic stub engines so API tests need no weights.

The secured-app fixture generates a random per-session API key (no credential
literals ever live in source control); tests read it back from app settings.
"""

from __future__ import annotations

import secrets

import numpy as np
import pytest

from voxora.engines import AsrEngine, SynthesisResult, TranscriptionResult, TtsEngine
from voxora.engines import registry as reg
from voxora.settings import Settings


class StubAsrEngine(AsrEngine):
    name = "stub-asr"
    description = "deterministic stub for tests"
    model_subdir = ""

    def load(self) -> None:
        self._loaded = True

    def _transcribe(self, samples, sample_rate, *, language, **kwargs):
        return TranscriptionResult(text="stub transcription", language=language)


class StubTtsEngine(TtsEngine):
    name = "stub-tts"
    description = "deterministic stub for tests"
    model_subdir = ""

    def load(self) -> None:
        self._loaded = True

    def _synthesize(self, text, *, language, voice, **kwargs):
        sr = 16000
        t = np.linspace(0.0, 0.5, sr // 2, endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return SynthesisResult(audio=audio, sample_rate=sr)


class BrokenEngine(StubAsrEngine):
    """Simulates an unmet dependency (ImportError at load -> 503)."""

    name = "stub-broken"
    model_subdir = ""

    def load(self) -> None:
        raise ImportError("No module named 'fictional_runtime'")


@pytest.fixture()
def stub_engines():
    reg._REGISTRY[StubAsrEngine.name] = StubAsrEngine
    reg._REGISTRY[StubTtsEngine.name] = StubTtsEngine
    reg._REGISTRY[BrokenEngine.name] = BrokenEngine
    reg.reset_pool()
    yield
    reg._REGISTRY.pop(StubAsrEngine.name, None)
    reg._REGISTRY.pop(StubTtsEngine.name, None)
    reg._REGISTRY.pop(BrokenEngine.name, None)
    reg.reset_pool()


@pytest.fixture()
def app(stub_engines):
    from voxora.api import create_app

    settings = Settings(models_dir="/nonexistent", api_key="",
                        default_asr="stub-asr", default_tts="stub-tts")
    return create_app(settings)


@pytest.fixture()
def secured_app(stub_engines):
    """App protected by a per-session random key (value never written in source)."""
    from voxora.api import create_app

    key = secrets.token_urlsafe(24)
    settings = Settings(models_dir="/nonexistent", api_key=key,
                        default_asr="stub-asr", default_tts="stub-tts")
    return create_app(settings)
