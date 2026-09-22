"""Piper (neural VITS, ONNX) — the low-resource realtime TTS baseline."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .base import EngineError, SynthesisResult, TtsEngine
from .registry import register

# Language -> (repo-relative directory, voice stem) inside models/piper-voices/.
# Voice files are <stem>.onnx + <stem>.onnx.json (MIT-licensed voices).
DEFAULT_VOICES: dict[str, str] = {
    "en": "en/en_US/lessac/medium/en_US-lessac-medium",
    "en-us-ryan": "en/en_US/ryan/high/en_US-ryan-high",
    "zh": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium",
    "de": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    "fr": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium",
    "ru": "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium",
    "es": "es/es_MX/ald/medium/es_MX-ald-medium",
}


@register
class PiperEngine(TtsEngine):
    name = "piper"
    description = "Piper neural TTS (ONNX); 15-30x realtime on one CPU core"
    license_note = "code MIT (piper1-gpl); voices MIT (rhasspy/piper-voices)"
    requires_extras = ("onnx",)
    import_probe = "piper"
    model_subdir = "piper-voices"

    def load(self) -> None:
        # Fail fast when the dependency is missing so the API can answer 503
        # instead of erroring per request; voices load lazily per request.
        import piper  # noqa: F401 — availability check only
        self._voices: dict[str, object] = {}
        self._voices_lock = threading.Lock()

    def _voice_path(self, voice: str | None, language: str | None) -> Path:
        key = voice or language or "en"
        rel = DEFAULT_VOICES.get(key)
        if voice and rel is None:
            # explicit voice requested but unknown -> do not silently fall back
            known = ", ".join(sorted(DEFAULT_VOICES))
            raise EngineError(f"unknown Piper voice '{voice}'. Known: {known}")
        if rel is None:
            rel = DEFAULT_VOICES.get(language or "en")
        if rel is None:
            known = ", ".join(sorted(DEFAULT_VOICES))
            raise EngineError(f"no Piper voice for language '{language}'. Known: {known}")
        p = self.models_dir / self.model_subdir / f"{rel}.onnx"
        if not p.exists():
            raise FileNotFoundError(f"Piper voice not found: {p}")
        return p

    def _synthesize(self, text: str, *, language: str | None,
                    voice: str | None, **kwargs) -> SynthesisResult:
        from piper import PiperVoice

        path = str(self._voice_path(voice, language))
        with self._voices_lock:
            pv = self._voices.get(path)
            if pv is None:
                pv = PiperVoice.load(path)
                self._voices[path] = pv
        buf = bytearray()
        for chunk in pv.synthesize(text):
            buf += chunk.audio_int16_bytes
        samples = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
        return SynthesisResult(audio=samples, sample_rate=pv.config.sample_rate)
