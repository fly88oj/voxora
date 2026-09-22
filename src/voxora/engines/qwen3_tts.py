"""Qwen3-TTS-0.6B CustomVoice via the official `qwen-tts` package (optional extra)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import SynthesisResult, TtsEngine
from .registry import register
from .runtime_utils import resolve_torch_dtype


@register
class Qwen3TtsEngine(TtsEngine):
    name = "qwen3-tts"
    description = "Qwen3-TTS-12Hz-0.6B CustomVoice (qwen-tts PyTorch); 10 languages, 9 voices"
    license_note = "code Apache-2.0; weights per Qwen3-TTS card"
    requires_extras = ("torch",)
    import_probe = "qwen_tts"
    model_subdir = "Qwen3-TTS-0.6B"

    def __init__(self, *args, dtype: str = "bfloat16", **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = dtype

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        root = self.models_dir / self.model_subdir
        if not (root / "model.safetensors").exists():
            raise FileNotFoundError(f"weights not found: {root}")
        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        self._model = Qwen3TTSModel.from_pretrained(
            str(root), device_map="cpu", dtype=resolve_torch_dtype(self.dtype))

    def _synthesize(self, text: str, *, language: str | None,
                    voice: str | None, **kwargs: Any) -> SynthesisResult:
        wavs, sr = self._model.generate_custom_voice(
            text=text, speaker=voice or "Vivian",
            language=language or "Auto")
        return SynthesisResult(audio=np.asarray(wavs[0], dtype=np.float32), sample_rate=sr)
