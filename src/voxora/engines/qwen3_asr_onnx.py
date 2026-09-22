"""Qwen3-ASR-0.6B (ONNX int8 export) via sherpa-onnx — 30-language LLM ASR."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import TARGET_SR
from .base import AsrEngine, TranscriptionResult
from .registry import register
from .runtime_utils import DEFAULT_ORT_THREADS, decode_offline


@register
class Qwen3AsrOnnxEngine(AsrEngine):
    name = "qwen3-asr-onnx"
    description = "Qwen3-ASR-0.6B int8 ONNX (sherpa-onnx); 30 languages + zh dialects"
    license_note = "code Apache-2.0 (sherpa-onnx); weights per Qwen3-ASR card (Apache-2.0)"
    requires_extras = ("onnx",)
    import_probe = "sherpa_onnx"
    model_subdir = "qwen3-asr-onnx-int8"

    def load(self) -> None:
        import sherpa_onnx

        root = self.models_dir / self.model_subdir
        files = [root / "conv_frontend.onnx", root / "encoder.int8.onnx",
                 root / "decoder.int8.onnx", root / "tokenizer"]
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"weights not found: {f}")
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(root / "conv_frontend.onnx"),
            encoder=str(root / "encoder.int8.onnx"),
            decoder=str(root / "decoder.int8.onnx"),
            tokenizer=str(root / "tokenizer"),
            num_threads=self.num_threads or DEFAULT_ORT_THREADS,
        )

    def _transcribe(self, samples: np.ndarray, sample_rate: int, *,
                    language: str | None, **kwargs: Any) -> TranscriptionResult:
        assert sample_rate == TARGET_SR
        text = decode_offline(self._recognizer, samples, sample_rate)
        return TranscriptionResult(text=text, language=language)
