"""Fun-ASR-Nano (int8 ONNX) via sherpa-onnx — zh/en/ja + 7 Chinese dialect groups."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import TARGET_SR
from .base import AsrEngine, TranscriptionResult
from .registry import register
from .runtime_utils import DEFAULT_ORT_THREADS, decode_offline


@register
class FunAsrNanoOnnxEngine(AsrEngine):
    name = "funasr-nano-onnx"
    description = "Fun-ASR-Nano int8 ONNX (sherpa-onnx); zh/en/ja + Chinese dialects"
    license_note = "code Apache-2.0; weights Apache-2.0 per Fun-ASR model card"
    requires_extras = ("onnx",)
    import_probe = "sherpa_onnx"
    model_subdir = "sherpa-onnx-funasr-nano-int8-2025-12-30"

    def load(self) -> None:
        import sherpa_onnx

        root = self.models_dir / self.model_subdir
        tok = next((p for p in (root / "tokenizer", root / "Qwen3-0.6B") if p.exists()), None)
        if tok is None:
            raise FileNotFoundError(f"tokenizer directory not found under {root}")
        files = [root / "encoder_adaptor.int8.onnx", root / "llm.int8.onnx",
                 root / "embedding.int8.onnx", tok]
        for f in files:
            if f is None or not f.exists():
                raise FileNotFoundError(f"weights not found: {f}")
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
            encoder_adaptor=str(root / "encoder_adaptor.int8.onnx"),
            llm=str(root / "llm.int8.onnx"),
            embedding=str(root / "embedding.int8.onnx"),
            tokenizer=str(tok),
            num_threads=self.num_threads or DEFAULT_ORT_THREADS,
            language="",
            itn=True,
        )

    def _transcribe(self, samples: np.ndarray, sample_rate: int, *,
                    language: str | None, **kwargs: Any) -> TranscriptionResult:
        assert sample_rate == TARGET_SR
        text = decode_offline(self._recognizer, samples, sample_rate)
        return TranscriptionResult(text=text, language=language)
