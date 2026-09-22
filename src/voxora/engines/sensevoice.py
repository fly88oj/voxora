"""SenseVoiceSmall via sherpa-onnx (ONNX, int8/fp32). zh/en/ja/ko/yue."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import TARGET_SR
from .base import AsrEngine, TranscriptionResult
from .registry import register
from .runtime_utils import DEFAULT_ORT_THREADS, decode_offline


@register
class SenseVoiceEngine(AsrEngine):
    name = "sensevoice"
    description = "SenseVoiceSmall (sherpa-onnx ONNX); fast zh/en/ja/ko/yue ASR with ITN"
    license_note = "code Apache-2.0 (sherpa-onnx); weights per SenseVoiceSmall card"
    requires_extras = ("onnx",)
    import_probe = "sherpa_onnx"
    model_subdir = "sherpa-sense-voice"

    def __init__(self, *args, quantized: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantized = quantized
        self._recognizer = None

    def load(self) -> None:
        import sherpa_onnx

        root = self.models_dir / self.model_subdir
        model = root / ("model.int8.onnx" if self.quantized else "model.onnx")
        if not model.exists():
            raise FileNotFoundError(f"weights not found: {model}")
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(root / "tokens.txt"),
            use_itn=True,
            language="auto",
            num_threads=self.num_threads or DEFAULT_ORT_THREADS,
        )

    def _transcribe(self, samples: np.ndarray, sample_rate: int, *,
                    language: str | None, **kwargs: Any) -> TranscriptionResult:
        assert sample_rate == TARGET_SR
        text = decode_offline(self._recognizer, samples, sample_rate)
        return TranscriptionResult(text=text, language=language)
