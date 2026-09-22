"""Streaming bilingual (zh-en) transducer Zipformer via sherpa-onnx.

Transcription feeds the audio in 100 ms chunks to mimic a live stream, so the
measured processing time includes the streaming decode loop exactly as the
benchmark did (docs/METHODOLOGY.md §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import TARGET_SR
from .base import AsrEngine, TranscriptionResult
from .registry import register
from .runtime_utils import DEFAULT_ORT_THREADS

CHUNK_SAMPLES = 1600  # 100 ms @ 16 kHz


@register
class ZipformerStreamingEngine(AsrEngine):
    name = "zipformer"
    description = "Streaming Zipformer bilingual zh-en (sherpa-onnx ONNX transducer)"
    license_note = "code Apache-2.0 (sherpa-onnx); model Apache-2.0 (k2-fsa release)"
    requires_extras = ("onnx",)
    import_probe = "sherpa_onnx"
    model_subdir = "sherpa-zipformer-zh-en"

    def __init__(self, *args, quantized: bool = True, chunk_samples: int = CHUNK_SAMPLES,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.quantized = quantized
        self.chunk_samples = chunk_samples
        self._recognizer = None

    def load(self) -> None:
        import sherpa_onnx

        root = self.models_dir / self.model_subdir
        enc = root / ("encoder-epoch-99-avg-1.int8.onnx" if self.quantized
                      else "encoder-epoch-99-avg-1.onnx")
        for f in (enc, root / "decoder-epoch-99-avg-1.onnx",
                  root / "joiner-epoch-99-avg-1.onnx", root / "tokens.txt"):
            if not f.exists():
                raise FileNotFoundError(f"weights not found: {f}")
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(root / "tokens.txt"),
            encoder=str(enc),
            decoder=str(root / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(root / "joiner-epoch-99-avg-1.onnx"),
            num_threads=self.num_threads or DEFAULT_ORT_THREADS,
        )

    def _transcribe(self, samples: np.ndarray, sample_rate: int, *,
                    language: str | None, **kwargs: Any) -> TranscriptionResult:
        assert sample_rate == TARGET_SR
        rec = self._recognizer
        stream = rec.create_stream()
        for i in range(0, len(samples), self.chunk_samples):
            stream.accept_waveform(sample_rate, samples[i:i + self.chunk_samples])
            while rec.is_ready(stream):
                rec.decode_stream(stream)
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        return TranscriptionResult(text=rec.get_result(stream), language=language)
