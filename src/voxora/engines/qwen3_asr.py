"""Qwen3-ASR-0.6B via the official `qwen-asr` PyTorch package (optional extra).

Recommended deployment on AMD Zen4/Zen5 CPUs: ``dtype=bfloat16`` with threads
pinned to physical cores (see docs/METHODOLOGY.md §6 — bf16 measured 3.9x
faster than fp32 with identical transcripts).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from ..audio import encode_wav
from .base import AsrEngine, TranscriptionResult
from .registry import register
from .runtime_utils import resolve_torch_dtype


@register
class Qwen3AsrEngine(AsrEngine):
    name = "qwen3-asr"
    description = "Qwen3-ASR-0.6B (qwen-asr PyTorch); 30 languages, bf16 recommended"
    license_note = "code Apache-2.0; weights per Qwen3-ASR card (Apache-2.0)"
    requires_extras = ("torch",)
    import_probe = "qwen_asr"
    model_subdir = "Qwen3-ASR-0.6B"

    def __init__(self, *args, dtype: str = "bfloat16", **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = dtype

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        root = self.models_dir / self.model_subdir
        if not (root / "model.safetensors").exists():
            raise FileNotFoundError(f"weights not found: {root}")
        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        self._model = Qwen3ASRModel.from_pretrained(
            str(root), dtype=resolve_torch_dtype(self.dtype), device_map="cpu",
            max_new_tokens=1024)

    def _transcribe(self, samples: np.ndarray, sample_rate: int, *,
                    language: str | None, **kwargs: Any) -> TranscriptionResult:
        # qwen-asr accepts paths/URLs; write a temp WAV (16-bit PCM).
        with NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            Path(tf.name).write_bytes(encode_wav(samples, sample_rate))
            r = self._model.transcribe(audio=tf.name, language=language, context="")[0]
        return TranscriptionResult(text=r.text,
                                   language=getattr(r, "language", None))
