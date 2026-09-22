"""Shared runtime helpers for engine implementations."""

from __future__ import annotations

import numpy as np

# Measured sweet spot for ONNX Runtime sessions on the reference 16-core CPU
# (docs/METHODOLOGY.md §6); engines fall back to it when no value is given.
DEFAULT_ORT_THREADS = 8


def decode_offline(recognizer, samples: np.ndarray, sample_rate: int) -> str:
    """Run one utterance through a sherpa-onnx OfflineRecognizer."""
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text


def resolve_torch_dtype(name: str):
    """Map a dtype name to a torch dtype for the CPU PyTorch engines."""
    import torch

    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]
