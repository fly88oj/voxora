"""Audio I/O helpers shared by engines, API, and benchmark runner.

All engines consume mono float32 PCM at 16 kHz. The API accepts any format
libsndfile can decode and resamples with linear interpolation (documented in
docs/METHODOLOGY.md — resampling behavior is documented in docs/API.md; use sox for
bit-exact experiment pipelines).
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


@dataclass
class Audio:
    samples: np.ndarray  # float32, mono, [-1, 1]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    """Linear-interpolation resampling of 1-D float samples."""
    if src_sr == dst_sr:
        return samples
    n_out = int(round(len(samples) * dst_sr / src_sr))
    x_src = np.arange(len(samples), dtype=np.float64)
    x_dst = np.linspace(0.0, len(samples) - 1, n_out)
    return np.interp(x_dst, x_src, samples.astype(np.float64)).astype(np.float32)


def load_audio(data: bytes | Path | str) -> Audio:
    """Decode audio from raw bytes (any libsndfile format) or a file path."""
    if isinstance(data, Path):
        data = str(data)
    arr, sr = sf.read(io.BytesIO(data) if isinstance(data, bytes) else data,
                      dtype="float32", always_2d=True)
    mono = arr.mean(axis=1) if arr.shape[1] > 1 else arr[:, 0]
    if sr != TARGET_SR:
        mono = resample_linear(mono, sr)
    return Audio(mono.astype(np.float32), TARGET_SR)


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float mono samples ([-1, 1]) to a 16-bit PCM WAV byte string."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())
