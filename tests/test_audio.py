"""Audio I/O helper tests."""

import numpy as np

from voxora.audio import encode_wav, load_audio, resample_linear


def _sine(sr: int, dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_resample_identity_when_same_rate():
    x = _sine(16000)
    assert resample_linear(x, 16000) is x


def test_resample_changes_length_proportionally():
    x = _sine(48000, dur=1.0)
    y = resample_linear(x, 48000, 16000)
    assert abs(len(y) - 16000) <= 1


def test_wav_roundtrip_preserves_duration():
    x = _sine(16000, dur=0.5)
    wav = encode_wav(x, 16000)
    audio = load_audio(wav)
    assert audio.sample_rate == 16000
    assert abs(audio.duration_s - 0.5) < 0.01


def test_load_resamples_to_16k():
    wav = encode_wav(_sine(22050, dur=1.0), 22050)
    audio = load_audio(wav)
    assert audio.sample_rate == 16000
    assert abs(audio.duration_s - 1.0) < 0.05
