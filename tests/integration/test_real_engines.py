"""Integration tests against real model weights.

Enable by pointing ``VOXORA_INTEGRATION_MODELS`` at a models directory populated
by ``scripts/download_models.sh``:

    VOXORA_INTEGRATION_MODELS=/path/to/models pytest -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

MODELS = os.environ.get("VOXORA_INTEGRATION_MODELS", "")
FIXTURES = Path(__file__).resolve().parents[2] / "data" / "fixtures"

requires_models = pytest.mark.skipif(
    not MODELS, reason="set VOXORA_INTEGRATION_MODELS to run integration tests")


@requires_models
def test_sensevoice_real_transcription():

    from voxora.audio import load_audio
    from voxora.engines import create_engine

    eng = create_engine("sensevoice", MODELS)
    wav = FIXTURES / "asr_zh.wav"
    audio = load_audio(wav)
    result = eng.transcribe(audio.samples, audio.sample_rate)
    assert result.rtf < 0.5, "SenseVoice should beat 2x realtime on a modern CPU"
    assert len(result.text.strip()) > 0


@requires_models
def test_piper_real_synthesis_roundtrip():
    from voxora.engines import create_engine

    eng = create_engine("piper", MODELS)
    result = eng.synthesize("The weather is really nice today.", language="en")
    assert result.duration_s > 1.0
    assert result.rtf < 1.0, "Piper should synthesize faster than realtime"


@requires_models
def test_api_with_real_engines():
    from fastapi.testclient import TestClient

    from voxora.api import create_app
    from voxora.settings import Settings

    app = create_app(Settings(models_dir=MODELS, num_threads=8))
    client = TestClient(app)
    with open(FIXTURES / "asr_zh.wav", "rb") as f:
        r = client.post("/v1/audio/transcriptions",
                        files={"file": ("asr_zh.wav", f, "audio/wav")},
                        data={"engine": "sensevoice"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"].strip()
    assert body["rtf"] is not None and body["rtf"] < 1.0

    r2 = client.post("/v1/audio/speech",
                     json={"text": "Integration check.", "engine": "piper",
                           "language": "en"})
    assert r2.status_code == 200
    assert r2.headers["X-RTF"]


def test_fixtures_exist():
    for name in ("asr_zh.wav", "piper_en_0.wav", "qwen3tts_zh_0.wav"):
        assert (FIXTURES / name).exists(), f"missing bundled fixture {name}"
