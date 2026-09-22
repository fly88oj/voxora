"""API contract tests using stub engines (no model weights required)."""

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _wav_bytes(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    pcm = (0.1 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_health(app):
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_engines_catalogue(app):
    client = TestClient(app)
    r = client.get("/v1/engines")
    assert r.status_code == 200
    engines = {e["name"]: e for e in r.json()["engines"]}
    assert "sensevoice" in engines
    # stub engines are loadable -> available
    assert engines["stub-asr"]["available"] is True
    # real engines without weights -> not available, but still listed
    assert engines["sensevoice"]["available"] is False


def test_transcription_contract(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
        data={"engine": "stub-asr"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "stub-asr"
    assert body["text"] == "stub transcription"
    assert body["audio_duration_s"] == 1.0
    assert body["processing_s"] >= 0.0
    assert "rtf" in body


def test_transcription_resamples_48k(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample48k.wav", _wav_bytes(0.5, sr=48000), "audio/wav")},
        data={"engine": "stub-asr"},
    )
    assert r.status_code == 200
    # duration is reported at the 16 kHz target rate after resampling
    assert 0.4 < r.json()["audio_duration_s"] < 0.6


def test_transcription_empty_audio_rejected(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"engine": "stub-asr"},
    )
    assert r.status_code == 400


def test_unknown_engine_404(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("s.wav", _wav_bytes(0.1), "audio/wav")},
        data={"engine": "does-not-exist"},
    )
    assert r.status_code == 404


def test_wrong_modality_400(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("s.wav", _wav_bytes(0.1), "audio/wav")},
        data={"engine": "stub-tts"},
    )
    assert r.status_code == 400


def test_speech_returns_wav_with_timing_headers(app):
    client = TestClient(app)
    r = client.post("/v1/audio/speech", json={"text": "hello", "engine": "stub-tts"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["X-Engine"] == "stub-tts"
    assert float(r.headers["X-Audio-Duration"]) == pytest.approx(0.5, abs=0.01)
    assert float(r.headers["X-Processing-Time"]) >= 0.0
    assert r.content[:4] == b"RIFF"


def test_speech_empty_text_422(app):
    client = TestClient(app)
    r = client.post("/v1/audio/speech", json={"text": ""})
    assert r.status_code == 422


def test_api_key_enforced_when_configured(secured_app):
    # The valid key is generated per session (see conftest) — never a literal.
    valid_key = secured_app.state.settings.api_key
    client = TestClient(secured_app)
    assert client.get("/v1/engines").status_code == 401
    assert client.get("/v1/engines", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/engines", headers={"X-API-Key": valid_key}).status_code == 200


def test_health_needs_no_key(secured_app):
    client = TestClient(secured_app)
    assert client.get("/health").status_code == 200


def test_transcription_garbage_bytes_400(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("garbage.wav", b"\x00\x01not-audio\xff" * 100, "audio/wav")},
        data={"engine": "stub-asr"},
    )
    assert r.status_code == 400


def test_unavailable_engine_503(app):
    # stub-broken raises ImportError at load -> EngineNotAvailable -> 503
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("s.wav", _wav_bytes(0.2), "audio/wav")},
        data={"engine": "stub-broken"},
    )
    assert r.status_code == 503
    assert "missing dependency" in r.json()["detail"]


def test_missing_weights_maps_to_503(app):
    # sensevoice with no weights dir -> FileNotFoundError -> 503 (not 500)
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("s.wav", _wav_bytes(0.2), "audio/wav")},
        data={"engine": "sensevoice"},
    )
    assert r.status_code == 503


# --- OpenAI SDK compatibility -------------------------------------------------

def test_openai_model_field_alias(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("s.wav", _wav_bytes(0.3), "audio/wav")},
        data={"model": "stub-asr"},  # OpenAI field name
    )
    assert r.status_code == 200
    assert r.json()["engine"] == "stub-asr"


def test_openai_speech_input_and_model_aliases(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/speech",
        json={"model": "stub-tts", "input": "hello", "voice": "alloy",
              "response_format": "wav"},
    )
    assert r.status_code == 200
    assert r.headers["X-Engine"] == "stub-tts"
    assert r.content[:4] == b"RIFF"


def test_openai_speech_unsupported_format_422(app):
    client = TestClient(app)
    r = client.post(
        "/v1/audio/speech",
        json={"model": "stub-tts", "input": "hello", "response_format": "mp3"},
    )
    assert r.status_code == 422


def test_models_endpoint_openai_shape(app):
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert "sensevoice" in ids and "piper" in ids
    assert all(m["object"] == "model" for m in body["data"])


def test_bearer_token_accepted(secured_app):
    valid_key = secured_app.state.settings.api_key
    client = TestClient(secured_app)
    assert client.get("/v1/models",
                      headers={"Authorization": f"Bearer {valid_key}"}).status_code == 200
    assert client.get("/v1/models",
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
