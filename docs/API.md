# REST API reference

Base URL: `http://127.0.0.1:8300` (configure with `--host/--port` or `VOXORA_HOST`/`VOXORA_PORT`). Interactive OpenAPI docs at `/docs`.

## Authentication

Optional. If the `VOXORA_API_KEY` environment variable is set on the server, every `/v1` request must authenticate with either header:

- `X-API-Key: <key>`, or
- `Authorization: Bearer <key>` (what OpenAI SDKs send)

The key is read from the environment only — never stored or logged by the application. Comparison is constant-time.

## OpenAI SDK compatibility

The audio endpoints speak the OpenAI field vocabulary, so stock OpenAI clients work by pointing them at Voxora:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8300/v1", api_key="<key-or-anything-if-unset>")

transcript = client.audio.transcriptions.create(model="sensevoice", file=open("sample.wav", "rb"))
print(transcript.text)

with client.audio.speech.with_streaming_response.create(
        model="piper", voice="en", input="Hello from Voxora.") as resp:
    resp.stream_to_file("out.wav")
```

Mapping: `model` = engine name · `input` = text · `voice` = Piper voice key or Qwen3-TTS speaker · `client.models.list()` works. Differences from the hosted API: only `wav` audio is returned (`response_format` other than `wav` is rejected with 422); responses carry extra telemetry fields (`rtf`, `processing_s`, `audio_duration_s`) which SDKs ignore; `speed`/`prompt`/`temperature` fields are ignored.

## GET /health

Liveness probe. No authentication.

```json
{"status": "ok", "version": "0.1"}
```

## GET /v1/models

OpenAI-style model list — every registered engine id. Same authentication rule as other `/v1` routes.

## GET /v1/engines

Engine catalogue with availability. `available` is true only when the runtime dependency is importable **and** the weights exist under the models directory.

```json
{"engines": [
  {"name": "sensevoice", "kind": "asr", "description": "…", "license": "…",
   "requires_extras": ["onnx"], "model_subdir": "sherpa-sense-voice",
   "weights_present": true, "importable": true, "available": true, "loaded": false}
]}
```

## POST /v1/audio/transcriptions

Transcribe an audio file with any ASR engine.

- **Content type**: `multipart/form-data`
- **Fields**:
  - `file` (required): audio file, up to 25 MB. Any format libsndfile decodes (wav/flac/mp3/ogg); multi-channel is downmixed, non-16 kHz input is resampled (linear) to 16 kHz mono.
  - `engine` (optional): engine name; default `VOXORA_DEFAULT_ASR` (`sensevoice`). `model` is accepted as the OpenAI-style alias.
  - `language` (optional): hint forwarded to engines that support it.

Response `200`:

```json
{
  "engine": "sensevoice",
  "text": "甚至出现交易几乎停滞的情况。",
  "language": null,
  "audio_duration_s": 4.204,
  "processing_s": 0.057,
  "rtf": 0.0136
}
```

`rtf` is `processing_s / audio_duration_s` (excludes decode/resample/HTTP).

Errors: `400` undecodable/empty audio · `401` bad API key · `404` unknown engine · `400` engine is not an ASR engine · `413` audio larger than 25 MB · `503` dependency missing or weights not found · `500` engine inference error.

## POST /v1/audio/speech

Synthesize text to a WAV file with any TTS engine.

- **Content type**: `application/json`

```json
{"text": "Hello.", "engine": "piper", "language": "en", "voice": null}
```

- `text` (required, 1–2000 chars; `input` accepted as the OpenAI-style alias), `engine` (default `VOXORA_DEFAULT_TTS`, i.e. `piper`; `model` accepted as alias), `language` (Piper: one of `en/zh/de/fr/ru/es`; Qwen3-TTS: `Chinese/English/Japanese/Korean/German/French/Russian/Spanish/Italian` or `Auto`), `voice` (Piper voice key or Qwen3-TTS speaker name), `response_format` (only `"wav"`; anything else is rejected with 422).

Response `200`: `audio/wav` bytes (16-bit PCM) plus telemetry headers:

| Header | Meaning |
|---|---|
| `X-Engine` | engine that served the request |
| `X-Processing-Time` | wall-clock synthesis seconds |
| `X-Audio-Duration` | produced audio seconds |
| `X-RTF` | processing / audio duration |

## Configuration reference

| Env var | Default | Meaning |
|---|---|---|
| `VOXORA_MODELS_DIR` | `models` | weights directory (see `scripts/download_models.sh`) |
| `VOXORA_COSYVOICE_REPO` | — | path to a recursive CosyVoice clone (cosyvoice engine only) |
| `VOXORA_HOST` / `VOXORA_PORT` | `127.0.0.1` / `8300` | bind address |
| `VOXORA_API_KEY` | empty (auth off) | require `X-API-Key` on `/v1` |
| `VOXORA_NUM_THREADS` | `8` | threads for engine sessions (see methodology §6 before raising) |
| `VOXORA_DEFAULT_ASR` / `VOXORA_DEFAULT_TTS` | `sensevoice` / `piper` | engines used when request omits `engine` |

## Python client example

```python
import requests

base = "http://127.0.0.1:8300"
with open("sample.wav", "rb") as f:
    r = requests.post(f"{base}/v1/audio/transcriptions",
                      files={"file": f}, data={"engine": "qwen3-asr-onnx"})
print(r.json()["text"])

r = requests.post(f"{base}/v1/audio/speech",
                  json={"text": "Server-side synthesis.", "engine": "piper",
                        "language": "en"})
with open("out.wav", "wb") as f:
    f.write(r.content)
```
