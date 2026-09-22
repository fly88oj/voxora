"""FastAPI application exposing every registered engine behind one REST contract.

Endpoints (OpenAPI docs at ``/docs``):

- ``GET  /health``                    — liveness + version
- ``GET  /v1/models``                 — OpenAI-style model catalogue (engine ids)
- ``GET  /v1/engines``                — engine catalogue with availability info
- ``POST /v1/audio/transcriptions``   — multipart audio file -> transcript JSON
- ``POST /v1/audio/speech``           — JSON text -> audio/wav bytes + timing headers

**OpenAI SDK compatibility.** The audio endpoints accept the OpenAI field
names (``model`` for the engine, ``input`` for the text, ``voice``), and
authenticate with either ``Authorization: Bearer <key>`` or ``X-API-Key``.
Point an OpenAI client at ``base_url="http://…:8300/v1"`` and use an engine
name as the model — see docs/API.md §OpenAI SDK compatibility. Differences:
only ``wav`` audio is returned (no mp3), and extra telemetry fields are added
to responses (ignored by the SDK).

Authentication: optional. When ``VOXORA_API_KEY`` is set, ``/v1`` requests
must carry it (either header). The key is never logged.
"""

from __future__ import annotations

import secrets

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field, model_validator

from .. import __version__
from ..audio import encode_wav, load_audio
from ..engines import (
    EngineError,
    EngineKind,
    EngineNotAvailable,
    create_engine,
    engine_classes,
    get_engine,
)
from ..engines.base import BaseEngine
from ..settings import Settings, get_settings

# Hard cap on uploaded audio (matches the common hosted-API limit class).
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class SpeechRequest(BaseModel):
    """Body of ``POST /v1/audio/speech``.

    Accepts both the native field names (``engine``/``text``) and the OpenAI
    names (``model``/``input``); ``text``/``engine`` win when both are given.
    """

    text: str | None = Field(default=None, max_length=2000)
    input: str | None = Field(default=None, max_length=2000)  # OpenAI field name
    engine: str | None = None
    model: str | None = None  # OpenAI field name
    voice: str | None = None
    language: str | None = None
    response_format: str | None = None  # only "wav" is produced

    @model_validator(mode="after")
    def _resolve_aliases(self) -> SpeechRequest:
        if self.text is None:
            self.text = self.input
        if self.engine is None:
            self.engine = self.model
        if not (self.text or "").strip():
            raise ValueError("text (or input) is required and must be non-empty")
        if self.response_format not in (None, "wav"):
            raise ValueError(f"response_format {self.response_format!r} is not supported; "
                             "only 'wav' is produced")
        return self


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Voxora API",
        description="CPU-only speech recognition & synthesis service — one "
                    "contract, eight engines, per-request RTF telemetry. "
                    "OpenAI-SDK compatible (audio endpoints).",
        version=__version__,
    )
    app.state.settings = settings

    def _auth(request: Request) -> None:
        """Accept X-API-Key or Authorization: Bearer; constant-time compare."""
        expected = settings.api_key
        if not expected:
            return
        header = request.headers.get("x-api-key") or ""
        if not header:
            auth = request.headers.get("authorization") or ""
            if auth.lower().startswith("bearer "):
                header = auth[7:]
        if not header or not secrets.compare_digest(header, expected):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def _engine_for(name: str, kind: EngineKind) -> BaseEngine:
        try:
            cls = engine_classes()[name]
        except KeyError:
            known = ", ".join(sorted(engine_classes()))
            raise HTTPException(status_code=404,
                                detail=f"unknown engine '{name}'. Known: {known}") from None
        if cls.kind is not kind:
            raise HTTPException(status_code=400,
                                detail=f"engine '{name}' does not support {kind.value}")
        return get_engine(name, settings.models_dir, num_threads=settings.num_threads)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models")
    def list_models(request: Request) -> dict:
        _auth(request)
        data = [{"id": name, "object": "model", "created": 0, "owned_by": "voxora",
                 "kind": cls.kind.value}
                for name, cls in sorted(engine_classes().items())]
        return {"object": "list", "data": data}

    @app.get("/v1/engines")
    def engines(request: Request) -> dict:
        _auth(request)
        import importlib.util

        out = []
        for name, cls in sorted(engine_classes().items()):
            probe = create_engine(name, settings.models_dir,
                                  num_threads=settings.num_threads)
            entry = probe.info()
            entry["importable"] = (importlib.util.find_spec(cls.import_probe) is not None
                                   if cls.import_probe else True)
            entry["available"] = entry["importable"] and entry["weights_present"]
            out.append(entry)
        return {"engines": out}

    @app.post("/v1/audio/transcriptions")
    def transcribe(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008 — standard FastAPI form idiom
        engine: str | None = Form(default=None),
        model: str | None = Form(default=None),  # OpenAI field name
        language: str | None = Form(default=None),
    ) -> dict:
        _auth(request)
        name = engine or model or settings.default_asr
        eng = _engine_for(name, EngineKind.ASR)
        data = file.file.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"audio larger than {MAX_AUDIO_BYTES // (1024*1024)} MB")
        try:
            audio = load_audio(data)
        except Exception as e:
            raise HTTPException(status_code=400,
                                detail=f"cannot decode audio: {e}") from e
        if audio.duration_s <= 0:
            raise HTTPException(status_code=400, detail="empty or undecodable audio")
        try:
            result = eng.transcribe(audio.samples, audio.sample_rate, language=language)
        except EngineNotAvailable as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except EngineError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {
            "engine": eng.name,
            "text": result.text,
            "language": result.language,
            "audio_duration_s": result.audio_duration_s,
            "processing_s": result.processing_s,
            "rtf": result.rtf,
        }

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest, request: Request) -> Response:
        _auth(request)
        name = req.engine or settings.default_tts
        eng = _engine_for(name, EngineKind.TTS)
        try:
            result = eng.synthesize(req.text, language=req.language, voice=req.voice)
        except EngineNotAvailable as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except EngineError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        wav = encode_wav(result.audio, result.sample_rate)
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={
                "X-Engine": eng.name,
                "X-Processing-Time": f"{result.processing_s:.3f}",
                "X-Audio-Duration": f"{result.duration_s:.3f}",
                "X-RTF": f"{result.rtf:.4f}" if result.rtf is not None else "",
            },
        )

    return app


app = create_app()
