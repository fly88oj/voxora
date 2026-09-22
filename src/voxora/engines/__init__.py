"""Built-in engines. Importing this module registers all of them."""

from . import (  # noqa: F401 — imported for their registration side effect
    cosyvoice,
    funasr_nano_onnx,
    piper,
    qwen3_asr,
    qwen3_asr_onnx,
    qwen3_tts,
    sensevoice,
    zipformer,
)
from .base import (
               AsrEngine,
               BaseEngine,
               EngineError,
               EngineKind,
               EngineNotAvailable,
               SynthesisResult,
               TranscriptionResult,
               TtsEngine,
)
from .registry import create_engine, engine_classes, get_engine, iter_engines, register, reset_pool

__all__ = [
    "AsrEngine", "BaseEngine", "EngineError", "EngineKind", "EngineNotAvailable",
    "SynthesisResult", "TtsEngine", "TranscriptionResult",
    "create_engine", "engine_classes", "get_engine", "iter_engines", "register",
    "reset_pool",
]
