"""Registry behaviour tests."""

import numpy as np
import pytest

from voxora.engines import EngineError, create_engine, engine_classes, register
from voxora.engines.base import AsrEngine, EngineKind


def test_builtin_engines_registered():
    names = engine_classes()
    for expected in ("sensevoice", "zipformer", "qwen3-asr-onnx", "funasr-nano-onnx",
                     "piper", "qwen3-asr", "qwen3-tts", "cosyvoice"):
        assert expected in names, f"missing engine {expected}"


def test_asr_tts_split():
    kinds = {n: c.kind for n, c in engine_classes().items()}
    assert kinds["sensevoice"] is EngineKind.ASR
    assert kinds["piper"] is EngineKind.TTS


def test_unknown_engine_raises():
    with pytest.raises(EngineError, match="unknown engine"):
        create_engine("nope", "models")


def test_duplicate_name_rejected():
    with pytest.raises(ValueError, match="duplicate engine name"):

        @register
        class DupEngine(AsrEngine):
            name = "sensevoice"

    # the duplicate never entered the registry
    assert engine_classes()["sensevoice"] is not None
    assert "dupengine" not in engine_classes()


def test_kind_mismatch_guard(stub_engines):
    eng = create_engine("stub-tts", "/nonexistent")
    with pytest.raises(EngineError, match="cannot transcribe"):
        eng.transcribe(np.zeros(1600, dtype="float32"), 16000)
