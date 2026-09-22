"""CosyVoice3 (Fun-CosyVoice3-0.5B) — optional extra loaded from a source checkout.

CosyVoice is not published as a plain PyPI package; point ``VOXORA_COSYVOICE_REPO``
at a recursive clone of FunAudioLLM/CosyVoice. The loader also patches
``torchaudio.load`` to a soundfile-backed implementation because torchaudio
>=2.9 routes all I/O through torchcodec, which does not resolve its symbols
against CPU-only torch builds (see docs/REPRODUCING.md §5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..settings import get_settings
from .base import EngineError, SynthesisResult, TtsEngine
from .registry import register

PROMPT_TEXT = ("You are a helpful assistant.<|endofprompt|>"
               "希望你以后能够做的比我还好呦。")


@register
class CosyVoiceEngine(TtsEngine):
    name = "cosyvoice"
    description = "CosyVoice3 0.5B zero-shot/cloning TTS (source checkout required)"
    license_note = "code Apache-2.0 (CosyVoice repo); weights per model card"
    requires_extras = ("torch",)
    import_probe = "torchaudio"
    model_subdir = "Fun-CosyVoice3-0.5B"

    def load(self) -> None:
        import torch

        repo = Path(get_settings().cosyvoice_repo)
        if not repo.exists():
            raise EngineError(
                "VOXORA_COSYVOICE_REPO must point to a recursive clone of "
                "https://github.com/FunAudioLLM/CosyVoice")
        import sys
        for p in (str(repo), str(repo / "third_party/Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)

        import soundfile as sf
        import torchaudio

        def _load_sf(filepath, *a, **k):
            data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T), sr

        torchaudio.load = _load_sf

        from cosyvoice.cli.cosyvoice import AutoModel

        root = self.models_dir / self.model_subdir
        if not (root / "llm.pt").exists():
            raise FileNotFoundError(f"weights not found: {root}")
        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        self._cv = AutoModel(model_dir=str(root))
        self._prompt_wav = str(repo / "asset/zero_shot_prompt.wav")

    def _synthesize(self, text: str, *, language: str | None,
                    voice: str | None, **kwargs: Any) -> SynthesisResult:
        pieces = []
        for chunk in self._cv.inference_zero_shot(text, PROMPT_TEXT, self._prompt_wav,
                                                  stream=False):
            arr = chunk["tts_speech"]
            pieces.append(arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr))
        if not pieces:
            raise EngineError("cosyvoice produced no audio")
        return SynthesisResult(audio=np.concatenate(pieces).squeeze(),
                               sample_rate=self._cv.sample_rate)
