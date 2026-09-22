"""Benchmark runner: one engine, one workload, one versioned JSON result file.

Result schema (``schema_version`` guards format evolution):

.. code-block:: json

    {
      "schema_version": 1,
      "engine": "sensevoice",
      "kind": "asr",
      "config": {"num_threads": 8, "repeat": 1, "...": "..."},
      "environment": {"python": "...", "cpu": {}, "libraries": {}},
      "aggregate": {"audio_s": 0, "rtf": 0, "passes": [0], "stats": {"n": 1, "mean": 0, "std": 0, "cv": 0}},
      "items": [{"file": "a.wav", "audio_s": 0, "processing_s": 0, "rtf": 0, "hyp": "..."}]
    }

RTF is ``processing_s / audio_duration_s``; values below 1.0 are faster than
realtime. Processing time is taken from the engine layer (``transcribe`` /
``synthesize`` wrap the call internally), so it excludes I/O, resampling and
result assembly — matching docs/METHODOLOGY.md §1.

With ``repeat > 1`` the whole workload runs N times in the same process and
``aggregate.stats`` reports the sample standard deviation and coefficient of
variation across passes (pass 1 is the cold pass and is included — see
docs/METHODOLOGY.md §6). ``aggregate.rtf`` stays the across-pass mean so
single-run consumers are unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import SCHEMA_VERSION, __version__
from ..audio import encode_wav, load_audio
from ..engines import EngineKind, create_engine
from ..environment import snapshot


def _stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / max(n - 1, 1)) ** 0.5
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv": round(std / mean, 4) if mean else None,
    }


def run_asr(engine_name: str, files: list[Path], *, models_dir: str,
            num_threads: int = 8, repeat: int = 1) -> dict[str, Any]:
    eng = create_engine(engine_name, models_dir, num_threads=num_threads)
    eng.ensure_loaded()
    audios = [(path.name, load_audio(path)) for path in files]
    total_audio = sum(a.duration_s for _, a in audios)

    pass_rtfs: list[float] = []
    items: list[dict[str, Any]] = []
    for _pass in range(max(repeat, 1)):
        total_proc = 0.0
        items = []
        for name, audio in audios:
            result = eng.transcribe(audio.samples, audio.sample_rate)
            total_proc += result.processing_s
            items.append({
                "file": name,
                "audio_s": round(audio.duration_s, 3),
                "processing_s": result.processing_s,
                "rtf": result.rtf,
                "hyp": result.text,
            })
        pass_rtfs.append(round(total_proc / max(total_audio, 1e-9), 4))

    return {
        "schema_version": SCHEMA_VERSION,
        "engine": eng.name,
        "kind": EngineKind.ASR.value,
        "config": {"num_threads": num_threads, "repeat": repeat, "load_s": eng.load_s,
                   "voxora_version": __version__},
        "environment": snapshot(),
        "aggregate": {
            "audio_s": round(total_audio, 2),
            "rtf": _stats(pass_rtfs)["mean"],
            "passes": pass_rtfs,
            "stats": _stats(pass_rtfs),
        },
        "items": items,
    }


def run_tts(engine_name: str, texts: list[str], *, models_dir: str,
            num_threads: int = 8, language: str | None = None,
            voice: str | None = None, wav_dir: Path | None = None,
            repeat: int = 1) -> dict[str, Any]:
    eng = create_engine(engine_name, models_dir, num_threads=num_threads)
    eng.ensure_loaded()
    if wav_dir is not None:
        wav_dir.mkdir(parents=True, exist_ok=True)

    pass_rtfs: list[float] = []
    total_audio = 0.0
    items: list[dict[str, Any]] = []
    for _pass in range(max(repeat, 1)):
        total_audio = 0.0
        total_proc = 0.0
        items = []
        for i, text in enumerate(texts):
            result = eng.synthesize(text, language=language, voice=voice)
            total_audio += result.duration_s
            total_proc += result.processing_s
            row: dict[str, Any] = {
                "text": text,
                "audio_s": round(result.duration_s, 3),
                "processing_s": result.processing_s,
                "rtf": result.rtf,
            }
            if wav_dir is not None and _pass == 0:
                out = wav_dir / f"{eng.name}_{i}.wav"
                out.write_bytes(encode_wav(result.audio, result.sample_rate))
                row["file"] = str(out)
            items.append(row)
        pass_rtfs.append(round(total_proc / max(total_audio, 1e-9), 4))

    return {
        "schema_version": SCHEMA_VERSION,
        "engine": eng.name,
        "kind": EngineKind.TTS.value,
        "config": {"num_threads": num_threads, "repeat": repeat, "load_s": eng.load_s,
                   "voxora_version": __version__, "language": language,
                   "voice": voice},
        "environment": snapshot(),
        "aggregate": {
            "audio_s": round(total_audio, 2),
            "rtf": _stats(pass_rtfs)["mean"],
            "passes": pass_rtfs,
            "stats": _stats(pass_rtfs),
        },
        "items": items,
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["run_asr", "run_tts", "write_result"]
