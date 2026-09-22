# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1] — 2026-09-22

Initial release.

### Added

- REST API service (`/v1/audio/transcriptions`, `/v1/audio/speech`,
  `/v1/engines`, `/health`) with per-request RTF telemetry and optional
  environment-based API key — the service runs entirely on CPU, and the
  default install needs no PyTorch.
- Engine abstraction with 8 engines: SenseVoiceSmall, streaming Zipformer,
  Qwen3-ASR (ONNX int8 + PyTorch routes), Fun-ASR-Nano (ONNX int8), Piper,
  Qwen3-TTS, CosyVoice3 — lazily loaded, optional heavy extras.
- Benchmark CLI (`voxora`) emitting versioned JSON results with an embedded
  environment fingerprint (CPU affinity + load average included), `--repeat`
  runs with mean/std/cv.
- Scoring layer (CER/WER) with pinned normalization rules and golden tests.
- Measured performance reference data from a Ryzen AI MAX+ 395 host: baseline
  results for all engines, TTS→ASR round-trip accuracy, an optimization
  sweep, and a run-to-run stability study (ambient-load attribution, pinned
  comparison runs).
- OpenAI SDK compatibility: audio endpoints accept `model`/`input` field
  names and `Authorization: Bearer` auth, plus a `/v1/models` catalogue —
  stock OpenAI clients work against `base_url=…/v1`.
- Evaluation report (docs/EVALUATION.md) as the single home for result
  tables; methodology limited to protocol; survey to rationale.
- Data erratum: a corrupted real-speech fixture (7.5 s appended) had
  understated 2026-09-22 ASR aggregates (~3%) and sweep RTFs (~14%);
  affected measurements re-run on 2026-09-23 with tuned-reference files
  that record CPU affinity and load average (see METHODOLOGY.md §8).
- Documentation set: API reference, methodology, reproduction guide, engine
  survey, model licenses, and README translations (zh-CN, de, fr, es, it).
