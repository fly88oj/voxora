# Voxora

[![CI](https://github.com/fly88oj/voxora/actions/workflows/ci.yml/badge.svg)](https://github.com/fly88oj/voxora/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Voxora is a REST API service for speech recognition (ASR) and text-to-speech (TTS) that runs entirely on CPU.**

- **One API, eight engines.** Transcribe or synthesize through a single OpenAI-compatible contract; pick an engine per request, mix and match freely — stock OpenAI SDKs work by pointing `base_url` at Voxora (see [docs/API.md](docs/API.md#openai-sdk-compatibility)).
- **CPU-only by design.** No GPU, no CUDA, no accelerator dependencies — the default install doesn't even need PyTorch. Deploy on any x86-64 Linux box, from a 2-core VM to a workstation.
- **Per-request telemetry.** Every response carries wall-clock timing and RTF (real-time factor), so you can watch what your hardware actually delivers.
- **Measured, not marketed.** Ships with reproducible benchmark tooling and reference measurements (raw data, environment fingerprints, methodology) so capacity planning rests on evidence — see [Performance reference](#performance-reference).

## Quickstart

```bash
# 1. Install (ONNX engines only — no PyTorch required)
pip install -e ".[onnx]"
#    ...or the full stack incl. PyTorch engines: pip install -e ".[onnx,torch]"

# 2. Fetch model weights (~3 GB for the default set; see the script for channels)
scripts/download_models.sh models

# 3. Serve
voxora-api --models-dir models --port 8300
```

Transcribe (any libsndfile format; resampled to 16 kHz mono server-side):

```bash
curl -s http://127.0.0.1:8300/v1/audio/transcriptions \
  -F file=@sample.wav -F engine=sensevoice
# {"engine":"sensevoice","text":"...","audio_duration_s":4.2,"processing_s":0.04,"rtf":0.01,...}
```

Synthesize (returns `audio/wav` with timing headers):

```bash
curl -s http://127.0.0.1:8300/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"text":"Hello from Voxora.","engine":"piper","language":"en"}' \
  -o out.wav -D -
# X-Engine: piper
# X-RTF: 0.05
```

Interactive OpenAPI docs: `http://127.0.0.1:8300/docs`. Full REST reference: [docs/API.md](docs/API.md).

## Engines

| Name | Kind | Languages | Extras | Weights |
|---|---|---|---|---|
| `sensevoice` | ASR | zh, yue, en, ja, ko | `onnx` | SenseVoiceSmall (~230 MB) |
| `zipformer` | ASR (streaming) | zh, en | `onnx` | bilingual Zipformer (~180 MB) |
| `qwen3-asr-onnx` | ASR | 30 langs + zh dialects | `onnx` | Qwen3-ASR int8 export (~1 GB) |
| `funasr-nano-onnx` | ASR | zh, en, ja + 7 dialect groups | `onnx` | Fun-ASR-Nano int8 (~750 MB) |
| `qwen3-asr` | ASR | 30 langs + zh dialects | `torch` | Qwen3-ASR-0.6B (~1.9 GB) |
| `piper` | TTS | per-voice (en/zh/de/fr/ru/es bundled) | `onnx` | 63 MB per voice |
| `qwen3-tts` | TTS | 10 langs, 9 voices | `torch` | Qwen3-TTS-0.6B (~2.5 GB) |
| `cosyvoice` | TTS | zh-first, cross-lingual, dialects | `torch` | Fun-CosyVoice3-0.5B (~10 GB) |

`GET /v1/engines` reports live availability (dependency installed + weights present). Licensing differs per engine and per checkpoint — see [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md) before redistributing.

## Performance reference

What can you expect from these engines on CPU? Voxora's answer is measured, versioned data rather than vendor numbers — full tables, accuracy results, and the optimization/stability studies live in one place: the **[evaluation report](docs/EVALUATION.md)** (protocol and validity: [docs/METHODOLOGY.md](docs/METHODOLOGY.md); raw data under [`data/`](data)).

Three operational facts from it:

1. **bf16 is free speed on AMD Zen 4/5** — 3.5–3.9× for the PyTorch engines, transcripts verified identical; and never oversubscribe threads (32 threads on 16 cores degrades them 5–46×).
2. **The LLM-ASR engines deploy best as int8 ONNX** — ~6× faster than PyTorch fp32 at 60% less memory, near-identical accuracy.
3. **Piper is the only realtime TTS tier** (RTF 0.04–0.06, first audio <300 ms); the LLM TTS engines, even tuned (RTF 1.4–6.0), suit asynchronous synthesis.

Caveat: measurements come from a shared, non-exclusive machine — absolute RTF tracks ambient CPU load (load and affinity are recorded in every result file); rankings measured back-to-back are stable. Details: [evaluation report §7](docs/EVALUATION.md#7-stability-findings).

## Built-in benchmark tooling

The same package ships the measurement harness that produced the reference data:

```bash
voxora list                                        # engine catalogue
voxora run --engine sensevoice --audio-dir data/fixtures -o r.json
voxora run --engine piper --text "Hello." --language en -o t.json --wav-dir wavs/
voxora run --engine zipformer --audio-dir data/fixtures --repeat 5 -o zf.json
```

Every result file embeds a schema version, the exact engine configuration, an environment fingerprint (CPU, library versions, **CPU affinity and load average at run time**), and — with `--repeat` — per-pass RTF plus mean/std/cv.

## Repository layout

```
src/voxora/
  engines/     engine interface + 8 implementations (lazy, optional deps)
  api/         FastAPI service (voxora-api)
  bench/       measurement CLI + versioned result writer
  scoring.py   CER/WER with pinned normalization rules
  audio.py     decode/resample/encode helpers
  environment.py  reproducibility fingerprint
data/
  results/       raw per-file results from the reference machine
  optimization/  thread/dtype/batching sweep results
  stability/     replication + pinned runs (load-vs-RTF evidence)
  fixtures/      small wav fixtures used by tests
  corpus.json    the round-trip corpus (9 languages)
  environment.json  fingerprint of the reference machine
docs/           API reference, methodology, licenses, survey, translations
tests/          unit + contract tests (no weights needed); integration marked separately
```

## Documentation

- [docs/](docs/README.md) — documentation index, organized in three layers:
  - **Product**: [docs/API.md](docs/API.md) — REST reference
  - **Evaluation report**: [docs/EVALUATION.md](docs/EVALUATION.md) — consolidated measured findings; [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — protocol and threats to validity; [docs/REPRODUCING.md](docs/REPRODUCING.md) — pinned environments and download channels
  - **Survey foundations**: [docs/SURVEY.md](docs/SURVEY.md) — why these eight engines; [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md) — per-checkpoint licensing

Translations of this README: [简体中文](docs/README.zh-CN.md) · [Deutsch](docs/README.de.md) · [Français](docs/README.fr.md) · [Español](docs/README.es.md) · [Italiano](docs/README.it.md). The English document is authoritative; translations may lag.

## Citation

If Voxora or its measurement data is useful in your research, please cite it — see [CITATION.cff](CITATION.cff).

## Contributing

Contributions welcome, including new engines and independent reproductions on other CPUs. See [CONTRIBUTING.md](CONTRIBUTING.md). By participating you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Code: Apache-2.0. Bundled fixtures: CC-BY-4.0. Model weights keep their upstream licenses — this repository distributes none of them.

## Acknowledgements

All heavy lifting is done by the upstream projects: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [Piper](https://github.com/OHF-voice/piper1-gpl), [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR), [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), and [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).
