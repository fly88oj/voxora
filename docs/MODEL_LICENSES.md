# Model and data licensing

**This repository distributes no model weights.** `scripts/download_models.sh` fetches them from their official sources. Each engine combines *runtime code*, *model weights*, and sometimes *voice data* with independent licenses — verify all three before redistribution. The table below reflects the licenses as stated by each upstream project at the time of writing (2026-09); re-check before production use.

| Engine component | Code license | Weights license | Source |
|---|---|---|---|
| sherpa-onnx runtime | Apache-2.0 | — | github.com/k2-fsa/sherpa-onnx |
| SenseVoiceSmall | Apache-2.0 | Apache-2.0 (model card) | github.com/FunAudioLLM/SenseVoice |
| Zipformer bilingual zh-en | Apache-2.0 | Apache-2.0 (k2-fsa release) | sherpa-onnx releases |
| Qwen3-ASR-0.6B (original + ONNX export) | Apache-2.0 | Apache-2.0 (Qwen model card) | QwenLM/Qwen3-ASR; ONNX export by community (zengshuishui) |
| Fun-ASR-Nano | Apache-2.0 | Apache-2.0 (model card) | github.com/FunAudioLLM/Fun-ASR |
| Piper runtime | MIT | — | github.com/OHF-voice/piper1-gpl |
| Piper voices (bundled set) | MIT | MIT (rhasspy/piper-voices) | huggingface.co/rhasspy/piper-voices |
| qwen-asr / qwen-tts packages | Apache-2.0 | Apache-2.0 (Qwen model cards) | QwenLM |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | Apache-2.0 | Qwen model card — check voice-cloning clauses | QwenLM/Qwen3-TTS |
| CosyVoice (code) | Apache-2.0 (repo); **ttsfrd dependency is non-commercial** — the repo works without it (wetext frontend) | — | github.com/FunAudioLLM/CosyVoice |
| Fun-CosyVoice3-0.5B | — | Model card on ModelScope (research-oriented disclaimer in repo README) | FunAudioLLM |

## Bundled data in this repository

- `data/fixtures/*.wav` — recordings either published by the upstream projects as sample assets (Qwen3-ASR sample) or synthesized locally during the study (Piper/Qwen3-TTS outputs of `data/corpus.json`).
- `data/results/`, `data/optimization/`, `data/environment.json` — measurement outputs produced by this study. Licensed CC-BY-4.0 (see `data/LICENSE`).

## Practical notes

1. **Commercial redistribution** requires the *weights* to allow it — Apache-2.0 model cards generally do; CosyVoice3's README carries a research-oriented disclaimer, so treat it as research-only unless clarified.
2. **Voice cloning** (CosyVoice zero-shot, Qwen3-TTS-Base) multiplies consent obligations for the *cloned* voice — obtain and document speaker authorization independently of any software license.
3. **Piper voices** vary per voice; the seven bundled here are MIT, others in the upstream repository may differ.
