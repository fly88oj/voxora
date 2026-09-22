# Engine survey and selection rationale

Voxora is a CPU-only ASR/TTS API service; its engines are its product surface. This document records **which engines were selected for the service, why, and what was left out**, so the scope of every performance claim is explicit. The selection was driven by one practical question — *"which open ASR/TTS models actually run well on CPU-only hardware, and how do they compare when measured identically?"* — and the same measurement harness produced the performance references shipped in `data/`.

## Selection criteria

1. **Fully local inference** with publicly downloadable weights; no vendor API fallback.
2. **CPU-viable runtime** exists today: ONNX/C runtime or a PyTorch CPU path.
3. **Language coverage**: engines were chosen to jointly cover the zh/en/ja/ko core plus the widest multilingual footprint.
4. **Licenses compatible with internal deployment at minimum** (see [MODEL_LICENSES.md](MODEL_LICENSES.md)).

## Included engines

| Engine | Role in the study | Why included |
|---|---|---|
| SenseVoiceSmall | zh/en/ja/ko/yue fast tier | Strong published zh results; ~230 MB; the fastest engine measured (RTF ≈ 0.01) |
| Streaming Zipformer (zh-en) | edge/streaming tier | The canonical streaming transducer baseline; 0.5 GB RAM class |
| Qwen3-ASR-0.6B (ONNX int8 + PyTorch) | multilingual quality tier | 30 languages + 22 zh dialects; both deployment routes measured head-to-head |
| Fun-ASR-Nano | Chinese-dialect tier | Explicit coverage of 7 dialect groups; Apache-2.0 weights |
| Piper | realtime TTS tier | The standard low-resource TTS; per-voice licensing is clean (MIT) |
| Qwen3-TTS-0.6B | multilingual quality TTS | 10 languages, 9 voices; the PyTorch CPU ceiling made visible |
| CosyVoice3 | zh expressiveness tier | Zero-shot cloning + dialect instruct control (dialect outputs verified by independent ASR round-trip — [evaluation report §4](EVALUATION.md#4-asr-accuracy-ttsasr-round-trip)) |

## Evaluated landscape, not included (with reasons)

| Candidate | Reason not benchmarked |
|---|---|
| Whisper family (openai-whisper, faster-whisper, whisper.cpp) | Well-benchmarked elsewhere; SenseVoice/Qwen3-ASR occupy its CPU niche in this comparison. Integrating `from_whisper` in sherpa-onnx is trivial if needed (see REPRODUCING.md §6). |
| Vosk / Kaldi | Legacy architecture; streaming niche already covered by Zipformer. |
| Kokoro / StyleTTS2 / MeloTTS | Strong TTS quality, but licensing or runtime maturity on CPU at study time made Piper + LLM-TTS the cleaner comparison pair. |
| Moonshine, Dolphin/Canary (sherpa-supported) | Niche (edge-English, multi-seat ASR); outside the language matrix. |
| Fun-ASR-MLT-Nano (31-language) | Not yet measured — flagged as future work; the zh/en/ja Nano checkpoint was measured instead. |
| Qwen3-TTS-1.7B | Same architecture family as the measured 0.6B; expected strictly slower on CPU. |
| Cloud-hosted TTS (e.g. DashScope realtime) | Violates the fully-local criterion; its latency figures (97 ms streaming) must not be compared with CPU numbers. |

## Related claims checked during the study

External claims were assessed against our measurements (the measured counterparts live in the [evaluation report](EVALUATION.md); this section records only the transferability reasoning):

- *"Qwen3-ASR int8 runs at RTF ≈ 0.71 on an Intel N100"* (community deployment note): consistent with our measured Zen 5 desktop RTF ≈ 0.10 — an N100-class core is plausibly ~7× slower, so the claim is internally coherent.
- *"Qwen3-TTS reaches 0.7–0.86× realtime on Apple Silicon (4 threads, C implementation)"*: not transferable to x86 PyTorch CPU — ISA and memory bandwidth dominate this workload; our measured gap is in [evaluation report §5](EVALUATION.md#5-tts-performance-and-intelligibility).
- *"CosyVoice3 streams with ≈150 ms first audio"*: a GPU/optimized-runtime figure; on CPU PyTorch the streaming mode degrades (see evaluation report §5).
