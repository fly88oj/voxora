# Evaluation report

This report presents the findings behind Voxora's [performance reference](../README.md#performance-reference). Every number is traceable to a committed raw-result file (see [Appendix](#10-data-map)); the measurement protocol, definitions, and threats to validity are specified in [METHODOLOGY.md](METHODOLOGY.md).

## 1. Questions

1. Which open ASR/TTS engines are deployable on CPU-only hardware, at what speed?
2. How accurate is their output, measured identically across engines?
3. Which configuration knobs matter (dtype, threads, quantization, batching)?
4. How stable are the numbers — and what actually drives their variance?

## 2. Environment

One reference machine: AMD Ryzen AI MAX+ 395 (16C/32T, Zen 5), 93 GB RAM, Ubuntu 24.04, Python 3.12. Per-engine library pins: [`data/environment.json`](../data/environment.json). The host is shared and non-exclusive — this shaped finding 4.6.

## 3. ASR performance

Full-corpus measurements (re-measured 2026-09-23 after the fixture repair described in [METHODOLOGY.md §8](METHODOLOGY.md#8-data-provenance-and-erratum); the reference host carried its usual ambient load — see §7), 16 kHz mono, wall-clock around the engine call ([`data/results/`](../data/results)). Tuned values are 5-pass means from [`data/optimization/tuned/`](../data/optimization/tuned) (cv ≤ 3% except where noted).

| Engine | Route | RTF (default) | RTF (tuned) | Peak RSS |
|---|---|---|---|---|
| SenseVoiceSmall | ONNX int8 | 0.0131 | 0.0103 (cv 10%: one quiet pass at 0.0084) | 1.6 GB |
| SenseVoiceSmall | ONNX fp32 | 0.0159 | — | 1.6 GB |
| SenseVoiceSmall | funasr (PyTorch) | 0.0360 | — | 3.2 GB |
| Zipformer (streaming) | ONNX int8 | 0.0355 | 0.0324 | 0.5 GB |
| Zipformer (streaming) | ONNX fp32 | 0.0417 | — | 0.5 GB |
| Qwen3-ASR-0.6B | ONNX int8 | 0.1082 | 0.0999 | 2.4 GB |
| Qwen3-ASR-0.6B | PyTorch fp32 | 0.6281 | 0.4696 | 6.0 GB |
| Qwen3-ASR-0.6B | PyTorch bf16 | — | 0.1202 / **0.0699 (batch8)** | 6.0 GB |
| Fun-ASR-Nano | ONNX int8 | 0.0973 | 0.0830 | 2.2 GB |

## 4. ASR accuracy (TTS→ASR round-trip)

Primary error per language (CER for zh/ja/ko, WER otherwise; engines score only languages they claim to support — cells at ~1.0 mean *unsupported*, not misrecognized):

| Engine | zh | zh+en mixed | en | ja | ko | de | fr | es | ru |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR (PyTorch) | 0 | 0 | 0 | 1% | 0 | 0 | 0 | 0 | 6% |
| Qwen3-ASR (ONNX int8) | 0.4% | 0 | 0 | 1% | 0 | 0 | 0 | 0 | 6% |
| SenseVoice (ONNX int8) | 2% | 7% | 0 | 1% | 0 | — | — | — | — |
| SenseVoice (funasr) | 1% | 3% | 0 | 1% | 0 | — | — | — | — |
| Fun-ASR-Nano | 2% | 6% | 0 | 1% | — | — | 75% | — | — |
| Zipformer int8 | 2% | 16% | 15% | — | — | — | — | — | — |

Notes: the Russian 6% is a single `ё`/`е` normalization mismatch (`днём`/`днем`), not a recognition failure; the 1% ja cells are one character (`良い`/`いい`) and also occur identically across engines — multithreaded float reductions make greedy decoding flip borderline characters between runs. Zipformer's streaming warm-up loses word onsets (fp32 en 12%, int8 15%). Zipformer's streaming warm-up loses word onsets (fp32 en 14.4% vs int8 19.1%). On real code-switched classroom speech, both Qwen3-ASR routes produced fully correct mixed zh/en transcripts with punctuation and ITN; SenseVoice and Fun-ASR-Nano captured the content with rougher casing. Dialect check: CosyVoice3's Cantonese and Sichuan synthesis was transcribed back correctly by SenseVoice (both routes) and Fun-ASR-Nano — the dialect capabilities are real.

## 5. TTS performance and intelligibility

| Engine | RTF (default fp32) | RTF (tuned) | First audio | Peak RSS |
|---|---|---|---|---|
| Piper | 0.04–0.06 | — | 150–263 ms | 0.4 GB |
| Qwen3-TTS-0.6B | 4.85 | **1.40 (bf16)** | ~22 s (streaming mode degenerates to full synthesis on CPU) | 5.7 GB |
| CosyVoice3-0.5B | 3.3–6.0 | — | 8–33 s per chunk (streaming mode) | 7.7 GB |

Intelligibility (best ASR referee per TTS, round-trip): CosyVoice3 0% (zh, n=2) · Qwen3-TTS 0.74% (9 languages, n=17) · Piper 1.04% (6 languages, n=12) — all near-perfect; naturalness differences need human listening (samples: `data/fixtures`, methodology §7).

## 6. Optimization findings

Raw sweep data: [`data/optimization/`](../data/optimization).

1. **bf16 is a free 3.5–3.9× on Zen 4/5** — Qwen3-ASR 0.470→0.120, Qwen3-TTS 4.85→1.40; transcripts verified byte-identical, TTS round-trip unchanged (0.5% vs 0.74%, [`tuned/qwen_tts_bf16_roundtrip.json`](../data/optimization/tuned/qwen_tts_bf16_roundtrip.json)).
2. **Batching adds ~1.7×** on top for throughput (Qwen3-ASR batch-of-8: 0.120→0.070); combined with bf16: 0.470→0.070 (6.7×).
3. **Thread oversubscription is catastrophic**: 32 threads on 16 cores degrades PyTorch engines 5–46× (Qwen3-TTS fp32: 20 s→134 s per utterance). ONNX engines prefer 4–8 threads.
4. **int8 ONNX is the best deployment route for LLM-ASR**: 0.108 vs 0.628 PyTorch fp32 (~6×), 60% less memory, near-identical accuracy (§4).
5. **Piper voice tier** (medium vs high) changes RTF 0.081→0.149 — both far beyond realtime; choose by quality.
6. **CosyVoice has no CPU dtype knob** (fp16/jit force-disabled upstream without CUDA); threads are the only lever.

## 7. Stability findings

Raw stability data: [`data/stability/`](../data/stability); protocol: [METHODOLOGY.md §6](METHODOLOGY.md#run-to-run-stability-protocol).

1. Within-process precision: cv 1.3–5.2% across 5-pass repeats (all ONNX engines, pinned or not; the re-measured tuned references in `tuned/` show cv ≤ 3% except SenseVoice's 10% — one pass landed in a quiet window).
2. **Ambient CPU load — not elapsed time — drives variance.** RTF tracked the recorded 1-minute loadavg almost linearly across runs minutes apart (e.g. SenseVoice 0.0080 @ load 5.4; Zipformer pinned to 4 contended cores 0.083 @ load 15.9).
3. `taskset` pinning helps only when the pinned cores are quiet; pinned-under-load was up to 2.5× slower than unpinned.
4. TTS cold first pass is a consistent outlier (Piper 0.105 vs ~0.038 warm) — report warm passes.
5. Rankings between engines, measured back-to-back, never changed across any condition tried.

## 8. Deployment guidance (derived)

| Scenario | Recommendation |
|---|---|
| Realtime zh/en/ja/ko first-pass ASR | SenseVoice ONNX int8, 8 threads |
| Multilingual (30 langs) ASR | Qwen3-ASR ONNX int8 (deploy); PyTorch bf16+batch if toolkit features needed |
| Chinese-dialect ASR | Fun-ASR-Nano ONNX int8 (zh/en/ja + 7 dialect groups) |
| Edge / low-memory streaming | Zipformer int8, 4 threads (0.5 GB RSS) |
| Realtime TTS | Piper (only realtime tier; <300 ms first audio) |
| High-quality TTS, async | Qwen3-TTS bf16 (RTF 1.4) or CosyVoice3 (dialects/cloning, RTF 3.3–6.0) |

## 9. Limitations

Full list in [METHODOLOGY.md §7 and §9](METHODOLOGY.md#7-threats-to-validity): round-trip circularity, one machine, shared non-exclusive CPU (load-dependent absolute RTF), small corpus, no human MOS, and the not-measured set (Fun-ASR-Nano `-hf` checkpoint, Qwen3-TTS-1.7B, Fun-ASR-MLT-Nano, concurrency).

## 10. Data map

| Claim family | Raw files |
|---|---|
| §3 ASR performance | `data/results/asr_*.json` |
| §4 round-trip accuracy | `data/results/eval.json` (+ per-engine `asr_*.json` hypotheses) |
| §5 TTS performance / first audio | `data/results/tts_*.json` |
| §6 optimization | `data/optimization/*.json` (incl. `qwen_asr_bf16_verify.json`, `qwen_tts_bf16_gen.json`) |
| §7 stability & load | `data/stability/**` (5-pass repeats, pinned runs with recorded affinity+loadavg, torch repeats) |
| Environment | `data/environment.json` |
