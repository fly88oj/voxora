# Measurement methodology

This document specifies exactly how every number in [`data/`](../data) was produced. It is the single source of truth for the protocol; the API and CLI implement it, and the unit tests pin its normalization rules.

## 1. Scope and definitions

- **Target**: pure-CPU inference (no GPU/accelerator calls at any point). PyTorch engines run CPU wheels; ONNX engines use the CPU execution provider.
- **RTF (real-time factor)**: `processing_s / audio_duration_s`. RTF < 1 is faster than realtime. RTF can exceed 1 (hypothesis longer than reference).
- **Processing time**: wall clock around the engine call only. File decoding, resampling, HTTP handling, and Python request plumbing are excluded. Model load (cold start) is reported separately (`load_s`) and never included in RTF.
- **Peak RSS**: process `ru_maxrss` of the whole Python process (includes interpreter + libraries), measured per engine in an isolated process. It is an upper bound on model memory, not the model size.
- **First audio latency (TTS)**: time from request start to the first audio chunk emitted (`stream=True` generators; for engines without streaming it equals total synthesis time).

## 2. Reference environment

One machine produced all bundled results (fingerprint: [`data/environment.json`](../data/environment.json)):

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI MAX+ 395, 16 cores / 32 threads (Zen 5) |
| Memory | 93 GB |
| OS | Ubuntu 24.04 (glibc 2.39), kernel 7.0 |
| Python | 3.12 |

Engines ran in isolated virtual environments with per-engine pins (exact versions in [REPRODUCING.md](REPRODUCING.md)); ONNX engines used `sherpa-onnx` 1.13.8 and PyPI `piper-tts`.

## 3. Corpus

- **Round-trip corpus** ([`data/corpus.json`](../data/corpus.json)): 17 reference sentences across 9 language groups (zh, zh+en mixed, en, de, fr, ru, es, ja, ko — two sentences each except the mixed group with one), synthesized by the TTS engines under test into **31 wav files** (Piper 12 files / 6 languages, Qwen3-TTS 17 files / 9 languages, CosyVoice3 2 files / zh), normalized to 16 kHz mono. Scoring joins on the 31 files.
- **Real-speech samples** (no published reference text, compared side by side but not scored): the official Qwen3-ASR sample utterances, the sherpa-onnx Zipformer `test_wavs` classroom recordings (natural zh-en code-switching), and one telephone-band 8 kHz file. The Zipformer runs record two honest failures on this set: the 8 kHz file (sr mismatch) and the 15 s English sample (streaming-decoder error) — visible as `error` items in the raw files.

## 4. Accuracy protocol: TTS→ASR round-trip

Accuracy in `data/results/eval.json` is **round-trip error**: synthesize a known text with a TTS engine, transcribe with an ASR engine, and score against the known text. It measures *"can the ASR engine understand speech produced by the TTS engines on this hardware"*, which is exactly the property a local voice pipeline needs — but it is **not** comparable to public benchmarks (LibriSpeech, AISHELL, …) and must not be cited as WER/CER on those datasets.

Scoring rules (implemented in [`voxora/scoring.py`](../src/voxora/scoring.py), pinned by `tests/test_scoring.py`):

1. Engine tag tokens (SenseVoice `<|zh|><|NEUTRAL|>…`) are stripped.
2. Case-folded.
3. Apostrophes are deleted (intra-word in EN/FR/DE contractions: `isn't → isnt`).
4. Remaining punctuation is dropped for CER (all whitespace removed) and mapped to spaces for WER (word boundaries preserved).
5. **Primary metric**: CER for zh/yue/ja/ko, WER otherwise.
6. Empty normalized reference scores 1.0. No unicode folding: Russian `ё` vs `е` counts as an error (a deliberate, documented strictness).

## 5. Streaming simulation

Streaming engines (Zipformer) are fed 100 ms chunks with decode calls between chunks — the same loop a live microphone would drive. Per-file RTF therefore includes streaming decode overhead. Worst-case single-chunk latency and tail (final flush) latency were measured separately in the original study and are visible in the raw result files.

## 6. Optimization sweep

[`data/optimization/`](../data/optimization) holds the sweep over configuration knobs, on a fixed 8-file subset (~55 s of audio) per configuration:

- **Thread count**: the full 1/2/4/8/16/32 ladder for SenseVoice; 2+ for Zipformer; 4+ for the LLM-decoder engines; 16/32 for PyTorch engines.
- **dtype**: fp32 vs bf16 (PyTorch engines; AMD Zen 4/5 expose AVX512-BF16).
- **Batching**: single-request vs batch-of-8 through one `transcribe()` call.
- **Decoding**: greedy vs modified beam search (Zipformer).
- **Quantization**: int8 vs fp32 ONNX.
- **Piper voice tier**: medium vs high.

bf16 outputs were verified before adoption (procedure): Qwen3-ASR bf16 transcripts were compared byte-wise against fp32 on the verification subset, and Qwen3-TTS bf16 audio was scored via ASR round-trip. bf16 is *not* claimed lossless in general; adopt it only after spot-checking your own traffic. The measured speedups and best configurations are reported in the [evaluation report §6](EVALUATION.md#6-optimization-findings).

### Run-to-run stability protocol

The reference host is shared and non-exclusive (≈2 cores of ambient services); this subsection defines the replication protocol, the findings live in [evaluation report §7](EVALUATION.md#7-stability-findings).

- **Replication**: `voxora run --repeat N` (per-pass RTF; pass 1 = cold, included); fresh-process replication for the PyTorch engines; pinned vs unpinned comparison via `taskset -c`.
- **Recording**: result files produced by the `voxora` CLI embed `environment.runtime.cpu_affinity` and `environment.runtime.loadavg_1min` automatically (the 2026-09-22 legacy files in `data/results` and `data/optimization` predate this field; `data/stability/pinned/` and everything after carry it) — compare only runs with comparable recorded load.
- **Pinning rule**: pin to physical cores only when those cores are exclusive; on a contended host prefer repeated unpinned runs and report the lowest-load run as the quiet-machine estimate. System-level isolation (isolcpus / systemd `CPUAffinity` on interfering services) is the only true exclusivity; it requires root and was unavailable on the reference host.
- **Interpretation**: treat absolute RTFs from non-exclusive machines as load-dependent (−15%…+150% observed); relative rankings measured back-to-back are stable.

## 7. Threats to validity

| Threat | Status |
|---|---|
| Circular round-trip scoring | **Inherent** — results measure TTS-engine-specific intelligibility; treat per-language numbers as a cross-engine comparison under identical conditions, not absolute accuracy. |
| Single machine | All bundled numbers are one CPU (Zen 5 desktop). Relative rankings may shift on other ISAs (notably Apple Silicon, which has different bf16/memory-bandwidth behavior for the PyTorch engines). |
| Non-exclusive CPU | Measurements share the host with ≈2 cores of ambient services; RTF depends on the ambient load at run time (−15%…+150% observed, §6 *Run-to-run stability*). Load and affinity are recorded by the `voxora` CLI (see §6); compare runs at comparable recorded load. |
| Small corpus | 31 scored utterances (17 reference sentences); per-language cells rest on 2–4 files. Aggregate rankings are robust; single-language differences of a few points are not. |
| No human MOS | TTS naturalness was not evaluated by listeners; intelligibility is the only quality proxy. |
| Software pinning | ONNX engines pinned via one sherpa-onnx wheel; PyTorch engines via the official `qwen-asr`/`qwen-tts` packages (see REPRODUCING.md). Different versions may change absolute numbers. |

## 8. Data provenance and erratum

During the per-file review of 2026-09-23, the Qwen English sample (`asr_en_16k.wav`, a real-speech extra, not a round-trip file) was found corrupted: 7.53 s of extraneous audio had been appended to its original 15.05 s, with a self-consistent header — so `wave`-based readers (the original bench scripts) processed 22.58 s while `soundfile`-based readers (the `voxora` runner) processed 15.05 s.

Consequences and handling:

1. **Full-corpus ASR aggregates** (all `data/results/asr_*.json` from 2026-09-22) included the inflated file → aggregates understated by ≈3.3%. All eight ASR configurations were **re-measured on 2026-09-23** after repairing the file; the files in `data/results/` are the corrected measurements.
2. **Optimization sweep** (2026-09-22, 8-file subset containing the same file): every configuration shared the inflated 62.65 s denominator, so **relative comparisons (thread ladders, dtype, batching) are unaffected**, but absolute sweep RTFs were understated by the factor 62.65/55.13 = 0.880. The sweep files are kept for the ladder shape; the tuned reference values quoted by the evaluation report come from fresh 5-pass re-measurements in [`data/optimization/tuned/`](../data/optimization/tuned) (with recorded load and cv).
3. Round-trip accuracy (`eval.json` scores) was unaffected — the corrupted file is not part of the scored corpus, and TTS measurements never touched it.

## 9. What was *not* measured

- Fun-ASR-Nano via its transformers `-hf` checkpoint (weights could not be fetched on the study network; the sherpa int8 ONNX route was measured instead).
- Qwen3-TTS-1.7B; Fun-ASR-MLT-Nano (31-language variant).
- Multi-stream concurrency; serving under load.
- Automatic speech quality models (e.g. PESQ/UTMOS) on TTS output.

Contributions re-running the suite on other CPUs (especially Intel / Apple Silicon) are welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md).
