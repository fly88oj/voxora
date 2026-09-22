# Reproducing the results

## 1. Default path (ONNX engines)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[onnx,dev]"
scripts/download_models.sh models          # ~3 GB
voxora run --engine sensevoice --audio-dir data/fixtures -o my_sensevoice.json
```

Each output JSON embeds its own environment fingerprint; diff it against [`data/environment.json`](../data/environment.json) — the reference machine record plus the exact per-engine library pins that produced `data/results/` — to understand which differences are hardware vs software. After downloading, the script verifies every ONNX-set weight against the committed [`data/checksums.sha256`](../data/checksums.sha256) manifest, so upstream re-releases are detected.

## 2. Full stack (PyTorch engines)

The torch engines were measured in isolated environments with the following pins (the official packages resolve their own transformers/torchaudio constraints):

| Engine | Package | Backend pins |
|---|---|---|
| Qwen3-ASR | `qwen-asr==0.0.6` | transformers 4.57.6, torch 2.14 **CPU wheel** |
| Qwen3-TTS | `qwen-tts==0.1.1` | transformers 4.57.3, torch 2.11 + torchaudio 2.11 CPU |
| CosyVoice3 | source clone `FunAudioLLM/CosyVoice` | torch 2.10 CPU, transformers 4.51.3 |

**Install CPU-only torch wheels** from your platform mirror (e.g. `https://mirrors.aliyun.com/pytorch-wheels/cpu/`) — the default PyPI torch pulls CUDA libraries that are useless here and can change thread-pool behavior.

## 3. Benchmark protocol

Follow [METHODOLOGY.md](METHODOLOGY.md) exactly: timing wraps the engine call only, 16 kHz mono input, threads as recorded in each result file. Load-recording and pinning rules (compare runs at comparable load; pin only to exclusive cores) are specified in [METHODOLOGY.md §6](METHODOLOGY.md#6-optimization-sweep); operationally:

```bash
taskset -c 0-7 voxora run --engine sensevoice --audio-dir data/fixtures \
  --repeat 5 -o sense.json        # only meaningful if cores 0-7 are quiet
```

To regenerate the round-trip corpus, synthesize `data/corpus.json` with the TTS engines, normalize to 16 kHz mono (`sox in.wav -r 16000 -c 1 out.wav`), and transcribe with each ASR engine.

## 4. Download channels (restricted networks)

`scripts/download_models.sh` encodes the channel policy that worked on a network where several CDNs are throttled:

1. **ModelScope first** for Qwen / FunAudioLLM checkpoints (official mirrors, fastest, resumable): `https://www.modelscope.cn/models/<org>/<repo>/resolve/master/<file>`
2. **GitHub releases** (sherpa-onnx tarballs) via curl with `--continue-at -` and retry loops.
3. **HuggingFace** direct or `https://hf-mirror.com` for Piper voices; if large-file transfers stall, sequential downloads with resume eventually complete (parallelism triggered rate limiting in the study network).

## 5. Known issues and their workarounds

| Issue | Workaround in this repo |
|---|---|
| torchaudio ≥ 2.9 routes all I/O through `torchcodec`, whose bundled libs fail to resolve symbols against CPU-only torch | `cosyvoice` engine monkey-patches `torchaudio.load` with a soundfile-backed loader (see `engines/cosyvoice.py`) |
| CosyVoice's `fp16/load_jit/load_trt` flags are hard-disabled without CUDA | CPU route always runs fp32 (see [evaluation report §6](EVALUATION.md#6-optimization-findings)) |
| setuptools ≥ 81 ships without `pkg_resources`, which CosyVoice imports | pin `setuptools<81` in that environment |
| `piper-tts` streaming chunk API differs between 1.2 and 1.3 | the engine consumes the chunk iterable from the installed version (both work) |

## 6. Adding an engine

1. Subclass `AsrEngine`/`TtsEngine` (see `engines/sensevoice.py`), implement `load()` + the modality method.
2. Decorate with `@register`, set metadata (`license_note`, `import_probe`, `model_subdir`).
3. Add an entry to `scripts/download_models.sh` and a row in `docs/MODEL_LICENSES.md`.
4. Add tests: unit tests with a stub; an integration test marked `@pytest.mark.integration`.
