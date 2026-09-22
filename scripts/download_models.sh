#!/usr/bin/env bash
# Download the model weights for all engines into ./models (or $1).
#
# Channel policy (learned the hard way on a restricted network, see
# docs/REPRODUCING.md): ModelScope first — it hosts official mirrors of the
# Qwen and FunAudioLLM models and is fast; the sherpa release tarballs come
# from GitHub releases with resume-retry because the connection may drop.
#
# Usage: scripts/download_models.sh [models_dir]
set -euo pipefail
MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

MS="https://www.modelscope.cn/models"
GH="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

fetch() { # <url> <dest>
  local url="$1" dest="$2"
  local _attempt
  for _attempt in $(seq 1 30); do
    curl -sfL --continue-at - --speed-limit 5120 --speed-time 40 \
      --connect-timeout 15 -o "$dest" "$url" && return 0
    sleep 2
  done
  echo "FAILED: $url" >&2
  return 1
}

# --- ASR (ONNX, works with the default install) ------------------------------
# SenseVoiceSmall (zh/en/ja/ko/yue)
if [ ! -d sherpa-sense-voice ]; then
  fetch "$GH/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2" sv.tar.bz2
  tar xjf sv.tar.bz2 && rm sv.tar.bz2
  mv sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 sherpa-sense-voice
fi

# Streaming Zipformer (bilingual zh-en)
if [ ! -d sherpa-zipformer-zh-en ]; then
  mkdir -p sherpa-zipformer-zh-en
  cd sherpa-zipformer-zh-en
  base="$MS/pkufool/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/master"
  for f in encoder-epoch-99-avg-1.onnx encoder-epoch-99-avg-1.int8.onnx \
           decoder-epoch-99-avg-1.onnx joiner-epoch-99-avg-1.onnx tokens.txt; do
    fetch "$base/$f" "$f"
  done
  cd ..
fi

# Qwen3-ASR-0.6B int8 ONNX (zengshuishui export)
if [ ! -d qwen3-asr-onnx-int8 ]; then
  mkdir -p qwen3-asr-onnx-int8/tokenizer
  base="$MS/zengshuishui/Qwen3-ASR-onnx/resolve/master"
  for f in model_0.6B/conv_frontend.onnx model_0.6B/encoder.int8.onnx \
           model_0.6B/decoder.int8.onnx; do
    fetch "$base/$f" "qwen3-asr-onnx-int8/$(basename "$f")"
  done
  for f in merges.txt tokenizer_config.json vocab.json preprocessor_config.json chat_template.json; do
    fetch "$base/tokenizer/$f" "qwen3-asr-onnx-int8/tokenizer/$f"
  done
fi

# Fun-ASR-Nano int8 ONNX
if [ ! -d sherpa-onnx-funasr-nano-int8-2025-12-30 ]; then
  fetch "$MS/csukuangfj/asr-models/resolve/master/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2" fn.tar.bz2
  tar xjf fn.tar.bz2 && rm fn.tar.bz2
fi

# --- TTS ----------------------------------------------------------------------
# Piper voices (MIT). Individual files from rhasspy/piper-voices; use hf-mirror
# when the direct HuggingFace CDN is unreachable.
if [ ! -d piper-voices ]; then
  mkdir -p piper-voices
  PV="https://huggingface.co/rhasspy/piper-voices/resolve/main"
  declare -A VOICES=(
    [en/en_US/lessac/medium/en_US-lessac-medium]=1
    [en/en_US/ryan/high/en_US-ryan-high]=1
    [zh/zh_CN/huayan/medium/zh_CN-huayan-medium]=1
    [de/de_DE/thorsten/medium/de_DE-thorsten-medium]=1
    [fr/fr_FR/siwis/medium/fr_FR-siwis-medium]=1
    [ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium]=1
    [es/es_MX/ald/medium/es_MX-ald-medium]=1
  )
  for v in "${!VOICES[@]}"; do
    d="piper-voices/$(dirname "$v")"; mkdir -p "$d"
    fetch "$PV/$v.onnx" "$d/$(basename "$v").onnx"
    fetch "$PV/$v.onnx.json" "$d/$(basename "$v").onnx.json"
  done
fi

# --- Optional PyTorch engines (install the 'torch' extra first) --------------
if [ "${VOXORA_DOWNLOAD_TORCH_MODELS:-0}" = "1" ]; then
  # Qwen3-ASR (transformers checkpoint)
  for f in model.safetensors config.json preprocessor_config.json \
           generation_config.json chat_template.json tokenizer_config.json \
           vocab.json merges.txt; do
    fetch "$MS/Qwen/Qwen3-ASR-0.6B/resolve/master/$f" "Qwen3-ASR-0.6B/$f"
  done
  # Qwen3-TTS CustomVoice (+bundled speech tokenizer)
  for f in model.safetensors config.json generation_config.json \
           preprocessor_config.json tokenizer_config.json vocab.json merges.txt \
           speech_tokenizer/model.safetensors speech_tokenizer/config.json \
           speech_tokenizer/preprocessor_config.json; do
    fetch "$MS/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/resolve/master/$f" \
      "Qwen3-TTS-0.6B/$f"
  done
  # CosyVoice3 (~10 GB)
  echo "CosyVoice3: run  git clone --recursive https://github.com/FunAudioLLM/CosyVoice"
  echo "then download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 from ModelScope into models/Fun-CosyVoice3-0.5B"
fi

# --- Integrity verification ---------------------------------------------------
# Cross-check the ONNX-set weights against the committed manifest (paths are
# relative to the models dir). Mismatches mean upstream re-released an artifact.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v sha256sum >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/../data/checksums.sha256" ]; then
  echo "Verifying weights against data/checksums.sha256 ..."
  if sha256sum -c "$SCRIPT_DIR/../data/checksums.sha256"; then
    echo "All engine weights verified."
  else
    echo "WARNING: checksum mismatches listed above — upstream artifacts changed." >&2
  fi
fi

echo "Done. Weights in: $(pwd)"
