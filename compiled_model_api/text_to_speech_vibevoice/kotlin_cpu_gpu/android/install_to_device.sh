#!/usr/bin/env bash
# Push the VibeVoice LiteRT graphs + assets to the app's EXTERNAL files dir (too big to bundle).
# adb writes there directly (no /data/local/tmp double-copy), so peak storage = the file set
# itself (~1.7 GB) — important on near-full devices. Get the files from Hugging Face or build them
# with conversion/build_vibevoice.py, then:
#   ./install_to_device.sh <dir-with-the-files>   (default: current dir)
#
# The first app launch is expected to fail with "Model not found" until this has been run.
set -e
PKG=com.google.ai.edge.examples.text_to_speech_vibevoice
DIR="${1:-.}"
DEST="/sdcard/Android/data/$PKG/files"
# The LMs and the σ-VAE decoder ship as fp32 graphs on CPU (Mali ML Drift miscomputes the decoder
# and rejects the LMs; fp16 collapses the LMs on ARM XNNPACK). Only the diffusion head is fp16/GPU.
FILES=(
    vv_base_lm_kv_fp32.tflite
    vv_tts_lm_kv_fp32.tflite
    vv_diffhead_fp16.tflite
    vv_decoder_fp32.tflite
    embed_tokens.f16
    glue.f32
    voice_en-Emma_woman.bin
)
adb shell mkdir -p "$DEST"
for F in "${FILES[@]}"; do
    echo "pushing $F ..."
    adb push "$DIR/$F" "$DEST/$F"
done
adb shell ls -la "$DEST"
echo "done — launch the VibeVoice TTS app."
