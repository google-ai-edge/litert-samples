#!/usr/bin/env bash
# Push the Matcha-TTS LiteRT graphs into the app's private filesDir (too big to bundle).
# Get them from Hugging Face (mlboydaisuke/Matcha-TTS-LiteRT) or build with the scripts in
# this dir, then:
#   ./scripts/install_to_device.sh <dir-with-the-tflites>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.text_to_speech
DIR="${1:-.}"
MODELS=(
    matcha_textenc_fp16.tflite
    matcha_decoder_fp16.tflite
    matcha_vocoder_fp16.tflite
    dp_g2p_matcha_fp16.tflite
)
for M in "${MODELS[@]}"; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG mkdir files 2>/dev/null || true
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Matcha TTS app."
