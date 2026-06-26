#!/usr/bin/env bash
# Push the 4 Mimi tflites + RVQ weights into the app's private filesDir (too big to bundle).
# Get them from Hugging Face (litert-community/Mimi) or build with the scripts in ../../conversion,
# then:  ./install_to_device.sh <dir-with-the-files>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.audio_codec
DIR="${1:-.}"
MODELS=(mimi_enc_conv_fp16.tflite mimi_enc_tx_fp16.tflite mimi_dec_tx_fp16.tflite mimi_deconly_fp16.tflite mimi_rvq.bin)
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in "${MODELS[@]}"; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
    adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Mimi Codec app."
