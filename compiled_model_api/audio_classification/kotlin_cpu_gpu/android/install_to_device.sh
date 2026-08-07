#!/usr/bin/env bash
# Push the 2 wav2vec2-KWS tflites into the app's private filesDir (too big to bundle).
# Build with ../../conversion/build_w2v2_split.py or get from Hugging Face
# (litert-community/wav2vec2-keyword-spotting), then:
#   ./install_to_device.sh <dir-with-the-tflites>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.audio_classification
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in w2v2_frontend_fp16.tflite w2v2_head_fp16.tflite; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
    adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Wav2Vec2 KWS app."
