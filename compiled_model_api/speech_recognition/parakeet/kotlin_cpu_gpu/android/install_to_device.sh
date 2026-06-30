#!/usr/bin/env bash
# Push the Parakeet model into the app's private filesDir (too big to bundle in the APK).
# Build with ../../conversion/build_parakeet_ship.py, or get it from Hugging Face
# (litert-community/Parakeet-tdt-ctc-110m-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.parakeet
DIR="${1:-.}"
M=parakeet_ship_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Parakeet ASR app."
