#!/usr/bin/env bash
# Push the ormbg tflite into the app's private filesDir.
# Build with ../conversion (build_ormbg.py) or download from Hugging Face
# (litert-community/ormbg-LiteRT), then:
#   ./install_to_device.sh <dir-with-ormbg.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.ormbg
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing ormbg.tflite ..."
adb push "$DIR/ormbg.tflite" "/data/local/tmp/ormbg.tflite"
adb shell chmod 644 "/data/local/tmp/ormbg.tflite"
adb shell run-as $PKG cp "/data/local/tmp/ormbg.tflite" "files/ormbg.tflite"
adb shell rm "/data/local/tmp/ormbg.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the ormbg app."
