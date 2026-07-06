#!/usr/bin/env bash
# Push the dewarp.tflite into the app's private filesDir.
# Build with ../conversion (build_ormbg.py) or download from Hugging Face
# (litert-community/DewarpNet-LiteRT), then:
#   ./install_to_device.sh <dir-with-dewarp.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.dewarpnet
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing dewarp.tflite ..."
adb push "$DIR/dewarp.tflite" "/data/local/tmp/dewarp.tflite"
adb shell chmod 644 "/data/local/tmp/dewarp.tflite"
adb shell run-as $PKG cp "/data/local/tmp/dewarp.tflite" "files/dewarp.tflite"
adb shell rm "/data/local/tmp/dewarp.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the ormbg app."
