#!/usr/bin/env bash
# Push the twinlite.tflite into the app's private filesDir.
# Build with ../conversion (build_ormbg.py) or download from Hugging Face
# (litert-community/TwinLiteNet-LiteRT), then:
#   ./install_to_device.sh <dir-with-twinlite.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.twinlitenet
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing twinlite.tflite ..."
adb push "$DIR/twinlite.tflite" "/data/local/tmp/twinlite.tflite"
adb shell chmod 644 "/data/local/tmp/twinlite.tflite"
adb shell run-as $PKG cp "/data/local/tmp/twinlite.tflite" "files/twinlite.tflite"
adb shell rm "/data/local/tmp/twinlite.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the ormbg app."
