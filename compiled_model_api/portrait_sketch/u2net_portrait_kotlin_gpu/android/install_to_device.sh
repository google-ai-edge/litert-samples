#!/usr/bin/env bash
# Push the portrait.tflite into the app's private filesDir.
# Build with ../conversion (build_ormbg.py) or download from Hugging Face
# (litert-community/U2Net-Portrait-Sketch-LiteRT), then:
#   ./install_to_device.sh <dir-with-portrait.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.portrait
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing portrait.tflite ..."
adb push "$DIR/portrait.tflite" "/data/local/tmp/portrait.tflite"
adb shell chmod 644 "/data/local/tmp/portrait.tflite"
adb shell run-as $PKG cp "/data/local/tmp/portrait.tflite" "files/portrait.tflite"
adb shell rm "/data/local/tmp/portrait.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the ormbg app."
