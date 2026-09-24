#!/usr/bin/env bash
# Push the dis.tflite into the app's private filesDir.
# Build with ../conversion (build_ormbg.py) or download from Hugging Face
# (litert-community/DIS-ISNet-LiteRT), then:
#   ./install_to_device.sh <dir-with-dis.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.dis
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing dis.tflite ..."
adb push "$DIR/dis.tflite" "/data/local/tmp/dis.tflite"
adb shell chmod 644 "/data/local/tmp/dis.tflite"
adb shell run-as $PKG cp "/data/local/tmp/dis.tflite" "files/dis.tflite"
adb shell rm "/data/local/tmp/dis.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the ormbg app."
