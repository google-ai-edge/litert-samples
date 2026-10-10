#!/usr/bin/env bash
# Push the ufld.tflite into the app's private filesDir.
# Build with ../conversion (build_ufld.py) or download from Hugging Face
# (litert-community/Ultra-Fast-Lane-Detection-LiteRT), then:
#   ./install_to_device.sh <dir-with-ufld.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.ufld
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing ufld.tflite ..."
adb push "$DIR/ufld.tflite" "/data/local/tmp/ufld.tflite"
adb shell chmod 644 "/data/local/tmp/ufld.tflite"
adb shell run-as $PKG cp "/data/local/tmp/ufld.tflite" "files/ufld.tflite"
adb shell rm "/data/local/tmp/ufld.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the ufld app."
