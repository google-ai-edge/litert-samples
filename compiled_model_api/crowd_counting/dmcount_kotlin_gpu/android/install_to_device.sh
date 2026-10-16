#!/usr/bin/env bash
# Push the dmcount.tflite into the app's private filesDir.
# Build with ../conversion (build_dmcount.py) or download from Hugging Face
# (litert-community/DM-Count-Crowd-LiteRT), then:
#   ./install_to_device.sh <dir-with-dmcount.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.crowdcount
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing dmcount.tflite ..."
adb push "$DIR/dmcount.tflite" "/data/local/tmp/dmcount.tflite"
adb shell chmod 644 "/data/local/tmp/dmcount.tflite"
adb shell run-as $PKG cp "/data/local/tmp/dmcount.tflite" "files/dmcount.tflite"
adb shell rm "/data/local/tmp/dmcount.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the crowd counting app."
