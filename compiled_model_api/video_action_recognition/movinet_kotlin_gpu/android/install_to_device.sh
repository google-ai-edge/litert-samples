#!/usr/bin/env bash
# Push the MoViNet-A0 stream tflite into the app's private filesDir.
# Build it with ../conversion (build_movinet.py) or download from Hugging Face
# (litert-community/MoViNet-A0-Stream-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.movinet
DIR="${1:-.}"
M=movinet_a0_stream.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the MoViNet app."
