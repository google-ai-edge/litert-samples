#!/usr/bin/env bash
# Push the CMGAN model into the app's private filesDir.
# Build it with ../../conversion/build_cmgan.py, or get it from Hugging Face
# (litert-community/CMGAN-LiteRT). Then:
#   ./install_to_device.sh <dir-with-cmgan_fp16.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.speech_enhancement
DIR="${1:-.}"; M=cmgan_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Speech Enhancement app."
