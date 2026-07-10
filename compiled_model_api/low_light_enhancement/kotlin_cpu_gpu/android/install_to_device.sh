#!/usr/bin/env bash
# Push the CPGA-Net tflite into the app's private filesDir (loaded from filesDir).
# Build with ../../conversion/build_cpga.py or get it from Hugging Face
# (litert-community/CPGA-Net-LowLight-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.low_light_enhancement
DIR="${1:-.}"
M=cpga_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Low-Light Enhancement app."
