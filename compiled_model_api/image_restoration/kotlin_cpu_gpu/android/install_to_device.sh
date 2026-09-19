#!/usr/bin/env bash
# Push the NAFNet tflite into the app's private filesDir (too big to bundle).
# Build with ../../conversion/build_nafnet.py or get it from Hugging Face
# (litert-community/NAFNet-GoPro-width32-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.image_restoration
DIR="${1:-.}"
M=nafnet_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the NAFNet app."
