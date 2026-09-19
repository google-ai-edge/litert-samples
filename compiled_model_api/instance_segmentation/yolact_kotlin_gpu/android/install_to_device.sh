#!/usr/bin/env bash
# Push the YOLACT tflite + priors into the app's private filesDir.
# Build with ../conversion (build_yolact.py) or download from Hugging Face
# (litert-community/YOLACT-ResNet50-LiteRT), then:
#   ./install_to_device.sh <dir-with-yolact.tflite-and-priors.bin>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.yolact
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in yolact.tflite priors.bin; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
    adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the YOLACT app."
