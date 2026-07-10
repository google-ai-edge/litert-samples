#!/usr/bin/env bash
# Push the dehazeformer_base.tflite into the app's private filesDir.
# Build with ../conversion (build_dehaze.py) or download from Hugging Face
# (litert-community/DehazeFormer-MCT-LiteRT), then:
#   ./install_to_device.sh <dir-with-dehazeformer_base.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.dehaze
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing dehazeformer_base.tflite ..."
adb push "$DIR/dehazeformer_base.tflite" "/data/local/tmp/dehazeformer_base.tflite"
adb shell chmod 644 "/data/local/tmp/dehazeformer_base.tflite"
adb shell run-as $PKG cp "/data/local/tmp/dehazeformer_base.tflite" "files/dehazeformer_base.tflite"
adb shell rm "/data/local/tmp/dehazeformer_base.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the dehazing app."
