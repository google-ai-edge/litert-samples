#!/usr/bin/env bash
# Push the clothseg.tflite into the app's private filesDir.
# Build with ../conversion (build_ormbg.py) or download from Hugging Face
# (litert-community/Cloth-Segmentation-U2Net-LiteRT), then:
#   ./install_to_device.sh <dir-with-clothseg.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.clothseg
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing clothseg.tflite ..."
adb push "$DIR/clothseg.tflite" "/data/local/tmp/clothseg.tflite"
adb shell chmod 644 "/data/local/tmp/clothseg.tflite"
adb shell run-as $PKG cp "/data/local/tmp/clothseg.tflite" "files/clothseg.tflite"
adb shell rm "/data/local/tmp/clothseg.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the ormbg app."
