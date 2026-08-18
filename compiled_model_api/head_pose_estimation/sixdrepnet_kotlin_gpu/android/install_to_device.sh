#!/usr/bin/env bash
# Push the 6drepnet.tflite into the app's private filesDir.
# Build with ../conversion (build_ormbg.py) or download from Hugging Face
# (litert-community/6DRepNet-HeadPose-LiteRT), then:
#   ./install_to_device.sh <dir-with-6drepnet.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.sixdrepnet
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing 6drepnet.tflite ..."
adb push "$DIR/6drepnet.tflite" "/data/local/tmp/6drepnet.tflite"
adb shell chmod 644 "/data/local/tmp/6drepnet.tflite"
adb shell run-as $PKG cp "/data/local/tmp/6drepnet.tflite" "files/6drepnet.tflite"
adb shell rm "/data/local/tmp/6drepnet.tflite"
adb shell run-as $PKG ls -la files/
echo "done — launch the ormbg app."
