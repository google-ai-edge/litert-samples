#!/usr/bin/env bash
# Push the XFeat model into the app's private filesDir.
# Build it with ../../conversion/convert_xfeat.py, or get it from Hugging Face
# (litert-community/xfeat-litert). Then:
#   ./install_to_device.sh <dir-with-xfeat_fp16.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.image_matching
DIR="${1:-.}"; M=xfeat_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Image Matching app."
