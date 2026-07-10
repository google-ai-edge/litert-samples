#!/usr/bin/env bash
# Push the gaze_fp16.tflite into the app's private filesDir.
# Build with ../../conversion (build_gaze.py) or download from Hugging Face
# (litert-community/L2CS-Gaze360-LiteRT), then:
#   ./install_to_device.sh <dir-with-gaze_fp16.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.gaze_estimation
DIR="${1:-.}"
M=gaze_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the gaze estimation app."
