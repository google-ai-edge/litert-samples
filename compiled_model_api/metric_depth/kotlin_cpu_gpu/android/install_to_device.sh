#!/usr/bin/env bash
# Push the Metric3D v2 tflite into the app's private filesDir (too big to bundle).
# Build with ../../conversion/build_m3d.py or get it from Hugging Face
# (mlboydaisuke/Metric3D-v2-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.metric_depth
DIR="${1:-.}"
M=metric3d_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Metric3D v2 app."
