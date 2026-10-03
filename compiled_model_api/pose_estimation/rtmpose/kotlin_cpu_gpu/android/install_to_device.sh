#!/usr/bin/env bash
# Push the RTMPose tflite into the app's private filesDir.
#   ./install_to_device.sh <dir-with-rtmpose_s_fp16.tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.pose_estimation.rtmpose
DIR="${1:-.}"
M=rtmpose_s_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the RTMPose app."
