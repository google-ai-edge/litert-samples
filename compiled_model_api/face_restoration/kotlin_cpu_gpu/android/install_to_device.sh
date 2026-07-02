#!/usr/bin/env bash
# Push the models into the app's private filesDir:
#   gfpgan_fp16.tflite (431 MB, too large to bundle) + yunet_fp16.tflite (0.3 MB, face detection).
# Build them with ../../conversion/build_gfpgan.py, or get gfpgan_fp16.tflite from Hugging Face
# (litert-community/GFPGAN-v1.4-LiteRT). Then:
#   ./install_to_device.sh <dir-with-the-tflites>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.face_restoration
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in gfpgan_fp16.tflite yunet_fp16.tflite; do
  [ -f "$DIR/$M" ] || { echo "skip $M (not in $DIR)"; continue; }
  echo "pushing $M ..."
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell chmod 644 "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Face Restoration app."
