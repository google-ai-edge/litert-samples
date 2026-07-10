#!/usr/bin/env bash
# Push the RAM++ graphs into the app's private filesDir.
# Build them with the scripts in ../../conversion/ (see README.md), or download from Hugging Face
# (litert-community/RAM-Plus-LiteRT). Then:
#   ./install_to_device.sh <dir-with-artifacts>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.image_tagging
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in ram_swin_s012_fp16.tflite ram_stage3_tail_fp16.tflite ram_reweight_fp16.tflite ram_taghead_fp16.tflite; do
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell chmod 644 "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Image Tagging app."
