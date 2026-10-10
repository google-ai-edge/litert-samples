#!/usr/bin/env bash
# Push the CLIPSeg graphs + host assets into the app's private filesDir.
# Build with ../../conversion/build_clipseg.py, or get them from Hugging Face
# (litert-community/CLIPSeg-rd64-LiteRT). Then:
#   ./install_to_device.sh <dir-with-artifacts>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.text_prompted_segmentation
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in clipseg_vision_fp16.tflite clipseg_text_fp16.tflite clipseg_decoder.tflite token_embedding_f16.bin text_projection_f16.bin vocab.json merges.txt; do
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell chmod 644 "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Text Prompted Segmentation app."
