#!/usr/bin/env bash
# Push the three TIGER-DnR models into the app's private filesDir.
# Build them with ../../conversion/build_tiger.py, or get them from Hugging Face
# (litert-community/TIGER-DnR-LiteRT). Then:
#   ./install_to_device.sh <dir-with-the-tflites>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.audio_source_separation
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in tiger_dialog_fp16.tflite tiger_effect_fp16.tflite tiger_music_fp16.tflite; do
  [ -f "$DIR/$M" ] || { echo "skip $M (not in $DIR)"; continue; }
  echo "pushing $M ..."
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell chmod 644 "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Audio Separation app."
