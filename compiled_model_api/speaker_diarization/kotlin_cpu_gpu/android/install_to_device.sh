#!/usr/bin/env bash
# Push the diarization models into the app's private filesDir.
# Build them with ../../conversion/build_diar.py, or get them from Hugging Face
# (litert-community/Speaker-Diarization-LiteRT). Then:
#   ./install_to_device.sh <dir-with-the-models>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.speaker_diarization
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in wespeaker_emb_fp16.tflite pyannote_seg30.onnx; do
  [ -f "$DIR/$M" ] || { echo "skip $M (not in $DIR)"; continue; }
  echo "pushing $M ..."
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell chmod 644 "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Diarization app."
