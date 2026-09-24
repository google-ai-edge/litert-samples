#!/usr/bin/env bash
# Push the CREPE tflite (~44.5 MB) into the app's private filesDir.
# Build it with ../../conversion/build_crepe.py, or get it from Hugging Face
# (litert-community/CREPE-pitch-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.pitch_detection
DIR="${1:-.}"
M=crepe_full_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Pitch Detection app."
