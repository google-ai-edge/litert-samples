#!/usr/bin/env bash
# Push the PANNs CNN14 tflite (~162 MB) into the app's private filesDir (too big to bundle).
# Build it with ../../conversion/build_panns.py, or get it from Hugging Face
# (litert-community/PANNs-CNN14-AudioSet-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.sound_event_detection
DIR="${1:-.}"
M=cnn14_audioset_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Sound Event Detection app."
