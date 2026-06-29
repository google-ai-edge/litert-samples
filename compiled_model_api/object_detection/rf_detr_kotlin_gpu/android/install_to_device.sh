#!/usr/bin/env bash
# Push the 2 RF-DETR Nano tflites into the app's private filesDir (too big to bundle).
# Build with ../conversion/build_rfdetr_split.py or get them from Hugging Face
# (litert-community/RF-DETR-Nano-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflites>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.rf_detr
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in rfdetr_graphA_fp16.tflite rfdetr_graphB_fp16.tflite; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
    adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the RF-DETR app."
