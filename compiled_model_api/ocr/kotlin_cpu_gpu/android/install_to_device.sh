#!/usr/bin/env bash
# Push the 2 PP-OCRv5 tflites into the app's private filesDir.
# Build with ../../conversion/build_det.py + build_rec.py or get from Hugging Face
# (litert-community/PP-OCRv5-LiteRT), then:  ./install_to_device.sh <dir>
set -e
PKG=com.google.ai.edge.examples.ocr
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in ppocr_det_fp16.tflite ppocr_rec_fp16.tflite; do
    echo "pushing $M ..."; adb push "$DIR/$M" "/data/local/tmp/$M"; adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"; adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/; echo "done — launch the PP-OCRv5 app."
