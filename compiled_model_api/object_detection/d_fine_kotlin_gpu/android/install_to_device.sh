#!/usr/bin/env bash
# Push the 2 D-FINE-S tflites into the app's private filesDir (too big to bundle).
# Build with ../conversion (build_dfine_split.py + build_dfine_fix3.py + pack_assets.py) or get them
# from Hugging Face (litert-community/D-FINE-S-LiteRT), then:
#   ./install_to_device.sh <dir-with-the-tflites>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.d_fine
DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in dfine_graphA_fp16.tflite dfine_graphB_fp16.tflite; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
    adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the D-FINE app."
