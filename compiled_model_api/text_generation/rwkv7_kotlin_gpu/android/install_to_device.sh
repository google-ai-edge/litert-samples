#!/usr/bin/env bash
# Push the RWKV-7 step graph and embedding table into the app's private
# filesDir (too big to bundle in the APK). Get the files from Hugging Face
# (litert-community/RWKV-7-World-0.1B-LiteRT) or build them with ../conversion/,
# then:
#   ./install_to_device.sh <dir-with-the-files>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.rwkv7
DIR="${1:-.}"
FILES=(
    rwkv7_step_fp16.tflite
    rwkv7_emb_fp16.bin
)
for F in "${FILES[@]}"; do
    echo "pushing $F ..."
    adb push "$DIR/$F" "/data/local/tmp/$F"
    adb shell chmod 644 "/data/local/tmp/$F"
    adb shell run-as $PKG mkdir files 2>/dev/null || true
    adb shell run-as $PKG cp "/data/local/tmp/$F" "files/$F"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the RWKV-7 Text Generation app."
