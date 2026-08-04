#!/usr/bin/env bash
# Push the KittenTTS LiteRT graphs, the voice table, and the G2P graph into the app's private
# filesDir (too big to bundle). Build them with ../../conversion/ (see the sample README), then:
#   ./install_to_device.sh <dir-with-the-files>   (default: current dir)
set -e
PKG=com.google.ai.edge.examples.text_to_speech_streaming
DIR="${1:-.}"
FILES=(
    kitten_predictor_fp16.tflite
    kitten_prosody_fp16.tflite
    kitten_vocoder_fp16.tflite
    voices.bin
    dp_g2p_matcha_fp16.tflite
)
for F in "${FILES[@]}"; do
    echo "pushing $F ..."
    adb push "$DIR/$F" "/data/local/tmp/$F"
    adb shell chmod 644 "/data/local/tmp/$F"
    adb shell run-as $PKG mkdir files 2>/dev/null || true
    adb shell run-as $PKG cp "/data/local/tmp/$F" "files/$F"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the KittenTTS Streaming app."
