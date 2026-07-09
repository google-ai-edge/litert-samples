#!/usr/bin/env bash
# Push the Dia2-1B LiteRT graphs + baked tables to the app's EXTERNAL files dir (too big to bundle).
# adb writes there directly (no /data/local/tmp double-copy), so peak storage = the file set itself
# (~4.1 GB) — important on near-full devices. Get the files from Hugging Face
# (https://huggingface.co/mlboydaisuke/Dia2-1B-LiteRT) or build them with the scripts in
# ../../conversion, then:
#   ./install_to_device.sh <dir-with-the-files>   (default: current dir)
#
# The first app launch is expected to fail with "Load failed" until this has been run.
set -e
PKG=com.google.ai.edge.examples.text_to_speech_dia2
DIR="${1:-.}"
DEST="/sdcard/Android/data/$PKG/files"

# Everything runs on CPU as fp32: the GPU delegate rejects the language models' KV-step
# FULLY_CONNECTED weight shapes, and fp16 collapses these deep stacks on ARM.
FILES=(
    dia2_temporal_fp32.tflite
    dia2_depformer_wi0_fp32.tflite
    dia2_depformer_wi1_fp32.tflite
    dia2_depformer_wi2_fp32.tflite
    dia2_mimi_dequant.tflite
    dia2_mimi_decode_t256.tflite
    dia2_combined_main.f16
    dia2_combined_second.f16
    dia2_temporal_audio.f16
    dia2_dep_audio.f16
    dia2_dep_in.f16
    dia2_dep_logits.f16
    dia2_constants.json
)

adb shell mkdir -p "$DEST"
for F in "${FILES[@]}"; do
    if [ ! -f "$DIR/$F" ]; then
        echo "missing: $DIR/$F" >&2
        exit 1
    fi
    echo "pushing $F ..."
    adb push "$DIR/$F" "$DEST/$F" > /dev/null
done

adb shell ls -la "$DEST"
echo "done — launch the Dia2 TTS app."
