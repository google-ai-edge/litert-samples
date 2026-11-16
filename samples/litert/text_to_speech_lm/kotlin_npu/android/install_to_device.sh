#!/usr/bin/env bash
# Download the Qwen3-TTS LiteRT model files from Hugging Face (too big to
# bundle in the APK) and push them into the app's private filesDir.
#
#   ./install_to_device.sh [download-dir]   (default: ./models)
#   FAST=0 ./install_to_device.sh           the reference graphs instead
#
# Install and launch the app once first so the package directory exists.
set -e
PKG=com.google.ai.edge.examples.text_to_speech_lm
REPO=https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main
DIR="${1:-./models}"
mkdir -p "$DIR"

# The app takes the MTP and codec graphs it finds in filesDir: the folded int8
# MTP graph and the two-part codec when they are there, else the reference
# graphs. The default installs the folded and split graphs; FAST=0 installs the
# reference graphs. Either way the other set is removed from the device, so the
# app runs the set this script installed.
FAST_GRAPHS=(mtp_folded_int8.tflite codec_partA.tflite codec_partB.tflite)
REFERENCE_GRAPHS=(mtp_fp32.tflite codec_decoder_fp32.tflite)
if [ "${FAST:-1}" != "0" ]; then
    GRAPHS=("${FAST_GRAPHS[@]}")
    OTHER_GRAPHS=("${REFERENCE_GRAPHS[@]}")
else
    GRAPHS=("${REFERENCE_GRAPHS[@]}")
    OTHER_GRAPHS=("${FAST_GRAPHS[@]}")
fi

# remote path -> device file name (tables are flattened)
FILES=(
    "talker_int4.tflite talker_int4.tflite"
    "tables/codec_embedding_fp32.npy codec_embedding_fp32.npy"
    "tables/mtp_embeddings_fp16.npy mtp_embeddings_fp16.npy"
    "tables/text_embedding_fp16.npy text_embedding_fp16.npy"
    "tables/text_projection_fp32.npz text_projection_fp32.npz"
    "voices/demo_speaker.npy demo_speaker.npy"
    "vocab.json vocab.json"
    "merges.txt merges.txt"
)
for GRAPH in "${GRAPHS[@]}"; do
    FILES+=("$GRAPH $GRAPH")
done

for PAIR in "${FILES[@]}"; do
    REMOTE="${PAIR%% *}"
    LOCAL="${PAIR##* }"
    if [ ! -f "$DIR/$LOCAL" ]; then
        echo "downloading $REMOTE ..."
        curl -L --fail --retry 3 -C - -o "$DIR/$LOCAL" "$REPO/$REMOTE"
    fi
done

for PAIR in "${FILES[@]}"; do
    LOCAL="${PAIR##* }"
    echo "pushing $LOCAL ..."
    adb push "$DIR/$LOCAL" "/data/local/tmp/$LOCAL"
    adb shell chmod 644 "/data/local/tmp/$LOCAL"
    adb shell run-as $PKG mkdir files 2>/dev/null || true
    adb shell run-as $PKG cp "/data/local/tmp/$LOCAL" "files/$LOCAL"
    adb shell rm "/data/local/tmp/$LOCAL"
done
echo "removing the other graph set, if present: ${OTHER_GRAPHS[*]} ..."
for GRAPH in "${OTHER_GRAPHS[@]}"; do
    adb shell run-as $PKG rm -f "files/$GRAPH"
done
# A running app keeps the graphs it loaded; the next launch reads this set.
adb shell am force-stop $PKG
adb shell run-as $PKG ls -la files/
echo "done — launch the Qwen3 TTS app (graphs: ${GRAPHS[*]})."
