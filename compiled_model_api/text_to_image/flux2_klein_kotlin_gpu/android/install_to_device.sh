#!/usr/bin/env bash
# Stage the FLUX.2-klein LiteRT graphs + precomputed host inputs into the app's external files
# directory. The graphs total ~6.2 GB and are never committed; produce them with the scripts in
# ../conversion, or download them from
# https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT
#
# Usage: ./install_to_device.sh <graphs_dir> <bins_dir>
set -euo pipefail

GRAPHS="${1:?usage: install_to_device.sh <graphs_dir> <bins_dir>}"
BINS="${2:?usage: install_to_device.sh <graphs_dir> <bins_dir>}"
DEST="/sdcard/Android/data/com.google.ai.edge.examples.flux2_klein/files"

GRAPH_FILES=(
  ke_enc0.tflite ke_enc1.tflite ke_enc2.tflite
  kc_prep.tflite kc_double0.tflite kc_double1.tflite
  kc_single0.tflite kc_single1.tflite kc_single2.tflite kc_single3.tflite
  kc_final.tflite kv_vae.tflite
)

for f in "${GRAPH_FILES[@]}"; do
  [ -f "${GRAPHS}/${f}" ] || { echo "missing ${GRAPHS}/${f}"; exit 1; }
done
[ -d "${BINS}" ] || { echo "missing ${BINS}"; exit 1; }

adb shell mkdir -p "${DEST}/klein_bins"
for f in "${GRAPH_FILES[@]}"; do
  echo "pushing ${f} ..."
  adb push "${GRAPHS}/${f}" "${DEST}/${f}"
done
adb push "${BINS}/." "${DEST}/klein_bins/"

# adb-created directories are group-owned by shell; make them traversable by the app.
adb shell chmod 777 "${DEST}/klein_bins"
adb shell 'chmod 666 '"${DEST}"'/klein_bins/*.bin'
echo "done — launch the FLUX.2-klein app and tap Generate."
