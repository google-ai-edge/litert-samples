#!/usr/bin/env bash
# Stage the FLUX.2-klein int8 LiteRT graphs and the host .bin tensors to the device.
#
# Usage: ./install_to_device.sh <graphs_dir> <bins_dir> [edit_bins_dir]
#   graphs_dir     holds the .tflite chunks (conversion/chunked_export_klein.py,
#                  conversion/build_klein_enc.py, conversion/vae_deploy_klein.py, and
#                  for editing conversion/chunked_export_klein.py --edit +
#                  conversion/vae_encode_klein.py)
#   bins_dir       klein_bins/ written by conversion/gen_prep_klein.py
#   edit_bins_dir  optional; the bins written by gen_prep_klein.py --edit.
#                  Supplying it also stages the nine editing graphs.
#
# Text-to-image needs 6.2 GB of graphs; adding editing brings it to 10.1 GB. They
# are never committed. The app reads them from its external files dir with the
# file-path CompiledModel.create overload.
set -euo pipefail

GRAPHS="${1:?usage: install_to_device.sh <graphs_dir> <bins_dir> [edit_bins_dir]}"
BINS="${2:?usage: install_to_device.sh <graphs_dir> <bins_dir> [edit_bins_dir]}"
EDIT_BINS="${3:-}"
PKG="com.google.ai.edge.examples.flux2_klein"
DEST="/sdcard/Android/data/${PKG}/files"

GRAPH_FILES=(
  ke_enc0.tflite ke_enc1.tflite ke_enc2.tflite
  kc_prep.tflite kc_double0.tflite kc_double1.tflite
  kc_single0.tflite kc_single1.tflite kc_single2.tflite kc_single3.tflite
  kc_final.tflite kv_vae.tflite
)

# The editing graphs are the same weights re-exported at the longer image
# sequence (256 noise + 256 reference tokens), plus the VAE encoder.
EDIT_GRAPH_FILES=(
  kce_prep.tflite kce_double0.tflite kce_double1.tflite
  kce_single0.tflite kce_single1.tflite kce_single2.tflite kce_single3.tflite
  kce_final.tflite kv_vae_enc.tflite
)

if [ -n "${EDIT_BINS}" ]; then
  GRAPH_FILES+=("${EDIT_GRAPH_FILES[@]}")
fi

for f in "${GRAPH_FILES[@]}"; do
  [ -f "${GRAPHS}/${f}" ] || { echo "missing ${GRAPHS}/${f}"; exit 1; }
done
[ -d "${BINS}" ] || { echo "missing ${BINS}"; exit 1; }

adb shell mkdir -p "${DEST}/klein_bins"
for f in "${GRAPH_FILES[@]}"; do
  echo "pushing ${f} ($(du -h "${GRAPHS}/${f}" | cut -f1)) ..."
  adb push "${GRAPHS}/${f}" "${DEST}/${f}"
done
adb push "${BINS}/." "${DEST}/klein_bins/"

if [ -n "${EDIT_BINS}" ]; then
  [ -d "${EDIT_BINS}" ] || { echo "missing ${EDIT_BINS}"; exit 1; }
  adb shell mkdir -p "${DEST}/klein_bins_edit"
  adb push "${EDIT_BINS}/." "${DEST}/klein_bins_edit/"
  adb shell chmod 777 "${DEST}/klein_bins_edit"
  adb shell 'chmod 666 '"${DEST}"'/klein_bins_edit/*.bin'
fi

# adb-created dirs are group-owned by shell; make them app-traversable.
adb shell chmod 777 "${DEST}/klein_bins"
adb shell 'chmod 666 '"${DEST}"'/klein_bins/*.bin'
echo "done -> ${DEST}"
