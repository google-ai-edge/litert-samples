#!/bin/bash
# Copies the app's bundled assets (tokenizer tables + pipeline meta) out of
# the HF staging dir. assets/ is not committed — run this before building.
set -euo pipefail
SRC="${1:-$HOME/models/bonsai-image-4b-tflite/hf_upload}"
DST="$(dirname "$0")/app/src/main/assets"
mkdir -p "$DST"
cp "$SRC/tokenizer/vocab.json" "$SRC/tokenizer/merges.txt" "$SRC/pipeline_meta.json" "$DST/"
ls -la "$DST"
