#!/bin/bash
# Copies the app's bundled resources (tokenizer tables + pipeline meta) out of
# the HF staging dir. Resources/ is not committed — run this before xcodegen.
set -euo pipefail
SRC="${1:-$HOME/models/bonsai-image-4b-tflite/hf_upload}"
DST="$(dirname "$0")/Resources"
mkdir -p "$DST"
cp "$SRC/tokenizer/vocab.json" "$SRC/tokenizer/merges.txt" "$SRC/pipeline_meta.json" "$DST/"
ls -la "$DST"
