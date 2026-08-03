#!/bin/bash
# Fetches the app's bundled resources (tokenizer tables + pipeline meta) from
# the model repo on Hugging Face. Resources/ is not committed — run this
# before xcodegen. Pass a local model download dir as $1 to copy from it
# instead of downloading.
set -euo pipefail
DST="$(cd "$(dirname "$0")" && pwd)/Resources"
mkdir -p "$DST"
if [ -n "${1:-}" ]; then
  cp "$1/tokenizer/vocab.json" "$1/tokenizer/merges.txt" "$1/pipeline_meta.json" "$DST/"
else
  BASE="https://huggingface.co/litert-community/Bonsai-Image-ternary-4B/resolve/main"
  curl -sfL "$BASE/tokenizer/vocab.json" -o "$DST/vocab.json"
  curl -sfL "$BASE/tokenizer/merges.txt" -o "$DST/merges.txt"
  curl -sfL "$BASE/pipeline_meta.json" -o "$DST/pipeline_meta.json"
fi
ls -la "$DST"
