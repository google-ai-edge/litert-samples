#!/bin/bash
# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

for tool in curl shasum gzip cmp; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done
SAMPLE="$(cd "$(dirname "$0")" && pwd)"
MODELS="$SAMPLE/Models"
REVISION="143eb4236e8c81ea6849ede31f7f088ea2393cfa"
BASE="https://huggingface.co/litert-community/Matcha-TTS/resolve/$REVISION"
mkdir -p "$MODELS"

while read -r expected name; do
  target="$MODELS/$name"
  actual=""
  if [ -f "$target" ]; then
    actual="$(shasum -a 256 "$target")"
    actual="${actual%% *}"
  fi
  if [ "$actual" = "$expected" ]; then
    echo "Verified existing $name"
    continue
  fi
  echo "Fetching $name at $REVISION"
  curl --fail --location --retry 3 --connect-timeout 30 \
    --output "$target.partial" "$BASE/$name"
  actual="$(shasum -a 256 "$target.partial")"
  actual="${actual%% *}"
  if [ "$actual" != "$expected" ]; then
    echo "Checksum mismatch for $name: expected $expected, received $actual" >&2
    exit 1
  fi
  mv "$target.partial" "$target"
done <<'FILES'
6e4b481f6874dfabc32ce73bf6f0ea1ba6ab5986ee6f76a27779364be8a53c73 dp_g2p_matcha_fp16.tflite
2c2e70235f0daaa9d82366416aad69b4f6e916f080a2da8b4b3b759f7cfd3069 matcha_decoder_fp16.tflite
adf2642699b09a9f0cc302a64396554ac698401d93cffad4462792d6df8589a5 matcha_textenc_fp16.tflite
7d0b4d975eb1558d02c0fa781b329f5fc650cfee6bffcf311de4b190d62020ac matcha_vocoder_fp16.tflite
18be34cda4afa1f59b9fb0b80d03f048bd9dd650e29846e4c6467e0b614c1ee7 emb.bin
5b3493a8cd4d20b72c7b91415afaf3f32335ebd81f349698e1cedc898c59f979 g2p_dict.txt.gz
7363e9e4dda1613aebff7005f2e8c0c76d9b0a1cac31de0f3483bef3089c6906 config.json
7b87bfeaaa072be236e8491d771b0cb97cc92c3e5d83e3558fff8849868810f5 g2p_meta.json
FILES

gzip -dc "$MODELS/g2p_dict.txt.gz" > "$MODELS/g2p_dict.txt.partial"
if [ -f "$MODELS/g2p_dict.txt" ] && cmp -s "$MODELS/g2p_dict.txt.partial" "$MODELS/g2p_dict.txt"; then
  rm "$MODELS/g2p_dict.txt.partial"
else
  mv "$MODELS/g2p_dict.txt.partial" "$MODELS/g2p_dict.txt"
fi
echo "Model assets ready in Models/; all eight published files passed SHA-256 verification."
