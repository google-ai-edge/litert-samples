#!/usr/bin/env bash
# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Builds the model files the demo loads (not checked in: 166-224 MB each):
#   $ARTIFACTS/sam2_tiny_{1024,512,384}_video.safetensors   weights (HF export)
#   app/public/models/sam2_chain_{384,512[,1024]}.tflite     SAM 2 model, authored
#                                                            with the Tensor API
# The weight files are also linked into app/public/models for ?build=browser.
#
#   tools/build_models.sh [sizes...]      default: 384 512
# Needs: the litert-samples Bazel workspace (this repo); a Python env ($PYTHON) with torch,
# transformers, safetensors, numpy; Bazel 7.
set -euo pipefail
if [ $# -gt 0 ]; then SIZES=("$@"); else SIZES=(384 512); fi
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$HERE")"
SAMPLES="${LITERT_SAMPLES:-$(cd "$PROJ/../../../.." && pwd)}"  # the litert-samples repo this sample lives in
ARTROOT="${ARTIFACTS:-$PROJ/artifacts}"
PY="${PYTHON:-$PROJ/.venv/bin/python}"
MODELS="$PROJ/app/public/models"
CONSTS="$MODELS/sam2_host_consts.safetensors"
mkdir -p "$ARTROOT" "$MODELS"

# 1. 1024 weights from facebook/sam2.1-hiera-tiny (the sample's exporter reads
#    rotary buffers that transformers 5.0.0 exposes; later 5.x compute them lazily).
BASE="$ARTROOT/sam2_tiny_1024_video.safetensors"
if [ ! -f "$BASE" ]; then
  "$PY" -m pip install -q "transformers==5.0.0"
  (cd "$SAMPLES/models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/verify" &&
   "$PY" export_weights_1024.py --out "$BASE")
  "$PY" -m pip install -q -U "transformers>=5"
fi

# 2. The native builder.
(cd "$SAMPLES" && bazel build --noenable_platform_specific_config --copt=-w //samples/web_demos/src/sam2/cc:sam2_chain_main)
BIN="$SAMPLES/bazel-bin/samples/web_demos/src/sam2/cc/sam2_chain_main"

for S in "${SIZES[@]}"; do
  W="$ARTROOT/sam2_tiny_${S}_video.safetensors"
  # 3. Smaller inputs: same learned weights, resolution tables re-derived by HF.
  [ "$S" = 1024 ] || [ -f "$W" ] || "$PY" "$HERE/export_weights.py" --size "$S" --base "$BASE" --out "$W"
  # 4. Author + serialize the SAM 2 model with the Tensor API.
  "$BIN" --weights="$W" --image_size="$S" --host_consts="$CONSTS" \
    --sam2_tflite="$MODELS/sam2_chain_$S.tflite" --build_only
  ln -f "$W" "$MODELS/" 2>/dev/null || cp -f "$W" "$MODELS/"
done
ls -la "$MODELS"
