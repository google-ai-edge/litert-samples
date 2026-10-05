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

# One-command verification of the three layers, fast enough to run often:
#
#   A. Tensor API pipeline, native C++   host-plan unit tests; the ModelChain
#      pipeline on CPU fp32 and Metal fp16
#   B. wasm target                        incremental emscripten build; the same
#      pipeline in Chrome on WebGPU, with the prebuilt model and with the SAM 2
#      model authored in the page by the wasm Tensor API
#   C. web demo                           the app's UI flow on WebGPU (clicks,
#      192-frame tracking, effects, layout, camera smooth / aligned), and
#      WebGPU health (hardware adapter, every signature on WebGPU, no WebGPU
#      errors, bounded GPU memory, steady per-frame time)
#
# Every pipeline run (A2, A3, B2, B3) uses 10 frames of the football sample and
# 6 objects, one per slot (2 clicks; 1 click; a box + a negative click joining
# at frame 6; a box; a box joining at frame 3; a box on the ball joining at frame 2)
# and is checked by verify_chain.py: preprocess graph vs numpy, every mask vs
# HF Sam2VideoModel, composite graph vs numpy. The HF reference is computed
# once per (size, memory, clip, prompts) from the numpy preprocess reference
# and cached, so all runs compare against the same ground truth.
#
#   tools/verify_all.sh [384|512]        (~2-3 min at 384 once the cache exists)
set -uo pipefail
S="${1:-384}"
T=10
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$HERE")"
# External inputs (override with env vars; defaults: the repo root and ./artifacts, ./.venv).
SAMPLES="${LITERT_SAMPLES:-$(cd "$PROJ/../../../.." && pwd)}"  # the litert-samples repo this sample lives in
ARTROOT="${ARTIFACTS:-$PROJ/artifacts}"            # weights + run outputs
ART="$ARTROOT/chain"
PY="${PYTHON:-$PROJ/.venv/bin/python}"
CACHE="$ART/ref_cache"
CLIP="$ART/football_640x360_24.rgba"
PROMPTS='0@0:0.44,0.28,1;0.46,0.40,1|1@0:0.484,0.79,1|2@6:0.14,0.32,2;0.215,0.645,3;0.15,0.34,0|3@0:0.194,0.342,2;0.253,0.632,3|4@3:0.594,0.352,2;0.658,0.632,3|5@2:0.45,0.71,2;0.53,0.865,3'
CONSTS="$PROJ/app/public/models/sam2_host_consts.safetensors"
BIN="$SAMPLES/bazel-bin/models/sam2/sam2_hiera_tiny_video/web_demo/cc/sam2_chain_main"
BZ=(--noenable_platform_specific_config --copt=-w)
mkdir -p "$ART" "$CACHE"
[ -f "$CLIP" ] || ffmpeg -loglevel error -y -i "$PROJ/app/public/assets/football_ai_studio.mp4" \
  -frames:v 24 -vf scale=640:360 -f rawvideo -pix_fmt rgba "$CLIP"

SUMMARY=()
FAILED=0
step() {  # name, command...
  local name=$1; shift
  local t0=$SECONDS
  echo "== $name"
  if "$@"; then SUMMARY+=("PASS  $(printf '%4ss' $((SECONDS - t0)))  $name")
  else SUMMARY+=("FAIL  $(printf '%4ss' $((SECONDS - t0)))  $name"); FAILED=1; fi
}
filter() { grep -E '^ +(ok|FAIL|info|GPU|\()|^VERIFY|median|loaded|CHECK|PASSED|FAILED|error' ; }

native_run() {  # tag, min_iou, min_mean_iou, flags...
  local tag=$1 min=$2 mean=$3; shift 3
  (cd "$ART" && rm -rf "$tag" && mkdir -p "$tag" && cp -f \
    "$SAMPLES/bazel-bin/models/sam2/sam2_hiera_tiny_video/web_demo/cc/sam2_chain_main.runfiles/litert_prebuilts/macos_arm64/libLiteRtMetalAccelerator.dylib" . &&
   "$BIN" --image_size=$S --host_consts="$CONSTS" --sam2_tflite="sam2_chain_$S.tflite" --reuse_tflite \
     --frames_rgba="$CLIP" --width=640 --height=360 --frames=$T --prompts="$PROMPTS" --dump_dir="$tag" \
     --dump_rgb=0,6,9 "$@" 2>&1 | grep -E '^median|^error') || return 1
  "$PY" "$HERE/verify_chain.py" --dump "$ART/$tag" --rgba "$CLIP" --ref_cache "$CACHE" \
    --min_iou $min --min_mean_iou $mean --tag "$tag" 2>&1 | filter
  local rc=${PIPESTATUS[0]}
  rm -f "$ART/$tag"/pixels_f*.f32
  return $rc
}
browser_run() {  # extra args
  (cd "$PROJ/app" && node test/e2e/chain_e2e.mjs --size=$S --frames=$T --ref_cache="$CACHE" "$@" 2>&1 | filter;
   exit ${PIPESTATUS[0]})
}

# ---- A. Tensor API pipeline, native C++
a1() {
  (cd "$SAMPLES" && bazel test "${BZ[@]}" //models/sam2/sam2_hiera_tiny_video/web_demo/cc:host_plan_test --test_output=errors 2>&1 | filter &&
   bazel build "${BZ[@]}" //models/sam2/sam2_hiera_tiny_video/web_demo/cc:sam2_chain_main 2>&1 | grep -E 'ERROR' ; exit ${PIPESTATUS[0]}) || return 1
  # Always re-author (a few seconds): a model left from older graph code would
  # otherwise be verified instead of the current one.
  "$BIN" --weights="$ARTROOT/sam2_tiny_${S}_video.safetensors" \
    --image_size=$S --host_consts="$CONSTS" --sam2_tflite="$ART/sam2_chain_$S.tflite" --build_only
}
step "A1 C++ unit tests + native build" a1
step "A2 native ModelChain, CPU fp32, 7-frame memory, overlay" native_run "v_cpu${S}_nmm7" 0.95 0.99 --nmm=7 --effect=overlay
step "A3 native ModelChain, Metal fp16, 2-frame memory, cutout" native_run "v_metal${S}_nmm2" 0.85 0.98 --nmm=2 --effect=cutout --accelerator=gpu

# ---- B. wasm target
b1() {
  local out rc
  out="$("$PROJ/wasm/build.sh" 2>&1)"; rc=$?
  # Compiler / ninja / make failures only: a bare 'error' also matches build
  # lines such as 'Linking ... libabsl_strerror.a'.
  if [ $rc -ne 0 ] || grep -qE 'error:|FAILED:|Error [0-9]+' <<<"$out"; then
    local msg; msg="$(grep -E 'error:|FAILED:|Error [0-9]+|not found|No such file' <<<"$out" | head -20)"
    echo "${msg:-$(tail -20 <<<"$out")}"
    echo "  wasm/build.sh failed (exit $rc); B2-B4 use the checked-in app/public/wasm"
    return 1
  fi
  ls -la "$PROJ/app/public/wasm" | tail -2
}
step "B1 wasm build (emscripten, JSPI; same LiteRT / Tensor API sources)" b1
step "B2 wasm pipeline on WebGPU fp16, 7-frame memory, overlay" browser_run --nmm=7 --effect=overlay --tag="v_wasm${S}_nmm7"
step "B3 wasm, SAM 2 model authored in the page, 2-frame, cutout" browser_run --nmm=2 --effect=cutout --build=browser --tag="v_wasm${S}_nmm2_browserbuild"
step "B4 wasm pipeline on WebGPU fp32 (exactness), 7-frame" browser_run --nmm=7 --precision=fp32 --tag="v_wasm${S}_fp32_nmm7"

# ---- C. web demo
c1() { (cd "$PROJ/app" && node test/e2e/ui_check.mjs 2>&1 | filter; exit ${PIPESTATUS[0]}); }
step "C1 web demo UI on WebGPU (clicks, tracking, effects, layout, camera)" c1
c2() { (cd "$PROJ/app" && node test/e2e/webgpu_check.mjs 2>&1 | filter; exit ${PIPESTATUS[0]}); }
step "C2 WebGPU health (adapter, full acceleration, errors, GPU memory, sustained speed)" c2

echo
echo "================ verify_all ($S px) ================"
printf '%s\n' "${SUMMARY[@]}"
exit $FAILED
