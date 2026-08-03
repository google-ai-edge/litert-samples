#!/bin/bash
# Fetches everything the build needs that is not committed here:
#   Resources/           tokenizer tables + pipeline meta (from the model repo
#                        on Hugging Face; pass a local model download dir as $1
#                        to copy from it instead)
#   Frameworks/          the LiteRT runtime dylib pair
#   third_party/LiteRT/  the LiteRT C API headers
# Run this before xcodegen + build. No LiteRT code is committed in this
# sample — headers are downloaded at the v2.1.6 release tag, the SAME release
# as the runtime binaries.
#
# Runtime pair: ai-edge-litert 2.1.6 (libLiteRt.dylib +
# libLiteRtMetalAccelerator.dylib from the SAME wheel). Keep the pair
# same-generation: mixing generations makes the accelerator reject the
# serialized options. 2.1.6 is the release this app was verified with.
set -euo pipefail
cd "$(dirname "$0")"
RUNTIME="${RUNTIME:-.litert-runtime}"
LITERT_TAG="${LITERT_TAG:-v2.1.6}"

mkdir -p Resources Frameworks

# --- tokenizer tables + pipeline meta --------------------------------------
if [ -n "${1:-}" ]; then
  cp "$1/tokenizer/vocab.json" "$1/tokenizer/merges.txt" "$1/pipeline_meta.json" Resources/
else
  BASE="https://huggingface.co/litert-community/Bonsai-Image-ternary-4B/resolve/main"
  curl -sfL "$BASE/tokenizer/vocab.json" -o Resources/vocab.json
  curl -sfL "$BASE/tokenizer/merges.txt" -o Resources/merges.txt
  curl -sfL "$BASE/pipeline_meta.json" -o Resources/pipeline_meta.json
fi

# --- LiteRT C API headers, pinned to the runtime's release tag -------------
HDR_BASE="https://raw.githubusercontent.com/google-ai-edge/LiteRT/$LITERT_TAG"
HDRS=(
  litert/c/internal/litert_scheduling_info.h
  litert/c/litert_any.h
  litert/c/litert_common.h
  litert/c/litert_compiled_model.h
  litert/c/litert_custom_op_kernel.h
  litert/c/litert_custom_tensor_buffer.h
  litert/c/litert_environment.h
  litert/c/litert_environment_options.h
  litert/c/litert_gl_types.h
  litert/c/litert_layout.h
  litert/c/litert_model.h
  litert/c/litert_model_types.h
  litert/c/litert_op_code.h
  litert/c/litert_opaque_options.h
  litert/c/litert_opencl_types.h
  litert/c/litert_options.h
  litert/c/litert_tensor_buffer.h
  litert/c/litert_tensor_buffer_requirements.h
  litert/c/litert_tensor_buffer_types.h
  litert/c/litert_webgpu_types.h
)
if [ ! -f "third_party/LiteRT/.tag-$LITERT_TAG" ]; then
  echo "Fetching LiteRT C headers at $LITERT_TAG"
  rm -rf third_party/LiteRT
  for h in "${HDRS[@]}"; do
    mkdir -p "third_party/LiteRT/$(dirname "$h")"
    curl -sfL "$HDR_BASE/$h" -o "third_party/LiteRT/$h" &
  done
  wait
  for h in "${HDRS[@]}"; do
    [ -s "third_party/LiteRT/$h" ] || { echo "FAILED to fetch $h"; exit 1; }
  done
  # build_config.h is bazel-generated; the generated file is an empty guard.
  mkdir -p third_party/LiteRT/litert/build_common
  printf '#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_\n#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_\n#endif\n' \
    > third_party/LiteRT/litert/build_common/build_config.h
  touch "third_party/LiteRT/.tag-$LITERT_TAG"
fi

# --- runtime dylib pair from the ai-edge-litert 2.1.6 wheel ----------------
if [ ! -f "$RUNTIME/libLiteRt.dylib" ]; then
  echo "Runtime pair not found at $RUNTIME — extracting from the ai-edge-litert wheel"
  TMP=$(mktemp -d)
  python3 -m pip download ai-edge-litert==2.1.6 --no-deps --only-binary :all: \
    --platform macosx_12_0_arm64 --python-version 310 --implementation cp -d "$TMP"
  unzip -o -q "$TMP"/ai_edge_litert-*.whl -d "$TMP/wheel" \
    "ai_edge_litert/libLiteRt.dylib" "ai_edge_litert/libLiteRtMetalAccelerator.dylib"
  mkdir -p "$RUNTIME"
  cp "$TMP/wheel/ai_edge_litert/"*.dylib "$RUNTIME/"
  rm -rf "$TMP"
fi
cp "$RUNTIME/libLiteRt.dylib" "$RUNTIME/libLiteRtMetalAccelerator.dylib" Frameworks/
ls -la Resources Frameworks
