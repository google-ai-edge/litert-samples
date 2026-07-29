#!/bin/bash
#
# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# build_prebuilt.sh — Download the LiteRT C++ SDK, set up dependencies, and
# build the segmentation binaries for Android arm64-v8a using CMake + NDK.
#
# Usage:
#   bash build_prebuilt.sh \
#       --litert_version=2.1.1 \
#       --ndk_path=/path/to/android-ndk \
#       --litert_so=/path/to/libLiteRt.so \
#       [--gpu_accel_so=/path/to/libLiteRtOpenClAccelerator.so] \
#       [--build_dir=build] \
#       [--sdk_dir=litert_cc_sdk]

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
LITERT_VERSION=""
NDK_PATH="${ANDROID_NDK:-}"
LITERT_SO=""
GPU_ACCEL_SO=""
BUILD_DIR="build"
SDK_DIR="litert_cc_sdk"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-26"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  echo "Usage: $0 \\"
  echo "    --litert_version=<version>   e.g. 2.1.2"
  echo "    --ndk_path=<path>            Path to Android NDK root"
  echo "    --litert_so=<path>           Path to libLiteRt.so from Maven .aar"
  echo "    [--gpu_accel_so=<path>]      Path to libLiteRtClGlAccelerator.so (optional)"
  echo "    [--build_dir=<dir>]          CMake build output dir (default: build)"
  echo "    [--sdk_dir=<dir>]            Where to extract the SDK zip (default: litert_cc_sdk)"
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    --litert_version=*) LITERT_VERSION="${arg#*=}" ;;
    --ndk_path=*)       NDK_PATH="${arg#*=}" ;;
    --litert_so=*)      LITERT_SO="${arg#*=}" ;;
    --gpu_accel_so=*)   GPU_ACCEL_SO="${arg#*=}" ;;
    --build_dir=*)      BUILD_DIR="${arg#*=}" ;;
    --sdk_dir=*)        SDK_DIR="${arg#*=}" ;;
    *) echo "Unknown argument: $arg"; usage ;;
  esac
done

# ── Validate required arguments ───────────────────────────────────────────────
[[ -z "$LITERT_VERSION" ]] && { echo "Error: --litert_version is required."; usage; }
[[ -z "$NDK_PATH" ]]       && { echo "Error: --ndk_path is required (or set ANDROID_NDK)."; usage; }
[[ -z "$LITERT_SO" ]]      && { echo "Error: --litert_so is required."; usage; }

if [[ ! -d "$NDK_PATH" ]]; then
  echo "Error: NDK path does not exist: $NDK_PATH"
  exit 1
fi
if [[ ! -f "$LITERT_SO" ]]; then
  echo "Error: libLiteRt.so not found at: $LITERT_SO"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Step 1: Download and extract the LiteRT C++ SDK ──────────────────────────
SDK_ZIP="litert_cc_sdk_${LITERT_VERSION}.zip"
SDK_URL="https://github.com/google-ai-edge/LiteRT/releases/download/${LITERT_VERSION}/litert_cc_sdk.zip"

if [[ ! -f "${SDK_DIR}/CMakeLists.txt" ]]; then
  echo "==> Downloading LiteRT C++ SDK v${LITERT_VERSION}..."
  mkdir -p "${SDK_DIR}"
  wget -q --show-progress -O "${SDK_DIR}/${SDK_ZIP}" "${SDK_URL}"
  echo "==> Extracting SDK..."
  # The zip contains a single 'litert_cc_sdk/' subfolder; extract its contents
  # directly into SDK_DIR so we get a flat layout (no double nesting).
  unzip -q "${SDK_DIR}/${SDK_ZIP}" -d "${SDK_DIR}/tmp_extract"
  mv "${SDK_DIR}/tmp_extract/litert_cc_sdk/"* "${SDK_DIR}/"
  rm -rf "${SDK_DIR}/tmp_extract" "${SDK_DIR}/${SDK_ZIP}"
  echo "==> SDK extracted to ${SDK_DIR}/"
else
  echo "==> SDK already present at ${SDK_DIR}/ — skipping download."
fi

# ── Step 2: Copy libLiteRt.so into the SDK directory ─────────────────────────
DEST_SO="${SDK_DIR}/libLiteRt.so"
SRC_SO_ABS="$(realpath "${LITERT_SO}" 2>/dev/null || echo "${LITERT_SO}")"
DEST_SO_ABS="$(realpath "${SCRIPT_DIR}/${DEST_SO}" 2>/dev/null || echo "${SCRIPT_DIR}/${DEST_SO}")"
if [[ "${SRC_SO_ABS}" != "${DEST_SO_ABS}" ]]; then
  echo "==> Copying libLiteRt.so into SDK..."
  cp -f "${LITERT_SO}" "${SCRIPT_DIR}/${DEST_SO}"
  echo "    Done: ${DEST_SO}"
else
  echo "==> libLiteRt.so is already in the SDK directory — skipping copy."
fi

# ── Step 3: Download STB headers (header-only) ───────────────────────────────
STB_DIR="${SCRIPT_DIR}/third_party/stb"
STB_BASE_URL="https://raw.githubusercontent.com/nothings/stb/master"

if [[ ! -f "${STB_DIR}/stb_image.h" ]]; then
  echo "==> Downloading STB headers..."
  mkdir -p "${STB_DIR}"
  wget -q -O "${STB_DIR}/stb_image.h"        "${STB_BASE_URL}/stb_image.h"
  wget -q -O "${STB_DIR}/stb_image_write.h"  "${STB_BASE_URL}/stb_image_write.h"
  wget -q -O "${STB_DIR}/stb_image_resize.h" "${STB_BASE_URL}/stb_image_resize.h"
  echo "    STB headers downloaded to ${STB_DIR}/"
else
  echo "==> STB headers already present — skipping download."
fi

# ── Step 4: Configure and build with CMake ───────────────────────────────────
TOOLCHAIN="${NDK_PATH}/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN" ]]; then
  echo "Error: Android CMake toolchain not found at: $TOOLCHAIN"
  exit 1
fi

SDK_ABS="$(cd "${SDK_DIR}" && pwd)"

echo "==> Configuring CMake..."
# Auto-detect GPU accelerator if present in the SDK dir and not explicitly set.
if [[ -z "$GPU_ACCEL_SO" && -f "${SDK_ABS}/libLiteRtOpenClAccelerator.so" ]]; then
  GPU_ACCEL_SO="${SDK_ABS}/libLiteRtOpenClAccelerator.so"
  echo "==> Auto-detected GPU accelerator: ${GPU_ACCEL_SO}"
fi

cmake -S "${SCRIPT_DIR}" \
      -B "${SCRIPT_DIR}/${BUILD_DIR}" \
      -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
      -DANDROID_ABI="${ANDROID_ABI}" \
      -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLITERT_CC_SDK_DIR="${SDK_ABS}"

echo "==> Building..."
cmake --build "${SCRIPT_DIR}/${BUILD_DIR}" --parallel "$(nproc)"

echo ""
echo "Build complete. Binaries are in: ${SCRIPT_DIR}/${BUILD_DIR}/"
ls -lh "${SCRIPT_DIR}/${BUILD_DIR}/cpp_segmentation_cpu" \
       "${SCRIPT_DIR}/${BUILD_DIR}/cpp_segmentation_gpu" \
       "${SCRIPT_DIR}/${BUILD_DIR}/cpp_segmentation_npu" 2>/dev/null || true
