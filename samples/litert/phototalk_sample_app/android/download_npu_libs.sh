#!/bin/bash
#
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
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/app/src/main/jniLibs/arm64-v8a"
TMP_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Dynamically parse LiteRT version from libs.versions.toml if not passed as argument
TOML_FILE="${SCRIPT_DIR}/gradle/libs.versions.toml"
PARSED_VERSION=""
if [ -f "$TOML_FILE" ]; then
    PARSED_VERSION=$(grep -E '^\s*litert\s*=' "$TOML_FILE" | head -n 1 | sed -E 's/.*"([^"]+)".*/\1/')
fi

LITERT_VERSION="${1:-${PARSED_VERSION:-2.2.0}}"
QNN_ARCH="${2:-79}"

echo "========================================================="
echo " LiteRT NPU Runtime Libraries Downloader"
echo " LiteRT Version : ${LITERT_VERSION}"
echo " Hexagon Target : v${QNN_ARCH} (SM8750 / Snapdragon 8 Elite)"
echo " Output Target  : ${TARGET_DIR}"
echo "========================================================="

DOWNLOAD_URL="https://github.com/google-ai-edge/LiteRT/releases/download/v${LITERT_VERSION}/litert_npu_runtime_libraries.zip"

echo "Downloading runtime libraries from: ${DOWNLOAD_URL}..."
if ! curl -f -L -o "${TMP_DIR}/litert_npu_runtime_libraries.zip" "${DOWNLOAD_URL}"; then
    echo "Error: Failed to download from ${DOWNLOAD_URL}."
    echo "Please check available releases at: https://github.com/google-ai-edge/LiteRT/releases"
    exit 1
fi

echo "Extracting runtime libraries package..."
unzip -q -o "${TMP_DIR}/litert_npu_runtime_libraries.zip" -d "${TMP_DIR}/npu_libs"

echo "Fetching official Qualcomm QAIRT SDK binaries..."
pushd "${TMP_DIR}/npu_libs" > /dev/null
bash fetch_qualcomm_library.sh
popd > /dev/null

RUNTIME_SRC="${TMP_DIR}/npu_libs/qualcomm_runtime_v${QNN_ARCH}/src/main/jni/arm64-v8a"
if [ ! -d "$RUNTIME_SRC" ]; then
    echo "Warning: Target architecture v${QNN_ARCH} not found in ${TMP_DIR}/npu_libs. Available architectures:"
    ls -d "${TMP_DIR}/npu_libs"/qualcomm_runtime_* || true
    exit 1
fi

echo "Installing Hexagon v${QNN_ARCH} runtime binaries into jniLibs..."
mkdir -p "${TARGET_DIR}"
cp -f "${RUNTIME_SRC}/"* "${TARGET_DIR}/"

echo "Qualcomm NPU runtime libraries successfully installed into ${TARGET_DIR}!"
