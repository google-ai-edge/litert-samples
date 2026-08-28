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

echo "Downloading LiteRT 2.2.0 NPU Runtime Libraries..."
curl -L -o "${TMP_DIR}/litert_npu_runtime_libraries.zip" "https://github.com/google-ai-edge/LiteRT/releases/download/v2.2.0/litert_npu_runtime_libraries.zip"

echo "Extracting runtime libraries package..."
unzip -q -o "${TMP_DIR}/litert_npu_runtime_libraries.zip" -d "${TMP_DIR}/npu_libs"

echo "Fetching official Qualcomm QAIRT SDK binaries..."
pushd "${TMP_DIR}/npu_libs" > /dev/null
bash fetch_qualcomm_library.sh
popd > /dev/null

echo "Copying Hexagon v79 (Snapdragon 8 Elite / SM8750) runtime binaries..."
mkdir -p "${TARGET_DIR}"
cp -f "${TMP_DIR}/npu_libs/qualcomm_runtime_v79/src/main/jni/arm64-v8a/"* "${TARGET_DIR}/"

echo "Qualcomm NPU runtime libraries successfully installed into ${TARGET_DIR}!"
