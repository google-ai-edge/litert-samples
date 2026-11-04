#!/usr/bin/env bash
# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

set -euo pipefail

cd "$(dirname "$0")"

QUALCOMM_SDK="2.50.0.260828"
QNN_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QUALCOMM_SDK}/v${QUALCOMM_SDK}.zip"

TMP_ZIP=$(mktemp)
TMP_DIR=$(mktemp -d)
DEST_DIR="dist/qairt"

cleanup() {
  rm -rf "${TMP_ZIP}" "${TMP_DIR}"
}
trap cleanup EXIT

echo "Downloading Qualcomm AI Runtime SDK (v${QUALCOMM_SDK})..."
curl -L "${QNN_URL}" -o "${TMP_ZIP}"

echo "Extracting Qualcomm libraries to ${DEST_DIR}..."
unzip -q "${TMP_ZIP}" "qairt/${QUALCOMM_SDK}/lib/*" -d "${TMP_DIR}"

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}/arm64-v8a"
cp -f "${TMP_DIR}/qairt/${QUALCOMM_SDK}/lib/aarch64-android/"*.so "${DEST_DIR}/arm64-v8a/"
cp -f "${TMP_DIR}/qairt/${QUALCOMM_SDK}/lib/hexagon-v"*"/unsigned/"*.so "${DEST_DIR}/arm64-v8a/"

echo "Done! Build with: ./gradlew :app:assembleDebug -Pnpu.libs=\$(pwd)/${DEST_DIR}"
