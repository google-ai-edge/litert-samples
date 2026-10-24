#!/usr/bin/env bash
# Copyright 2026 Google LLC
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
#
# Builds, installs, optionally pushes a `.litertlm` model, and launches the
# Gemma 3 1B Qualcomm NPU sample on a connected Android device.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="com.example.gemma3_npu"
DEVICE_SERIAL="${ANDROID_SERIAL:-}"
MODEL_PATH=""
SKIP_BUILD=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Builds, installs, and launches ${PACKAGE_NAME} on a connected Qualcomm device.

Options:
  -s, --serial <serial>   Target ADB device serial (defaults to \$ANDROID_SERIAL).
  -m, --model <path>      Optional local '.litertlm' file to push to the app's
                          scoped storage directory before launching.
      --skip-build        Skip './gradlew :app:assembleDebug' and reuse existing APK.
  -h, --help              Show this help message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--serial)
      DEVICE_SERIAL="${2:?Missing argument for --serial}"
      shift 2
      ;;
    -m|--model)
      MODEL_PATH="${2:?Missing argument for --model}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ADB_CMD=(adb)
if [[ -n "${DEVICE_SERIAL}" ]]; then
  ADB_CMD+=( -s "${DEVICE_SERIAL}" )
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "Building debug APK..."
  (cd "${SCRIPT_DIR}" && ./gradlew :app:assembleDebug)
fi

APK_PATH="${SCRIPT_DIR}/app/build/outputs/apk/debug/app-debug.apk"
if [[ ! -f "${APK_PATH}" ]]; then
  echo "Error: APK not found at ${APK_PATH}" >&2
  exit 1
fi

echo "Installing ${APK_PATH}..."
"${ADB_CMD[@]}" install -r "${APK_PATH}"

if [[ -n "${MODEL_PATH}" ]]; then
  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: Model file not found: ${MODEL_PATH}" >&2
    exit 1
  fi
  REMOTE_DIR="/sdcard/Android/data/${PACKAGE_NAME}/files"
  MODEL_FILENAME="$(basename "${MODEL_PATH}")"
  echo "Pushing ${MODEL_FILENAME} to ${REMOTE_DIR}/..."
  "${ADB_CMD[@]}" shell mkdir -p "${REMOTE_DIR}"
  "${ADB_CMD[@]}" push "${MODEL_PATH}" "${REMOTE_DIR}/${MODEL_FILENAME}"
fi

echo "Launching ${PACKAGE_NAME}/.MainActivity..."
"${ADB_CMD[@]}" shell am start -n "${PACKAGE_NAME}/.MainActivity"
