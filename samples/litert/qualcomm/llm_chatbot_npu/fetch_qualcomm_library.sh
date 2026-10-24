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
# Provisions Qualcomm AI Runtime (QAIRT / QNN HTP) shared libraries into
# app/src/main/jniLibs/arm64-v8a/ from either a local QAIRT SDK directory or
# the public Qualcomm Software Center archive.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_QAIRT_VERSION="2.44.0.260225"
DEFAULT_HEXAGON_VERSION="79"
DEFAULT_DEST_DIR="${SCRIPT_DIR}/app/src/main/jniLibs/arm64-v8a"

QAIRT_DIR="${QAIRT_ROOT:-}"
QAIRT_VERSION="${DEFAULT_QAIRT_VERSION}"
HEXAGON_VERSION="${DEFAULT_HEXAGON_VERSION}"
DEST_DIR="${DEFAULT_DEST_DIR}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Copies or downloads Qualcomm QNN HTP runtime libraries into the Android app's
arm64-v8a JNI directory (${DEFAULT_DEST_DIR}).

Options:
  --qairt-dir <path>         Path to an extracted local QAIRT SDK root
                             (defaults to \$QAIRT_ROOT if set; downloads from
                             Qualcomm Software Center otherwise).
  --qairt-version <version>  QAIRT SDK release version to download when
                             --qairt-dir is not set (default: ${DEFAULT_QAIRT_VERSION}).
  --hexagon-version <ver>    Target Hexagon DSP architecture version:
                             69, 73, 75, 79 (SM8750 default), or 81.
  --dest-dir <path>          Destination arm64-v8a jniLibs directory
                             (default: ${DEFAULT_DEST_DIR}).
  -h, --help                 Show this help message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qairt-dir)
      QAIRT_DIR="${2:?Missing argument for --qairt-dir}"
      shift 2
      ;;
    --qairt-version)
      QAIRT_VERSION="${2:?Missing argument for --qairt-version}"
      shift 2
      ;;
    --hexagon-version)
      HEXAGON_VERSION="${2:?Missing argument for --hexagon-version}"
      shift 2
      ;;
    --dest-dir)
      DEST_DIR="${2:?Missing argument for --dest-dir}"
      shift 2
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

case "${HEXAGON_VERSION}" in
  69|73|75|79|81) ;;
  *)
    echo "Error: Unsupported --hexagon-version '${HEXAGON_VERSION}'. Expected one of: 69, 73, 75, 79, 81." >&2
    exit 1
    ;;
esac

TMP_DIR=""
cleanup() {
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT

if [[ -z "${QAIRT_DIR}" ]]; then
  TMP_DIR="$(mktemp -d)"
  QAIRT_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_VERSION}/v${QAIRT_VERSION}.zip"
  ARCHIVE_PATH="${TMP_DIR}/qairt_sdk.zip"

  echo "Downloading Qualcomm AI Runtime SDK v${QAIRT_VERSION}..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "${ARCHIVE_PATH}" "${QAIRT_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${ARCHIVE_PATH}" "${QAIRT_URL}"
  else
    echo "Error: Neither 'curl' nor 'wget' is installed." >&2
    exit 1
  fi

  echo "Extracting QNN shared libraries..."
  unzip -q "${ARCHIVE_PATH}" \
    "qairt/${QAIRT_VERSION}/lib/aarch64-android/libQnnHtp.so" \
    "qairt/${QAIRT_VERSION}/lib/aarch64-android/libQnnSystem.so" \
    "qairt/${QAIRT_VERSION}/lib/aarch64-android/libQnnHtpV${HEXAGON_VERSION}Stub.so" \
    "qairt/${QAIRT_VERSION}/lib/aarch64-android/libQnnHtpPrepare.so" \
    "qairt/${QAIRT_VERSION}/lib/hexagon-v${HEXAGON_VERSION}/unsigned/libQnnHtpV${HEXAGON_VERSION}Skel.so" \
    -d "${TMP_DIR}" || true

  QAIRT_DIR="${TMP_DIR}/qairt/${QAIRT_VERSION}"
fi

if [[ ! -d "${QAIRT_DIR}" ]]; then
  echo "Error: QAIRT directory not found: ${QAIRT_DIR}" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

REQUIRED_LIBS=(
  "${QAIRT_DIR}/lib/aarch64-android/libQnnHtp.so"
  "${QAIRT_DIR}/lib/aarch64-android/libQnnSystem.so"
  "${QAIRT_DIR}/lib/aarch64-android/libQnnHtpV${HEXAGON_VERSION}Stub.so"
  "${QAIRT_DIR}/lib/hexagon-v${HEXAGON_VERSION}/unsigned/libQnnHtpV${HEXAGON_VERSION}Skel.so"
)

for lib in "${REQUIRED_LIBS[@]}"; do
  if [[ ! -f "${lib}" ]]; then
    echo "Error: Required library missing: ${lib}" >&2
    exit 1
  fi
  cp -f "${lib}" "${DEST_DIR}/"
done

PREPARE_LIB="${QAIRT_DIR}/lib/aarch64-android/libQnnHtpPrepare.so"
if [[ -f "${PREPARE_LIB}" ]]; then
  cp -f "${PREPARE_LIB}" "${DEST_DIR}/"
fi

echo "Installed Qualcomm Hexagon v${HEXAGON_VERSION} QNN libraries into ${DEST_DIR}"
