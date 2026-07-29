#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

tmp_dir=$(mktemp -d)
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

# QAIRT SDK path provided by the user
SOURCE_DIR="/mnt/c/Users/rawat/Downloads/v2.42.0.251225/qairt/2.42.0.251225"

QNN_VERSIONS=(69 73 75 79 81)
JNI_ARM64_DIR="src/main/jni/arm64-v8a"
DEST_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))

for version in "${QNN_VERSIONS[@]}"; do
  echo "Copying libraries to ${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnHtp.so
  cp -rf "${SOURCE_DIR}/lib/aarch64-android/libQnnHtp.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnSystem.so
  cp -rf "${SOURCE_DIR}/lib/aarch64-android/libQnnSystem.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnHtpV${version}Skel.so
  cp -rf "${SOURCE_DIR}/lib/hexagon-v${version}/unsigned/libQnnHtpV${version}Skel.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnHtpV${version}Stub.so
  cp -rf "${SOURCE_DIR}/lib/aarch64-android/libQnnHtpV${version}Stub.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"

  # libQnnHtpPrepare.so (Required for JIT compilation)
  cp -rf "${SOURCE_DIR}/lib/aarch64-android/libQnnHtpPrepare.so" \
    "${DEST_DIR}/qualcomm_runtime_v${version}/${JNI_ARM64_DIR}/"
done
