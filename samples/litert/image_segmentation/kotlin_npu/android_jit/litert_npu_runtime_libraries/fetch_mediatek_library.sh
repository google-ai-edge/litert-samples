#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

tmp_dir=$(mktemp -d)
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

NEURO_PILOT_SDK_URL='https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/66f2c33a-2005-4f0b-afef-2053c8654e4f.gz'
V8_VERSION='v8_0_10'
V9_VERSION='v9_0_3'

pushd "$tmp_dir"
wget "$NEURO_PILOT_SDK_URL" -O neuro_pilot_sdk.gz
tar -xvzf neuro_pilot_sdk.gz > /dev/null
popd

SOURCE_DIR="${tmp_dir}/neuro_pilot"
JNI_ARM64_DIR="src/main/jni/arm64-v8a"
DEST_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))

echo "Copying libraries to ${DEST_DIR}/mediatek_runtime_v8/${JNI_ARM64_DIR}/"
mkdir -p "${DEST_DIR}/mediatek_runtime_v8/${JNI_ARM64_DIR}/"
cp -rf "${SOURCE_DIR}/${V8_VERSION}/usdk/lib64/libneuronusdk_adapter.mtk.so" \
  "${DEST_DIR}/mediatek_runtime_v8/${JNI_ARM64_DIR}/"

echo "Copying libraries to ${DEST_DIR}/mediatek_runtime_v9/${JNI_ARM64_DIR}/"
mkdir -p "${DEST_DIR}/mediatek_runtime_v9/${JNI_ARM64_DIR}/"
cp -rf "${SOURCE_DIR}/${V9_VERSION}/usdk/lib64/libneuronusdk_adapter.so" \
  "${DEST_DIR}/mediatek_runtime_v9/${JNI_ARM64_DIR}/"
