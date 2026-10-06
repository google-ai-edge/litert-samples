#!/bin/bash

set -euo pipefail

tmp_dir=$(mktemp -d)
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

# Keep these versions paired. The NPU compiler from the driver release must be
# compatible with the OpenVINO runtime used by the LiteRT Intel plugins.
NPU_DRIVER_VERSION='1.38.0.20260910-34487311128'
NPU_DRIVER_ARCHIVE="intel-linux-npu-driver-${NPU_DRIVER_VERSION}-android-x86_64.tar.gz"
NPU_DRIVER_URL="https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/${NPU_DRIVER_ARCHIVE}"
NPU_DRIVER_CONTENT_DIR="intel-linux-npu-driver-${NPU_DRIVER_VERSION}-android-x86_64"
NPU_DRIVER_SHA256='b8c964ce6db77a5fecfeb53ffcb72cfb91b595b78340a7c1fea233cea90355de'

OPENVINO_VERSION='2026.3.1.22476.56d9685302d'
OPENVINO_ARCHIVE="openvino_toolkit_android_${OPENVINO_VERSION}_x86_64.tgz"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.3.1/linux/${OPENVINO_ARCHIVE}"
OPENVINO_CONTENT_DIR="openvino_toolkit_android_${OPENVINO_VERSION}_x86_64"
OPENVINO_SHA256='757eb8140039c9060a9e2668dc3885ab0daf24fabe7adf71a8e9e81da148049c'

# libLiteRtCompilerPlugin_IntelOpenvino.so and libLiteRtDispatch_IntelOpenvino.so
# come from the litert-npu-runtime-intel-openvino Maven package, not from here.
# Level Zero is part of the device NPU driver stack and is intentionally not
# bundled. The published Maven package owns its Android manifest declarations.

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
JNI_X86_64_DIR='src/main/jni/x86_64'
DEST_DIR="${SCRIPT_DIR}/../app_intel/${JNI_X86_64_DIR}"

verify_sha256() {
  local file=$1
  local expected=$2

  if ! echo "${expected}  ${file}" | sha256sum --check --quiet -; then
    echo "ERROR: SHA-256 mismatch for $(basename "$file")" >&2
    exit 1
  fi
}

copy_library() {
  local source_file=$1

  if [[ ! -f "$source_file" ]]; then
    echo "ERROR: Expected library not found: $source_file" >&2
    exit 1
  fi

  echo "Copying $(basename "$source_file")"
  cp -f "$source_file" "$DEST_DIR/"
}

echo "Downloading Intel NPU driver ${NPU_DRIVER_VERSION}"
wget "$NPU_DRIVER_URL" -O "${tmp_dir}/${NPU_DRIVER_ARCHIVE}"
verify_sha256 "${tmp_dir}/${NPU_DRIVER_ARCHIVE}" "$NPU_DRIVER_SHA256"

echo "Extracting Intel NPU driver"
tar -xzf "${tmp_dir}/${NPU_DRIVER_ARCHIVE}" -C "$tmp_dir"

echo "Downloading OpenVINO ${OPENVINO_VERSION} for Android x86_64"
wget "$OPENVINO_URL" -O "${tmp_dir}/${OPENVINO_ARCHIVE}"
verify_sha256 "${tmp_dir}/${OPENVINO_ARCHIVE}" "$OPENVINO_SHA256"

echo "Extracting OpenVINO"
tar -xzf "${tmp_dir}/${OPENVINO_ARCHIVE}" -C "$tmp_dir"

NPU_DRIVER_LIB_DIR="${tmp_dir}/${NPU_DRIVER_CONTENT_DIR}/lib"
OPENVINO_RUNTIME_DIR="${tmp_dir}/${OPENVINO_CONTENT_DIR}/runtime"
OPENVINO_LIB_DIR="${OPENVINO_RUNTIME_DIR}/lib/intel64"
OPENVINO_TBB_LIB_DIR="${OPENVINO_RUNTIME_DIR}/3rdparty/tbb/lib"

mkdir -p "$DEST_DIR"

echo "Copying Intel NPU compiler libraries to ${DEST_DIR}"
NPU_DRIVER_LIBS=(
  libopenvino_intel_npu_compiler.so
  libopenvino_intel_npu_compiler_loader.so
  libc++_shared.so
)
for library in "${NPU_DRIVER_LIBS[@]}"; do
  copy_library "${NPU_DRIVER_LIB_DIR}/${library}"
done

echo "Copying OpenVINO NPU runtime libraries to ${DEST_DIR}"
OPENVINO_LIBS=(
  libopenvino.so
  libopenvino_ir_frontend.so
  libopenvino_tensorflow_lite_frontend.so
  libopenvino_intel_npu_plugin.so
)
for library in "${OPENVINO_LIBS[@]}"; do
  copy_library "${OPENVINO_LIB_DIR}/${library}"
done

echo "Copying matching OpenVINO TBB libraries to ${DEST_DIR}"
OPENVINO_TBB_LIBS=(
  libtbb.so
  libtbbmalloc.so
  libtbbmalloc_proxy.so
)
for library in "${OPENVINO_TBB_LIBS[@]}"; do
  copy_library "${OPENVINO_TBB_LIB_DIR}/${library}"
done

echo
echo "Intel runtime libraries are ready in ${DEST_DIR}:"
for library_path in "$DEST_DIR"/*.so; do
  echo "  $(basename "$library_path")"
done
