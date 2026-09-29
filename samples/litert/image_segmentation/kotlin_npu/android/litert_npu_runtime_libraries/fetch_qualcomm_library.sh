#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

tmp_dir=$(mktemp -d)
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

# LINT.IfChange(fetch_qairt_sdk_version)
QAIRT_URL='https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip'
QAIRT_CONTENT_DIR='qairt/2.50.0.260828'
# LINT.ThenChange(
#     ./fetch_qualcomm_library_jit.sh:fetch_qairt_sdk_version,
#     ../../../opensource_only/third_party/qairt/workspace.bzl:bazel_qairt_sdk_version,
#     ../../../opensource_only/ci/tools/python/vendor_sdk/qualcomm/setup.py:wheel_qairt_sdk_version,
#     ../../vendors/CMakeLists.txt:qairt_headers_dir,
# )

pushd "$tmp_dir"
wget "$QAIRT_URL" -O qairt_sdk.zip
unzip qairt_sdk.zip *.so > /dev/null
popd

SOURCE_DIR="${tmp_dir}/${QAIRT_CONTENT_DIR}"
HTP_VERSIONS=(68 69 73 75 79 81)
DSP_VERSIONS=(65 66)
JNI_ARM64_DIR="src/main/jni/arm64-v8a"
DEST_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))

# copy_lib <source path relative to the SDK root> <destination module>
copy_lib() {
  local source_path="${SOURCE_DIR}/$1"
  local module="$2"
  if [[ ! -f "${source_path}" ]]; then
    return 1
  fi
  echo "Copying $(basename "${source_path}") to ${module}"
  mkdir -p "${DEST_DIR}/${module}/${JNI_ARM64_DIR}/"
  cp -rf "${source_path}" "${DEST_DIR}/${module}/${JNI_ARM64_DIR}/"
}

# copy_first_lib <destination module> <source path>... -- copies the first one that exists.
copy_first_lib() {
  local module="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if copy_lib "${candidate}" "${module}"; then
      return 0
    fi
  done
  return 0
}

# Libraries used by every Qualcomm SoC, alongside the LiteRT runtime AAR.
copy_first_lib "qualcomm_runtime_common" "lib/aarch64-android/libQnnSystem.so"

# Libraries shared by the HTP SoCs only.
copy_first_lib "qualcomm_runtime_htp" "lib/aarch64-android/libQnnHtp.so"

# Libraries shared by the DSP SoCs only.
copy_first_lib "qualcomm_runtime_dsp" "lib/aarch64-android/libQnnDsp.so"

# SoC specific libraries.
for version in "${HTP_VERSIONS[@]}"; do
  copy_first_lib "qualcomm_runtime_v${version}" \
    "lib/hexagon-v${version}/unsigned/libQnnHtpV${version}Skel.so"
  copy_first_lib "qualcomm_runtime_v${version}" \
    "lib/aarch64-android/libQnnHtpV${version}Stub.so"
done

for version in "${DSP_VERSIONS[@]}"; do
  copy_first_lib "qualcomm_runtime_v${version}" \
    "lib/hexagon-v${version}/unsigned/libQnnDspV${version}Skel.so" \
    "lib/hexagon-v${version}/unsigned/libSnpeDspV${version}Skel.so"
  copy_first_lib "qualcomm_runtime_v${version}" \
    "lib/aarch64-android/libQnnDspV${version}Stub.so" \
    "lib/aarch64-android/libSnpeDspV${version}Stub.so"
done
