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

# Script to deploy and run the image segmentation binary on an Android device
# via ADB.  Use this after building with build_prebuilt.sh.
#
# Usage:
#   ./deploy_and_run_on_android.sh \
#       --accelerator=[gpu|npu|cpu] \
#       --phone=[s24|s25|dim9400] \
#       [--use_gl_buffers] \
#       [--jit] \
#       [--host_npu_lib=<path>] \
#       [--host_npu_dispatch_lib=<path>] \
#       [--host_npu_compiler_lib=<path>] \
#       <cmake_build_dir>              # e.g. build/
set -e

# --- Default values ---
ACCELERATOR="gpu"
PHONE="s25"
BINARY_BUILD_PATH=""

usage() {
    echo "Usage: $0 --accelerator=[gpu|npu|cpu] --phone=[s24|s25|dim9400] <cmake_build_dir>"
    echo "  --accelerator : Specify the accelerator to use (gpu, npu, or cpu)."
    echo "  --phone       : Specify the phone model (s24/s25 for Qualcomm, dim9400 for MediaTek)."
    echo "  --jit         : Use JIT compilation (true). Defaults to false (Qualcomm AOT)."
    echo "  <cmake_build_dir> : Path to cmake build output directory (e.g. build/)."
    exit 1
}

if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided."
    usage
fi

TEMP=$(getopt -o '' --long accelerator:,phone:,use_gl_buffers,jit,host_npu_lib:,host_npu_dispatch_lib:,host_npu_compiler_lib: -- "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing options." >&2
    usage
fi

eval set -- "$TEMP"
unset TEMP

USE_GL_BUFFERS=false
USE_JIT=false
HOST_NPU_LIB=""
HOST_NPU_DISPATCH_LIB=""
HOST_NPU_COMPILER_LIB=""

while true; do
    case "$1" in
        '--accelerator')
            ACCELERATOR="$2"
            if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "npu" && "$ACCELERATOR" != "cpu" ]]; then
                echo "Error: Invalid value for --accelerator. Must be 'gpu', 'npu', or 'cpu'." >&2
                usage
            fi
            shift 2
            ;;
        '--phone')
            PHONE="$2"
            shift 2
            ;;
        '--use_gl_buffers')
            USE_GL_BUFFERS=true
            shift
            ;;
        '--jit')
            USE_JIT=true
            shift
            ;;
        '--host_npu_lib')
            HOST_NPU_LIB="$2"
            shift 2
            ;;
        '--host_npu_dispatch_lib')
            HOST_NPU_DISPATCH_LIB="$2"
            shift 2
            ;;
        '--host_npu_compiler_lib')
            HOST_NPU_COMPILER_LIB="$2"
            shift 2
            ;;
        '--')
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

echo "Selected Accelerator: $ACCELERATOR"
echo "Use GL Buffers: $USE_GL_BUFFERS"

if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments or invalid option."
    usage
fi

BINARY_BUILD_PATH="$1"

if [ ! -d "$BINARY_BUILD_PATH" ]; then
    echo "Error: The provided cmake_build_dir ($BINARY_BUILD_PATH) is not a valid directory."
    exit 1
fi

# --- Configuration ---
# Resolve to the use_prebuilt_litert source directory (script lives there).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Asset sources (shaders, images, models) come from the source directory.
HOST_SHADER_DIR="${SCRIPT_DIR}/shaders"
HOST_TEST_IMAGE_DIR="${SCRIPT_DIR}/test_images"
HOST_MODEL_DIR="${SCRIPT_DIR}/models"

# Determine binary name (npu on MediaTek still uses the same binary; vendor
# is passed as a runtime argument instead of separate build targets).
PACKAGE_NAME="cpp_segmentation_${ACCELERATOR}"

# CMake build products are flat in the build dir (no nested package path).
OUTPUT_PATH="${BINARY_BUILD_PATH}/${PACKAGE_NAME}"

# libLiteRt.so lives in the SDK directory (not the cmake build dir).
C_LIBRARY_LOCATION="${SCRIPT_DIR}/litert_cc_sdk"

# Device paths
DEVICE_BASE_DIR="/data/local/tmp/cpp_segmentation_android"
DEVICE_EXEC_NAME="cpp_segmentation_executable"
DEVICE_SHADER_DIR="${DEVICE_BASE_DIR}/shaders"
DEVICE_TEST_IMAGE_DIR="${DEVICE_BASE_DIR}/test_images"
DEVICE_MODEL_DIR="${DEVICE_BASE_DIR}/models"
DEVICE_NPU_LIBRARY_DIR="${DEVICE_BASE_DIR}/npu"

# GPU accelerator .so: check build dir first, then fall back to SDK dir.
HOST_GPU_LIBRARY_DIR="${BINARY_BUILD_PATH}"
SDK_GPU_ACCEL="${SCRIPT_DIR}/litert_cc_sdk/libLiteRtOpenClAccelerator.so"

# Set NPU library paths (Qualcomm).
if [[ -z "$HOST_NPU_LIB" ]]; then
    echo "Note: --host_npu_lib not set; defaulting to QAIRT SDK path."
    HOST_NPU_LIB="${BINARY_BUILD_PATH}/qairt/lib/"
fi
if [[ -z "$HOST_NPU_DISPATCH_LIB" ]]; then
    echo "Note: --host_npu_dispatch_lib not set; defaulting relative to build dir."
    HOST_NPU_DISPATCH_LIB="${BINARY_BUILD_PATH}"
fi
if [[ "$USE_JIT" == "true" && -z "$HOST_NPU_COMPILER_LIB" ]]; then
    HOST_NPU_COMPILER_LIB="${BINARY_BUILD_PATH}"
fi

# MTK library paths.
HOST_MTK_DISPATCH_LIB="${BINARY_BUILD_PATH}"
HOST_MTK_COMPILER_LIB="${BINARY_BUILD_PATH}"

# Qualcomm LD paths.
LD_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"
ADSP_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"

# --- NPU / MTK phone configuration ---
IS_MTK=false
QNN_STUB_LIB=""
QNN_SKEL_LIB=""
QNN_SKEL_PATH_ARCH=""
if [[ "$ACCELERATOR" == "npu" ]]; then
    case "$PHONE" in
        's24')
            QNN_STUB_LIB="libQnnHtpV75Stub.so"
            QNN_SKEL_LIB="libQnnHtpV75Skel.so"
            QNN_SKEL_PATH_ARCH="hexagon-v75"
            ;;
        's25')
            QNN_STUB_LIB="libQnnHtpV79Stub.so"
            QNN_SKEL_LIB="libQnnHtpV79Skel.so"
            QNN_SKEL_PATH_ARCH="hexagon-v79"
            ;;
        'dim9400')
            IS_MTK=true
            ;;
        *)
            echo "Error: Unsupported phone model '$PHONE'. Supported: s24, s25 (Qualcomm), dim9400 (MediaTek)." >&2
            exit 1
            ;;
    esac
fi

# --- Model Selection ---
MODEL_FILENAME="selfie_multiclass_256x256.tflite"
if [[ "$ACCELERATOR" == "npu" && "$USE_JIT" == "false" ]]; then
    if [[ "$PHONE" == "s24" ]]; then
        MODEL_FILENAME="selfie_multiclass_256x256_SM8650.tflite"
    elif [[ "$PHONE" == "s25" ]]; then
        MODEL_FILENAME="selfie_multiclass_256x256_SM8750.tflite"
    fi
fi

# --- Script Logic ---
echo "Starting deployment to Android device..."

HOST_EXEC_PATH="${OUTPUT_PATH}"
echo "Using binary: ${HOST_EXEC_PATH}"

if [ ! -f "${HOST_EXEC_PATH}" ]; then
    echo "Error: Executable not found at ${HOST_EXEC_PATH}"
    echo "Please run build_prebuilt.sh first."
    exit 1
fi

echo "Target device directory: ${DEVICE_BASE_DIR}"

# Clean up previous deployment
adb shell "rm -rf ${DEVICE_BASE_DIR}/*.so"

# Create directories on device
adb shell "mkdir -p ${DEVICE_BASE_DIR}"
adb shell "mkdir -p ${DEVICE_SHADER_DIR}"
adb shell "mkdir -p ${DEVICE_TEST_IMAGE_DIR}"
adb shell "mkdir -p ${DEVICE_MODEL_DIR}"
adb shell "mkdir -p ${DEVICE_NPU_LIBRARY_DIR}"
echo "Created directories on device."

# Push executable
adb push --sync "${HOST_EXEC_PATH}" "${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Pushed executable."

# Push shaders
adb push --sync "${HOST_SHADER_DIR}/passthrough_shader.vert" "${DEVICE_SHADER_DIR}/"
adb push --sync "${HOST_SHADER_DIR}/mask_blend_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push --sync "${HOST_SHADER_DIR}/resize_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push --sync "${HOST_SHADER_DIR}/preprocess_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push --sync "${HOST_SHADER_DIR}/deinterleave_masks.glsl" "${DEVICE_SHADER_DIR}/"
echo "Pushed shaders."

# Push test images
adb push --sync "${HOST_TEST_IMAGE_DIR}/image.jpeg" "${DEVICE_TEST_IMAGE_DIR}/"
echo "Pushed test images."

# Push model files
adb push --sync "${HOST_MODEL_DIR}/${MODEL_FILENAME}" "${DEVICE_MODEL_DIR}/"
echo "Pushed segmentation models."

# Push libLiteRt.so
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/"
adb push --sync "${C_LIBRARY_LOCATION}/libLiteRt.so" "${DEVICE_BASE_DIR}/"
echo "Pushed libLiteRt.so."

# Push GPU accelerator (optional)
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/"
if [[ "$ACCELERATOR" == "gpu" ]]; then
    GPU_ACCEL_FILE=""
    if [[ -f "${HOST_GPU_LIBRARY_DIR}/libLiteRtOpenClAccelerator.so" ]]; then
        GPU_ACCEL_FILE="${HOST_GPU_LIBRARY_DIR}/libLiteRtOpenClAccelerator.so"
    elif [[ -f "${SDK_GPU_ACCEL}" ]]; then
        GPU_ACCEL_FILE="${SDK_GPU_ACCEL}"
    fi
    if [[ -n "$GPU_ACCEL_FILE" ]]; then
        adb push --sync "${GPU_ACCEL_FILE}" "${DEVICE_BASE_DIR}/"
        echo "Pushed GPU accelerator library."
    else
        echo "Warning: libLiteRtOpenClAccelerator.so not found. GPU acceleration will use fallback."
    fi
fi

# Push NPU libraries
if [[ "$ACCELERATOR" == "npu" ]]; then
    if [[ "$IS_MTK" == "true" ]]; then
        # ---- MediaTek path ----
        if [[ -f "${HOST_MTK_DISPATCH_LIB}/libLiteRtDispatch_MediaTek.so" ]]; then
            adb push --sync "${HOST_MTK_DISPATCH_LIB}/libLiteRtDispatch_MediaTek.so" "${DEVICE_NPU_LIBRARY_DIR}/"
            echo "Pushed MediaTek dispatch library."
        else
            echo "Warning: libLiteRtDispatch_MediaTek.so not found at ${HOST_MTK_DISPATCH_LIB}."
        fi
        echo "Note: NeuroPilot runtime libs are system libs on the device."
        if [[ "$USE_JIT" == "true" ]]; then
            MTK_COMPILER="${HOST_MTK_COMPILER_LIB}/libLiteRtCompilerPlugin_MediaTek.so"
            if [[ -f "$MTK_COMPILER" ]]; then
                adb push --sync "$MTK_COMPILER" "${DEVICE_NPU_LIBRARY_DIR}/"
                echo "Pushed MediaTek compiler plugin."
            else
                echo "Warning: libLiteRtCompilerPlugin_MediaTek.so not found."
            fi
        fi
    else
        # ---- Qualcomm path ----
        adb push --sync "${HOST_NPU_DISPATCH_LIB}/libLiteRtDispatch_Qualcomm.so" "${DEVICE_NPU_LIBRARY_DIR}/"
        echo "Pushed Qualcomm dispatch library."
        adb push --sync "${HOST_NPU_LIB}/aarch64-android/libQnnHtp.so" "${DEVICE_NPU_LIBRARY_DIR}/"
        adb push --sync "${HOST_NPU_LIB}/aarch64-android/${QNN_STUB_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
        adb push --sync "${HOST_NPU_LIB}/aarch64-android/libQnnSystem.so" "${DEVICE_NPU_LIBRARY_DIR}/"
        adb push --sync "${HOST_NPU_LIB}/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_NPU_LIBRARY_DIR}/"
        adb push --sync "${HOST_NPU_LIB}/${QNN_SKEL_PATH_ARCH}/unsigned/${QNN_SKEL_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
        echo "Pushed Qualcomm NPU libraries."
        if [[ "$USE_JIT" == "true" ]]; then
            adb push --sync "${HOST_NPU_COMPILER_LIB}/libLiteRtCompilerPlugin_Qualcomm.so" "${DEVICE_NPU_LIBRARY_DIR}/"
            echo "Pushed Qualcomm compiler plugin library."
        fi
    fi
fi

# Set execute permissions
adb shell "chmod +x ${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Set execute permissions on device."

echo "Cleaning up previous run results"
adb shell "rm -f ${DEVICE_BASE_DIR}/output_segmented.png"

echo ""
echo "Deployment complete."

MODEL_PATH="./models/${MODEL_FILENAME}"
RUN_COMMAND="./${DEVICE_EXEC_NAME} ${MODEL_PATH} ./test_images/image.jpeg ./output_segmented.png"
if [[ "$ACCELERATOR" == "gpu" ]] && $USE_GL_BUFFERS; then
    RUN_COMMAND="${RUN_COMMAND} true"
fi

if [[ "$ACCELERATOR" == "npu" ]]; then
    if [[ "$IS_MTK" == "true" ]]; then
        MTK_LD_PATH="${DEVICE_NPU_LIBRARY_DIR}/:${DEVICE_BASE_DIR}/:/system_ext/lib64/"
        FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${MTK_LD_PATH}\" ${RUN_COMMAND}"
        if [[ "$USE_JIT" == "true" ]]; then
            FULL_COMMAND="${FULL_COMMAND} true mediatek"
        else
            FULL_COMMAND="${FULL_COMMAND} false mediatek"
        fi
    else
        FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ADSP_LIBRARY_PATH=\"${ADSP_LIBRARY_PATH}\" ${RUN_COMMAND}"
        if [[ "$USE_JIT" == "true" ]]; then
            FULL_COMMAND="${FULL_COMMAND} true"
        fi
    fi
else
    FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ${RUN_COMMAND}"
fi

echo "To run the segmentation on the device, use a command like this:"
echo "  adb shell \"${FULL_COMMAND}\""
adb shell "${FULL_COMMAND}"

echo ""
echo "To pull the result:"
echo "  adb pull ${DEVICE_BASE_DIR}/output_segmented.png ."
adb pull ${DEVICE_BASE_DIR}/output_segmented.png .
