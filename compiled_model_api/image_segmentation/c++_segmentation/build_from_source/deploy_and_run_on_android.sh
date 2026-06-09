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

# Script to deploy and run the image merger on an Android device via ADB
set -e

# --- Default values ---
ACCELERATOR="cpu" # Default accelerator if not specified
PHONE="s25" # Default phone model
BINARY_BUILD_PATH=""

# --- Usage ---
usage() {
    echo "Usage: $0 --accelerator=[gpu|npu|cpu] --phone=[s24|s25|s26|pixel10|dim9400] <binary_build_path>"
    echo "  --accelerator : Specify the accelerator to use (gpu, npu, or cpu)."
    echo "  --phone       : Specify the phone model (s24/s25 for Qualcomm, s26 for Samsung, pixel10 for Google Tensor, dim9400 for MediaTek)."
    echo "  --backend     : For GPU, specify backend (opengl or opencl). Defaults to opencl."
    echo "  --jit         : Use JIT compilation (true). Defaults to false (Qualcomm AOT)."
    echo "  <binary_build_path> : Path to bazel-bin/."
    exit 1
}

# --- Argument Parsing ---
# Check if any arguments are provided at all.
if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided."
    usage
fi

# Parse options
TEMP=$(getopt -o '' --long accelerator:,phone:,backend:,use_gl_buffers,jit,host_npu_lib:,host_npu_dispatch_lib:,host_npu_compiler_lib: -- "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing options." >&2
    usage
fi

eval set -- "$TEMP"
unset TEMP

USE_GL_BUFFERS=false
USE_JIT=false
BACKEND="opencl"
HOST_NPU_LIB=""
HOST_NPU_DISPATCH_LIB=""
HOST_NPU_COMPILER_LIB=""

while true; do
    case "$1" in
        '--accelerator')
            ACCELERATOR="$2"
            # Validate accelerator value
            if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "npu" && "$ACCELERATOR" != "cpu" ]]; then
                echo "Error: Invalid value for --accelerator. Must be 'gpu', 'npu', or 'cpu'." >&2
                usage
                exit 1
            fi
            shift 2
            ;;
        '--phone')
            PHONE="$2"
            shift 2
            ;;
        '--backend')
            BACKEND="$2"
            shift 2
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
            # This case should ideally not be reached if getopt is working correctly
            # and all options are defined.
            # However, it can catch unexpected issues or be used for positional args after options.
            break
            ;;
    esac
done

echo "Selected Accelerator: $ACCELERATOR"

# The remaining argument should be the binary_build_path
if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments or invalid option."
    usage
fi

BINARY_BUILD_PATH="$1"

# Check if the binary_build_path is a valid directory.
if [ ! -d "$BINARY_BUILD_PATH" ]; then
    echo "Error: The provided binary_build_path ($BINARY_BUILD_PATH) is not a valid directory."
    exit 1
fi

# --- Configuration ---
ROOT_DIR="compiled_model_api/image_segmentation/c++_segmentation"

PACKAGE_LOCATION="${ROOT_DIR}/build_from_source"
C_LIBRARY_LOCATION="${BINARY_BUILD_PATH}/external/litert_archive/litert/c"

# --- NPU Configuration ---
QNN_STUB_LIB=""
QNN_SKEL_LIB=""
QNN_SKEL_PATH_ARCH=""
IS_QUALCOMM=false
IS_GOOGLE_TENSOR=false
GOOGLE_TENSOR_MODEL=""
IS_MTK=false
case "$PHONE" in
    's24')
        QNN_STUB_LIB="libQnnHtpV75Stub.so"
        QNN_SKEL_LIB="libQnnHtpV75Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v75"
        IS_QUALCOMM=true
        ;;
    's25')
        QNN_STUB_LIB="libQnnHtpV79Stub.so"
        QNN_SKEL_LIB="libQnnHtpV79Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v79"
        IS_QUALCOMM=true
        ;;
    's26')
        IS_SAMSUNG=true
        ;;

    'pixel10')
        IS_GOOGLE_TENSOR=true
        GOOGLE_TENSOR_MODEL="Tensor_G5"
        ;;
    'dim9400')
        # MediaTek Dimensity 9400 (MT6991) — NeuroPilot v8
        IS_MTK=true
        ;;
    *)
        echo "Error: Unsupported phone model '$PHONE'. Supported models are 's24', 's25', 's26', 'pixel10', 'dim9400'." >&2
        exit 1
        ;;
esac

if [[ "$ACCELERATOR" == "npu" ]] && [[ "$IS_GOOGLE_TENSOR" == "true" ]]; then
    PACKAGE_NAME="cpp_segmentation_npu_google_tensor"
elif [[ "$ACCELERATOR" == "npu" ]] && [[ "$IS_MTK" == "true" ]]; then
    PACKAGE_NAME="cpp_segmentation_npu_mtk"
elif [[ "$ACCELERATOR" == "npu" ]] && [[ "$IS_SAMSUNG" == "true" ]]; then
    PACKAGE_NAME="cpp_segmentation_npu_samsung"
else
    PACKAGE_NAME="cpp_segmentation_${ACCELERATOR}"
fi
OUTPUT_PATH="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}"

# Device paths
DEVICE_BASE_DIR="/data/local/tmp/cpp_segmentation_android"
DEVICE_EXEC_NAME="cpp_segmentation_executable"
DEVICE_SHADER_DIR="${DEVICE_BASE_DIR}/shaders"
DEVICE_TEST_IMAGE_DIR="${DEVICE_BASE_DIR}/test_images"
DEVICE_MODEL_DIR="${DEVICE_BASE_DIR}/models"
DEVICE_NPU_LIBRARY_DIR="${DEVICE_BASE_DIR}/npu"
DEVICE_MTK_LIBRARY_DIR="${DEVICE_BASE_DIR}/mtk"


# Host paths (relative to this script's location or project root)
HOST_SHADER_DIR="${PACKAGE_LOCATION}/shaders"
HOST_TEST_IMAGE_DIR="${PACKAGE_LOCATION}/test_images"
HOST_MODEL_DIR="${PACKAGE_LOCATION}/models"
HOST_GPU_LIBRARY_DIR="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_prebuilts/android_arm64/"

# Set NPU library path based on the --npu_dispatch_lib_path flag
if [[ "$IS_QUALCOMM" == "true" ]] && [[ -z "$HOST_NPU_LIB" ]]; then
    echo "Defaulting to QNN libraries path."
    HOST_NPU_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/qairt/lib/"
fi
if [[ "$IS_QUALCOMM" == "true" ]] && [[ -z "$HOST_NPU_DISPATCH_LIB" ]]; then
    echo "Defaulting to internal dispatch library path."
    HOST_NPU_DISPATCH_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_archive/litert/vendors/qualcomm/dispatch"
elif [[ "$IS_GOOGLE_TENSOR" == "true" ]] && [[ -z "$HOST_NPU_DISPATCH_LIB" ]]; then
    echo "Defaulting to Google Tensor dispatch library path."
    HOST_NPU_DISPATCH_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_archive/litert/vendors/google_tensor/dispatch"
elif [[ "$IS_SAMSUNG" == "true" ]] && [[ -z "$HOST_NPU_DISPATCH_LIB" ]]; then
    echo "Defaulting to Samsung Exynos dispatch library path."
    HOST_NPU_DISPATCH_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_archive/litert/vendors/samsung/dispatch"
fi
if [[ "$USE_JIT" == "true" ]]; then
    echo "Using NPU JIT compilation."
    if [[ "$IS_QUALCOMM" == "true" ]] && [[ -z "$HOST_NPU_COMPILER_LIB" ]]; then
        HOST_NPU_COMPILER_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_archive/litert/vendors/qualcomm/compiler"
    elif [[ "$IS_GOOGLE_TENSOR" == "true" ]] && [[ -z "$HOST_NPU_COMPILER_LIB" ]]; then
        HOST_NPU_COMPILER_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_archive/litert/vendors/google_tensor/compiler"
    elif [[ "$IS_SAMSUNG" == "true" ]] && [[ -z "$HOST_NPU_COMPILER_LIB" ]]; then
        HOST_NPU_COMPILER_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_archive/litert/vendors/samsung/compiler"
    fi
fi

# MTK library paths — both dispatch and compiler plugin are built from @litert_archive
# and deployed from the cpp_segmentation_npu_mtk runfiles. The NeuroPilot runtime
# (libneuronusdk_adapter.mtk.so etc.) is a system lib in /system_ext/lib64/ on device.
HOST_MTK_DISPATCH_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/cpp_segmentation_npu_mtk.runfiles/litert_archive/litert/vendors/mediatek/dispatch"
# MTK compiler plugin — built from @litert_archive; same runfiles tree.
HOST_MTK_COMPILER_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/cpp_segmentation_npu_mtk.runfiles/litert_archive/litert/vendors/mediatek/compiler"

# Qualcomm NPU library path
LD_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"
ADSP_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"

# --- Model Selection ---
MODEL_FILENAME="selfie_multiclass_256x256.tflite"
if [[ "$ACCELERATOR" == "npu" ]] && [[ "$USE_JIT" == "false" ]]; then
    if [[ "$IS_GOOGLE_TENSOR" == "true" ]]; then
        MODEL_FILENAME="selfie_multiclass_256x256_${GOOGLE_TENSOR_MODEL}.tflite"
    elif [[ "$PHONE" == "s24" ]]; then
        MODEL_FILENAME="selfie_multiclass_256x256_SM8650.tflite"
    elif [[ "$PHONE" == "s25" ]]; then
        MODEL_FILENAME="selfie_multiclass_256x256_SM8750.tflite"
    elif [[ "$PHONE" == "s26" ]]; then
        MODEL_FILENAME="selfie_multiclass_256x256_E9965.tflite"
    fi
fi


# --- Script Logic ---
echo "Starting deployment to Android device..."

# Determine executable path
HOST_EXEC_PATH="${OUTPUT_PATH}"
echo "Using output path: ${HOST_EXEC_PATH}"

if [ ! -f "${HOST_EXEC_PATH}" ]; then
    echo "Error: Executable not found at ${HOST_EXEC_PATH}"
    echo "Please ensure the project has been built and the correct path is provided."
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
adb shell "mkdir -p ${DEVICE_MTK_LIBRARY_DIR}"
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

# Push c api shared library
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/"
adb push --sync "${C_LIBRARY_LOCATION}/libLiteRt.so" "${DEVICE_BASE_DIR}/"
echo "Pushed c api shared library."

# Push gpu accelerator shared library
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/"
if [[ "$ACCELERATOR" == "gpu" ]]; then
    adb push --sync "${HOST_GPU_LIBRARY_DIR}/libLiteRtClGlAccelerator.so" "${DEVICE_BASE_DIR}/"
fi
echo "Pushed gpu accelerator shared library."

# Push NPU libraries (Qualcomm or MediaTek depending on phone)
if [[ "$ACCELERATOR" == "npu" ]]; then
  if [[ "$IS_SAMSUNG" == "true" ]]; then
    # ---- Samsung Exynos path ----
    adb push --sync "${HOST_NPU_DISPATCH_LIB}/libLiteRtDispatch_Samsung.so" "${DEVICE_NPU_LIBRARY_DIR}/"
    echo ${HOST_NPU_DISPATCH_LIB}/libLiteRtDispatch_Samsung.so
    echo "Pushed Samsung Exynos dispatch library."
    # ONE API runtime (libEden_nn.so etc.) lives in /vendor/lib64/ on device.
    echo "Note: ONE API runtime libs are system libs on the device."
    if [[ "$USE_JIT" == "true" ]]; then
        SAMSUNG_COMPILER="${HOST_NPU_COMPILER_LIB}/libLiteRtCompilerPlugin_Samsung.so"
        if [[ -f "$SAMSUNG_COMPILER" ]]; then
            adb push --sync "$SAMSUNG_COMPILER" "${DEVICE_NPU_LIBRARY_DIR}/"
            echo "Pushed Samsung compiler plugin."
        else
            echo "Warning: libLiteRtCompilerPlugin_Samsung.so not found at ${SAMSUNG_COMPILER}."
            echo "  Build it from LiteRT source and pass --host_npu_compiler_lib=<dir>."
        fi
    fi
  elif [[ "$IS_GOOGLE_TENSOR" == "true" ]]; then
    # ---- Google Tensor path ----
    adb push --sync "${HOST_NPU_DISPATCH_LIB}/libLiteRtDispatch_GoogleTensor.so" "${DEVICE_NPU_LIBRARY_DIR}/"
    echo "Pushed Google Tensor NPU dispatch library."
    # If JIT, push compiler plugin.
    if [[ "$USE_JIT" == "true" ]]; then
        adb push --sync "${HOST_NPU_COMPILER_LIB}/libLiteRtCompilerPlugin_google_tensor.so" "${DEVICE_NPU_LIBRARY_DIR}/"
        echo "Pushed Google Tensor NPU compiler library."
    fi
  elif [[ "$IS_MTK" == "true" ]]; then
    # ---- MediaTek path ----
    # Push the pre-built dispatch library from litert_npu_runtime_libraries.
    adb push --sync "${HOST_MTK_DISPATCH_LIB}/libLiteRtDispatch_MediaTek.so" "${DEVICE_NPU_LIBRARY_DIR}/"
    echo "Pushed MediaTek dispatch library."
    # NeuroPilot runtime (libneuronusdk_adapter.mtk.so etc.) lives in
    # /system_ext/lib64/ on the device -- nothing extra to push.
    echo "Note: NeuroPilot runtime libs are system libs on the device."
    if [[ "$USE_JIT" == "true" ]]; then
        MTK_COMPILER="${HOST_MTK_COMPILER_LIB}/libLiteRtCompilerPlugin_MediaTek.so"
        if [[ -f "$MTK_COMPILER" ]]; then
            adb push --sync "$MTK_COMPILER" "${DEVICE_NPU_LIBRARY_DIR}/"
            echo "Pushed MediaTek compiler plugin."
        else
            echo "Warning: libLiteRtCompilerPlugin_MediaTek.so not found."
            echo "  Build it from LiteRT source and set HOST_MTK_COMPILER_LIB."
        fi
    fi
  elif [[ "$IS_QUALCOMM" == "true" ]]; then
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
echo "To run the async segmentation on the device, use a command like this:"

MODEL_PATH="./models/${MODEL_FILENAME}"

RUN_COMMAND="./${DEVICE_EXEC_NAME} ${MODEL_PATH} ./test_images/image.jpeg ./output_segmented.png"
if [[ "$ACCELERATOR" == "gpu" ]]; then
    RUN_COMMAND="${RUN_COMMAND} ${BACKEND}"
fi

if [[ "$ACCELERATOR" == "npu" ]]; then
    if [[ "$IS_SAMSUNG" == "true" ]]; then
        # Samsung: dispatch dir is /npu/, also include /vendor/lib64/ for ONE API runtime.
        SAMSUNG_LD_PATH="${DEVICE_NPU_LIBRARY_DIR}/:${DEVICE_BASE_DIR}/"
        FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${SAMSUNG_LD_PATH}\" ${RUN_COMMAND}"
        # Pass JIT flag + 'samsung' vendor to binary.
        if [[ "$USE_JIT" == "true" ]]; then
            FULL_COMMAND="${FULL_COMMAND} true samsung"
        else
            FULL_COMMAND="${FULL_COMMAND} false samsung"
        fi
    elif [[ "$IS_GOOGLE_TENSOR" == "true" ]]; then
        FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ${RUN_COMMAND}"
        if [[ "$USE_JIT" == "true" ]]; then
            FULL_COMMAND="${FULL_COMMAND} true google_tensor"
        else
            FULL_COMMAND="${FULL_COMMAND} false google_tensor"
        fi
    elif [[ "$IS_MTK" == "true" ]]; then
        # MTK: dispatch dir is /npu/, also include /system_ext/lib64/ for NeuroPilot.
        MTK_LD_PATH="${DEVICE_NPU_LIBRARY_DIR}/:${DEVICE_BASE_DIR}/:/system_ext/lib64/"
        FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${MTK_LD_PATH}\" ${RUN_COMMAND}"
        # Pass JIT flag + 'mediatek' vendor to binary.
        if [[ "$USE_JIT" == "true" ]]; then
            FULL_COMMAND="${FULL_COMMAND} true mediatek"
        else
            FULL_COMMAND="${FULL_COMMAND} false mediatek"
        fi
    else
        # Qualcomm: standard LD_LIBRARY_PATH + ADSP path.
        FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ADSP_LIBRARY_PATH=\"${ADSP_LIBRARY_PATH}\" ${RUN_COMMAND}"
        if [[ "$USE_JIT" == "true" ]]; then
            FULL_COMMAND="${FULL_COMMAND} true qualcomm"
        else
            FULL_COMMAND="${FULL_COMMAND} false qualcomm"
        fi
    fi
else
    FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ${RUN_COMMAND}"
fi

echo "  adb shell \"${FULL_COMMAND}\""
adb shell "${FULL_COMMAND}"

echo ""
echo "To pull the result:"
echo "  adb pull ${DEVICE_BASE_DIR}/output_segmented.png ."
adb pull ${DEVICE_BASE_DIR}/output_segmented.png .
