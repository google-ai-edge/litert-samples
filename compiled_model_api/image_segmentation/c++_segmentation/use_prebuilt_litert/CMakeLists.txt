# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.22)
project(cpp_segmentation CXX)

# The LiteRT C++ SDK requires C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# LiteRT C++ SDK
# ---------------------------------------------------------------------------
# LITERT_CC_SDK_DIR must point to the directory that contains the SDK's own
# CMakeLists.txt, libLiteRt.so, and *.cmake files.
#
# With this repo's layout:
#   litert_cc_sdk/    ← pass THIS directory
#     CMakeLists.txt
#     libLiteRt.so
#     abseil-cpp.cmake
#     ...
#
# build_prebuilt.sh sets this automatically. To pass manually:
#   -DLITERT_CC_SDK_DIR=$(pwd)/litert_cc_sdk
if(NOT DEFINED LITERT_CC_SDK_DIR)
  message(FATAL_ERROR "LITERT_CC_SDK_DIR must be set. "
    "Pass -DLITERT_CC_SDK_DIR=/path/to/sdk on the cmake command line.")
endif()

# Pull in the SDK's CMakeLists.txt which exposes the `litert_cc_api` target.
# Suppress deprecation warnings that come from inside the SDK's own headers
# (litert_expected.h uses an older Error() constructor marked deprecated in v2.1.1).
add_compile_options(-Wno-deprecated-declarations)
add_subdirectory("${LITERT_CC_SDK_DIR}" "${CMAKE_BINARY_DIR}/litert_cc_sdk")
include_directories("${LITERT_CC_SDK_DIR}")

# The SDK fetches OpenCL headers via FetchContent into ${CMAKE_BINARY_DIR}/opencl_headers.
# Expose CL/cl.h so our targets can compile litert_tensor_buffer.h.
include_directories("${CMAKE_BINARY_DIR}/opencl_headers")


# ---------------------------------------------------------------------------
# STB (header-only image library, bundled in third_party/stb/)
# ---------------------------------------------------------------------------
# The headers are expected in:
#   <source_dir>/third_party/stb/stb_image.h
#   <source_dir>/third_party/stb/stb_image_write.h
#   <source_dir>/third_party/stb/stb_image_resize.h
# build_prebuilt.sh downloads them automatically. Include via the source dir.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Shared library/utility targets
# ---------------------------------------------------------------------------
add_library(image_utils STATIC
  image_utils.cc
)
target_include_directories(image_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(timing_utils STATIC
  timing_utils.cc
)
target_link_libraries(timing_utils PUBLIC litert_cc_api)
target_include_directories(timing_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(image_processor STATIC
  image_processor.cc
)
target_include_directories(image_processor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(image_processor
  PUBLIC
    image_utils
    litert_cc_api
    EGL
    GLESv3
)

# ---------------------------------------------------------------------------
# Helper macro to build a segmentation executable
# ---------------------------------------------------------------------------
macro(add_segmentation_target target_name main_source)
  add_executable(${target_name} ${main_source})
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target_name}
    PRIVATE
      image_processor
      image_utils
      timing_utils
      litert_cc_api
      absl::log_internal_check_op
      absl::log_internal_message
      absl::log_internal_log_sink_set
      absl::log
      absl::check
      EGL
      GLESv3
      android
      log
  )
  # Strip debug info in release builds to reduce binary size.
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${target_name} PRIVATE -O2)
  endif()
endmacro()

# ---------------------------------------------------------------------------
# CPU executable
# ---------------------------------------------------------------------------
add_segmentation_target(cpp_segmentation_cpu main_cpu.cc)

# ---------------------------------------------------------------------------
# GPU executable
# ---------------------------------------------------------------------------
add_segmentation_target(cpp_segmentation_gpu main_gpu.cc)

# NOTE: libLiteRtOpenClAccelerator.so (GPU accelerator) and libLiteRt.so are
# loaded at runtime via LD_LIBRARY_PATH by deploy_and_run_on_android.sh.
# No compile-time linkage needed; just ensure the .so files are pushed to
# the device alongside the binary.

# ---------------------------------------------------------------------------
# NPU executable (Qualcomm + MediaTek via vendor arg)
# ---------------------------------------------------------------------------
add_segmentation_target(cpp_segmentation_npu main_npu.cc)
