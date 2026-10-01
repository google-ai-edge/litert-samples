# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

# wasm build of the SAM 2 ModelChain pipeline (emscripten, JSPI).
#
# Compiles, from the same sources as the native Bazel build:
#   - the LiteRT Tensor API (graph building + .tflite serialization)
#   - the SAM 2 Tensor API graphs (litert-samples)
#   - LiteRT's ModelChain (tensor/runners/model_chain.cc) and C++ API headers
#   - this project's pipeline (../cc)
# against litert_js_runtime.cc, a LiteRT runtime that executes on WebGPU
# through LiteRT.js. Configure through build.sh.

cmake_minimum_required(VERSION 3.22)
project(sam2_chain_wasm CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

foreach(v LITERT_ROOT SAMPLES_ROOT ABSL_ROOT FLATBUFFERS_ROOT GEN_DIR)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "${v} must be set (see build.sh)")
  endif()
endforeach()
set(CC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cc)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
add_subdirectory(${ABSL_ROOT} absl EXCLUDE_FROM_ALL)

set(SAM2 ${SAMPLES_ROOT}/models/sam2/sam2_hiera_tiny_video/tensor_api)
add_executable(sam2_chain
  # LiteRT Tensor API
  ${LITERT_ROOT}/tensor/tensor.cc
  ${LITERT_ROOT}/tensor/buffer.cc
  ${LITERT_ROOT}/tensor/internal/graph.cc
  ${LITERT_ROOT}/tensor/internal/graph_probe.cc
  ${LITERT_ROOT}/tensor/internal/graph_traversal.cc
  ${LITERT_ROOT}/tensor/internal/shape.cc
  ${LITERT_ROOT}/tensor/backends/tflite/arithmetic_tflite.cc
  ${GEN_DIR}/tflite_flatbuffer_conversion.cc
  ${LITERT_ROOT}/tensor/examples/gemma3/safetensor_loader.cc
  # LiteRT ModelChain + the two LiteRT C sources the C++ API calls directly
  ${LITERT_ROOT}/tensor/runners/model_chain.cc
  ${LITERT_ROOT}/litert/c/internal/litert_logging.cc
  ${LITERT_ROOT}/litert/c/litert_layout.cc
  # SAM 2 Tensor API graphs
  ${SAM2}/sam2_image/sam2_graph.cc
  ${SAM2}/sam2_image/sam2_weights.cc
  ${SAM2}/sam2_video/sam2v_graph.cc
  ${SAM2}/sam2_video/sam2v_weights.cc
  # Pipeline
  ${CC_DIR}/chain_graphs.cc
  ${CC_DIR}/host_plan.cc
  ${CC_DIR}/signature_stage.cc
  ${CC_DIR}/sam2_pipeline.cc
  # Browser runtime + JS API
  litert_js_runtime.cc
  sam2_wasm.cc
)

target_include_directories(sam2_chain PRIVATE
  ${GEN_DIR}              # runtime_stubs.inc, tflite/schema/mutable/schema_generated.h
  ${LITERT_ROOT}
  ${SAMPLES_ROOT}
  ${CC_DIR}
  ${FLATBUFFERS_ROOT}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(sam2_chain PRIVATE -O2 -w -fwasm-exceptions)

target_link_libraries(sam2_chain PRIVATE
  absl::btree absl::check absl::flat_hash_map absl::flat_hash_set absl::log
  absl::status absl::statusor absl::strings absl::str_format absl::span
  absl::inlined_vector absl::random_random absl::algorithm_container
  absl::optional absl::cleanup
)

target_link_options(sam2_chain PRIVATE
  -O2
  -fwasm-exceptions
  -lembind
  --profiling-funcs
  -sJSPI
  -sMODULARIZE=1
  -sEXPORT_ES6=1
  -sEXPORT_NAME=createSam2Chain
  -sENVIRONMENT=web
  -sALLOW_MEMORY_GROWTH=1
  -sMAXIMUM_MEMORY=4GB
  -sINITIAL_MEMORY=64MB
  -sSTACK_SIZE=4MB
  -sFORCE_FILESYSTEM=1
  -sEXPORTED_RUNTIME_METHODS=FS
  -sINCOMING_MODULE_JS_API=litert,print,printErr,locateFile
  --js-library=${CMAKE_CURRENT_SOURCE_DIR}/litert_js_bridge.js
)
set_target_properties(sam2_chain PROPERTIES SUFFIX ".mjs"
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/litert_js_bridge.js)
