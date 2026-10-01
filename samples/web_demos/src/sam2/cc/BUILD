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

# SAM 2 video as a LiteRT ModelChain over Tensor API models.
#
# Built in the litert-samples workspace, which provides @litert_archive and
# the SAM 2 Tensor API graph libraries (//models/sam2/...):
#   bazel build //samples/web_demos/src/sam2/cc:sam2_chain_main

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "@litert_archive//litert/build_common:special_rule.bzl",
    "litert_gpu_accelerator_prebuilts",
)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "chain_graphs",
    srcs = ["chain_graphs.cc"],
    hdrs = ["chain_graphs.h"],
    deps = [
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_config",
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_graph",
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video:sam2v_graph",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@litert_archive//tensor",
        "@litert_archive//tensor:arithmetic",
        "@litert_archive//tensor:buffer",
        "@litert_archive//tensor:datatypes",
        "@litert_archive//tensor/backends/tflite:arithmetic_tflite",
        "@litert_archive//tensor/backends/tflite:tflite_flatbuffer_conversion",
    ],
)

cc_library(
    name = "host_plan",
    srcs = ["host_plan.cc"],
    hdrs = ["host_plan.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert_archive//tensor/examples/gemma3:safetensor_loader",
        "@litert_archive//tensor/examples/gemma3:safetensors",
    ],
)

cc_library(
    name = "signature_stage",
    srcs = ["signature_stage.cc"],
    hdrs = ["signature_stage.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_archive//litert/cc:litert_tensor_buffer_types",
        "@litert_archive//tensor/runners:model_chain",
        "@litert_archive//tensor/runners/litert:litert_buffer",
    ],
)

cc_library(
    name = "sam2_pipeline",
    srcs = ["sam2_pipeline.cc"],
    hdrs = ["sam2_pipeline.h"],
    deps = [
        ":chain_graphs",
        ":host_plan",
        ":signature_stage",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "@litert_archive//tensor/runners:model_chain",
        "@litert_archive//tensor/runners/litert:litert_buffer",
    ],
)

cc_binary(
    name = "sam2_chain_main",
    srcs = ["sam2_chain_main.cc"],
    data = litert_gpu_accelerator_prebuilts(),
    deps = [
        ":chain_graphs",
        ":host_plan",
        ":sam2_pipeline",
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_weights",
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video:sam2v_weights",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_options",
        "@litert_archive//litert/cc/options:litert_gpu_options",
        "@litert_archive//tensor/backends/tflite:tflite_flatbuffer_conversion",
    ],
)

cc_test(
    name = "host_plan_test",
    srcs = ["host_plan_test.cc"],
    args = ["--goldens=$(location //samples/web_demos/src/sam2/cc/testdata:goldens.safetensors)"],
    data = [
        "//samples/web_demos/src/sam2/cc/testdata:goldens.safetensors",
        "//samples/web_demos/src/sam2/cc/testdata:sam2_host_consts.safetensors",
    ],
    deps = [
        ":host_plan",
        "@com_google_googletest//:gtest",
        "@litert_archive//tensor/examples/gemma3:safetensor_loader",
        "@litert_archive//tensor/examples/gemma3:safetensors",
    ],
)
