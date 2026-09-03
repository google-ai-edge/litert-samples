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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "@litert_archive//litert/build_common:special_rule.bzl",
    "litert_android_linkopts",
    "litert_gpu_accelerator_prebuilts",
)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sam2v_graph",
    srcs = ["sam2v_graph.cc"],
    hdrs = ["sam2v_graph.h"],
    deps = [
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_config",
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_graph",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@litert_archive//tensor",
        "@litert_archive//tensor:arithmetic",
        "@litert_archive//tensor:buffer",
        "@litert_archive//tensor:datatypes",
        "@litert_archive//tensor/backends/tflite:arithmetic_tflite",
    ],
)

cc_library(
    name = "sam2v_weights",
    srcs = ["sam2v_weights.cc"],
    hdrs = ["sam2v_weights.h"],
    deps = [
        ":sam2v_graph",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert_archive//tensor",
        "@litert_archive//tensor/examples/gemma3:safetensor_loader",
    ],
)

cc_binary(
    name = "sam2v_main",
    srcs = ["sam2v_main.cc"],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"] + select({
        "@litert_archive//litert:android": ["-DABSL_FLAGS_STRIP_NAMES=0"],
        "//conditions:default": [],
    }),
    data = litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    deps = [
        ":sam2v_graph",
        ":sam2v_weights",
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_config",
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_graph",
        "//models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_weights",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@litert_archive//litert/cc:litert_common",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_options",
        "@litert_archive//litert/cc/options:litert_gpu_options",
        "@litert_archive//tensor",
        "@litert_archive//tensor:arithmetic",
        "@litert_archive//tensor:buffer",
        "@litert_archive//tensor:datatypes",
        "@litert_archive//tensor/backends/tflite:arithmetic_tflite",
        "@litert_archive//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "@litert_archive//tensor/runners/litert:litert_dynamic_runner",
    ],
)
