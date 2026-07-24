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
    name = "codec_config",
    hdrs = ["codec_config.h"],
)

cc_library(
    name = "codec_graph",
    srcs = ["codec_graph.cc"],
    hdrs = ["codec_graph.h"],
    deps = [
        ":codec_config",
        "@litert_archive//tensor",
        "@litert_archive//tensor:arithmetic",
        "@litert_archive//tensor:buffer",
        "@litert_archive//tensor:datatypes",
        "@litert_archive//tensor/backends/tflite:arithmetic_tflite",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "codec_weights",
    srcs = ["codec_weights.cc"],
    hdrs = ["codec_weights.h"],
    deps = [
        ":codec_config",
        ":codec_graph",
        "@litert_archive//tensor",
        "@litert_archive//tensor:buffer",
        "@litert_archive//tensor:datatypes",
        "@litert_archive//tensor/examples/gemma3:safetensor_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "codec_main",
    srcs = ["codec_main.cc"],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"] + select({
        "@litert_archive//litert:android": ["-DABSL_FLAGS_STRIP_NAMES=0"],
        "//conditions:default": [],
    }),
    data = litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    deps = [
        ":codec_config",
        ":codec_graph",
        ":codec_weights",
        "@litert_archive//litert/cc:litert_common",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_options",
        "@litert_archive//litert/cc/options:litert_gpu_options",
        "@litert_archive//tensor",
        "@litert_archive//tensor:datatypes",
        "@litert_archive//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "@litert_archive//tensor/runners/litert:litert_dynamic_runner",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
    ],
)
