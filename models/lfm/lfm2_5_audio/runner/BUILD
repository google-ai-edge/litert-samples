# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@litert_lm//omni:omni_runner.bzl", "omni_runner_binary")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")

package(default_visibility = ["//models:__subpackages__"])

proto_library(
    name = "lfm2_audio_proto",
    srcs = ["lfm2_audio.proto"],
)

cc_proto_library(
    name = "lfm2_audio_cc_proto",
    deps = [":lfm2_audio_proto"],
)

py_proto_library(
    name = "lfm2_audio_py_proto",
    visibility = ["//models/lfm/lfm2_5_audio:__subpackages__"],
    deps = [":lfm2_audio_proto"],
)

cc_library(
    name = "lfm2_audio_model_config",
    hdrs = ["lfm2_audio_model_config.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@litert_lm//runtime/executor:executor_settings_base",
    ],
)

cc_library(
    name = "io_types",
    hdrs = ["io_types.h"],
    deps = [":lfm2_audio_model_config"],
)

cc_library(
    name = "input_fan_out_stage",
    srcs = ["input_fan_out_stage.cc"],
    hdrs = ["input_fan_out_stage.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/base:fan_out_stage",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:stage",
    ],
)

cc_library(
    name = "audio_stage",
    srcs = ["audio_stage.cc"],
    hdrs = ["audio_stage.h"],
    deps = [
        ":io_types",
        ":lfm2_audio_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_element_type",
        "@litert_archive//litert/cc:litert_layout",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_ranked_tensor_type",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/base:fan_out_stage",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:sequenced",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/utils:audio_resampler",
        "@litert_lm//omni/utils:tensor_utils",
    ],
)

cc_library(
    name = "text_stage",
    srcs = ["text_stage.cc"],
    hdrs = ["text_stage.h"],
    deps = [
        ":io_types",
        ":lfm2_audio_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_element_type",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/base:fan_out_stage",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:sequenced",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/utils:tensor_utils",
        "@litert_lm//support/tokenizer",
    ],
)

cc_library(
    name = "frontend_stage",
    srcs = ["frontend_stage.cc"],
    hdrs = ["frontend_stage.h"],
    deps = [
        ":io_types",
        ":lfm2_audio_model_config",
        ":text_stage",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert_archive//litert/cc:litert_element_type",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:sequenced",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/utils:tensor_utils",
        "@litert_lm//support/tokenizer",
    ],
)

cc_library(
    name = "generation_stage",
    srcs = ["generation_stage.cc"],
    hdrs = ["generation_stage.h"],
    deps = [
        ":io_types",
        ":lfm2_audio_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_element_type",
        "@litert_archive//litert/cc:litert_layout",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_ranked_tensor_type",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/any2any:any2any_sampler",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_lm_runner",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/utils:tensor_utils",
        "@litert_lm//runtime/executor:llm_executor_io_types",
        "@litert_lm//support/tokenizer",
        "@litert_lm//support/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "lfm2_audio_factory",
    srcs = ["lfm2_audio_factory.cc"],
    hdrs = ["lfm2_audio_factory.h"],
    deps = [
        ":audio_stage",
        ":frontend_stage",
        ":generation_stage",
        ":input_fan_out_stage",
        ":lfm2_audio_model_config",
        ":text_stage",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/any2any:any2any_resources",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_lm_runner",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:stage",
        "@litert_lm//runtime/components:model_resources",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//support/tokenizer",
    ],
)

cc_library(
    name = "lfm2_audio",
    srcs = ["lfm2_audio.cc"],
    deps = [
        ":lfm2_audio_cc_proto",
        ":lfm2_audio_factory",
        ":lfm2_audio_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_options",
        "@litert_lm//omni:omni_engine_header",
        "@litert_lm//omni:omni_registry",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/any2any:any2any_session",
        "@litert_lm//omni/base:custom_model_utils",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//runtime/components:model_resources",
        "@litert_lm//runtime/proto:any2any_metadata_cc_proto",
        "@litert_lm//runtime/proto:any2any_model_type_cc_proto",
        "@litert_lm//runtime/proto:any2any_session_config_cc_proto",
    ],
    alwayslink = 1,
)

omni_runner_binary(
    name = "lfm2_audio_runner",
    models = [":lfm2_audio"],
)

cc_library(
    name = "lfm2_audio_test_utils",
    testonly = 1,
    srcs = ["lfm2_audio_test_utils.cc"],
    hdrs = ["lfm2_audio_test_utils.h"],
    deps = [
        ":lfm2_audio_cc_proto",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_lm//omni/any2any:any2any_test_utils",
        "@litert_lm//runtime/proto:any2any_metadata_cc_proto",
        "@litert_lm//runtime/proto:any2any_model_type_cc_proto",
        "@litert_lm//schema/core:litertlm_header_schema",
        "@litert_archive//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "lfm2_audio_test",
    srcs = ["lfm2_audio_test.cc"],
    deps = [
        ":audio_stage",
        ":frontend_stage",
        ":generation_stage",
        ":input_fan_out_stage",
        ":io_types",
        ":lfm2_audio",
        ":lfm2_audio_cc_proto",
        ":lfm2_audio_factory",
        ":lfm2_audio_model_config",
        ":lfm2_audio_test_utils",
        ":text_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_expected",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni:omni_engine",
        "@litert_lm//omni:omni_registry",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/any2any:any2any_session",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_lm_runner",
        "@litert_lm//omni/base:mock_litert_runner",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:sequenced",
        "@litert_lm//runtime/components:model_resources",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//runtime/executor:llm_executor_io_types",
        "@litert_lm//runtime/proto:any2any_metadata_cc_proto",
        "@litert_lm//runtime/proto:any2any_model_type_cc_proto",
        "@litert_lm//runtime/proto:any2any_session_config_cc_proto",
        "@litert_lm//runtime/proto:llm_metadata_cc_proto",
        "@litert_lm//support/tokenizer",
        "@litert_lm//support/util:convert_tensor_buffer",
        "@litert_lm//support/util:test_utils",
    ],
)

cc_test(
    name = "lfm2_audio_runner_test",
    srcs = ["lfm2_audio_runner_test.cc"],
    data = [
        ":lfm2_audio_runner",
    ],
    deps = [
        ":lfm2_audio",
        ":lfm2_audio_cc_proto",
        ":lfm2_audio_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_lm//omni:omni_runner_lib",
        "@litert_lm//runtime/proto:any2any_metadata_cc_proto",
        "@litert_lm//runtime/proto:any2any_model_type_cc_proto",
        "@litert_lm//support/util:test_utils",
    ],
)
