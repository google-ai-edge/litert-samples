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
    name = "z_image_turbo_proto",
    srcs = ["z_image_turbo.proto"],
)

cc_proto_library(
    name = "z_image_turbo_cc_proto",
    deps = [":z_image_turbo_proto"],
)

py_proto_library(
    name = "z_image_turbo_py_proto",
    visibility = ["//models/tongyi/z_image_turbo:__subpackages__"],
    deps = [":z_image_turbo_proto"],
)

cc_library(
    name = "z_image_turbo_model_config",
    srcs = ["z_image_turbo_model_config.cc"],
    hdrs = ["z_image_turbo_model_config.h"],
    deps = [
        ":z_image_turbo_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "denoiser_stage",
    srcs = ["denoiser_stage.cc"],
    hdrs = ["denoiser_stage.h"],
    deps = [
        ":z_image_turbo_model_config",
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
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/text2image:io_types",
        "@litert_lm//omni/text2image:text2image_math",
        "@litert_lm//omni/text2image:text_encoder_stage",
    ],
)

cc_library(
    name = "z_image_turbo_factory",
    srcs = ["z_image_turbo_factory.cc"],
    hdrs = ["z_image_turbo_factory.h"],
    deps = [
        ":denoiser_stage",
        ":z_image_turbo_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:model_cache",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/text2image:io_types",
        "@litert_lm//omni/text2image:prompt_source",
        "@litert_lm//omni/text2image:text_encoder_stage",
        "@litert_lm//omni/text2image:vae_decoder_stage",
        "@litert_lm//runtime/components:model_resources",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//runtime/proto:embedding_metadata_cc_proto",
        "@litert_lm//runtime/proto:text2image_metadata_cc_proto",
        "@litert_lm//support/tokenizer",
    ],
)

cc_library(
    name = "z_image_turbo",
    srcs = ["z_image_turbo.cc"],
    deps = [
        ":z_image_turbo_cc_proto",
        ":z_image_turbo_factory",
        ":z_image_turbo_model_config",
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
        "@litert_lm//omni/base:custom_model_utils",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/text2image:prompt_source",
        "@litert_lm//omni/text2image:text2image_session",
        "@litert_lm//runtime/components:model_resources",
        "@litert_lm//runtime/proto:image_gen_model_type_cc_proto",
        "@litert_lm//runtime/proto:text2image_metadata_cc_proto",
        "@litert_lm//runtime/proto:text2image_session_config_cc_proto",
    ],
    alwayslink = 1,
)

omni_runner_binary(
    name = "z_image_turbo_runner",
    models = [":z_image_turbo"],
)

cc_library(
    name = "z_image_turbo_test_utils",
    testonly = 1,
    srcs = ["z_image_turbo_test_utils.cc"],
    hdrs = ["z_image_turbo_test_utils.h"],
    deps = [
        ":z_image_turbo_cc_proto",
        ":z_image_turbo_model_config",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@litert_archive//litert/cc:litert_element_type",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_layout",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_ranked_tensor_type",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_archive//litert/cc:litert_tensor_buffer_types",
        "@litert_lm//omni/any2any:any2any_test_utils",
        "@litert_lm//omni/base:custom_model_utils",
        "@litert_lm//runtime/proto:image_gen_model_type_cc_proto",
        "@litert_lm//runtime/proto:text2image_metadata_cc_proto",
        "@litert_lm//schema/core:litertlm_header",
        "@litert_lm//schema/core:litertlm_header_schema",
        "@litert_lm//schema/core:litertlm_section",
        "@litert_archive//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "z_image_turbo_model_config_test",
    srcs = ["z_image_turbo_model_config_test.cc"],
    deps = [
        ":z_image_turbo_cc_proto",
        ":z_image_turbo_model_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@litert_lm//support/util:test_utils",
    ],
)

cc_test(
    name = "denoiser_stage_test",
    srcs = ["denoiser_stage_test.cc"],
    deps = [
        ":denoiser_stage",
        ":z_image_turbo_model_config",
        ":z_image_turbo_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_environment_options",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/any2any:any2any_test_utils",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:model_cache",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/text2image:io_types",
        "@litert_lm//omni/text2image:text2image_math",
        "@litert_lm//omni/text2image:text_encoder_stage",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//support/util:test_utils",
        "@litert_archive//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "z_image_turbo_test",
    srcs = ["z_image_turbo_test.cc"],
    deps = [
        ":z_image_turbo",
        ":z_image_turbo_factory",
        ":z_image_turbo_model_config",
        ":z_image_turbo_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_environment_options",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni:omni_engine",
        "@litert_lm//omni:omni_registry",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/any2any:any2any_test_utils",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:model_cache",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//omni/text2image:io_types",
        "@litert_lm//omni/text2image:prompt_source",
        "@litert_lm//omni/text2image:vae_decoder_stage",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//runtime/proto:image_gen_model_type_cc_proto",
        "@litert_lm//runtime/proto:text2image_metadata_cc_proto",
        "@litert_lm//runtime/proto:text2image_session_config_cc_proto",
        "@litert_lm//support/util:test_utils",
        "@litert_archive//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "z_image_turbo_runner_test",
    srcs = ["z_image_turbo_runner_test.cc"],
    data = [
        ":z_image_turbo_runner",
    ],
    deps = [
        ":z_image_turbo",
        ":z_image_turbo_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert_lm//omni:omni_runner_lib",
        "@litert_lm//support/util:test_utils",
    ],
)
