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
    name = "janus_pro_proto",
    srcs = ["janus_pro.proto"],
)

cc_proto_library(
    name = "janus_pro_cc_proto",
    deps = [":janus_pro_proto"],
)

py_proto_library(
    name = "janus_pro_py_proto",
    visibility = ["//models/deepseek/janus_pro:__subpackages__"],
    deps = [":janus_pro_proto"],
)

cc_library(
    name = "janus_pro_model_config",
    hdrs = ["janus_pro_model_config.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@litert_lm//runtime/executor:executor_settings_base",
    ],
)

cc_library(
    name = "io_types",
    hdrs = ["io_types.h"],
    deps = [],
)

cc_library(
    name = "stage_utils",
    srcs = ["stage_utils.cc"],
    hdrs = ["stage_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert_archive//litert/cc:litert_element_type",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_ranked_tensor_type",
        "@litert_archive//litert/cc:litert_tensor_buffer",
    ],
)

cc_library(
    name = "fan_out_stage",
    srcs = ["fan_out_stage.cc"],
    hdrs = ["fan_out_stage.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/base:fan_out_stage",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:stage",
    ],
)

cc_test(
    name = "fan_out_stage_test",
    srcs = ["fan_out_stage_test.cc"],
    deps = [
        ":fan_out_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@litert_lm//omni:omni_session",
        "@litert_lm//support/util:test_utils",
    ],
)

cc_library(
    name = "text_stage",
    srcs = ["text_stage.cc"],
    hdrs = ["text_stage.h"],
    deps = [
        ":io_types",
        ":janus_pro_model_config",
        ":stage_utils",
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
        "@litert_archive//litert/cc:litert_ranked_tensor_type",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/base:fan_out_stage",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:sequenced",
        "@litert_lm//omni/base:stage",
        "@litert_lm//support/tokenizer",
    ],
)

cc_test(
    name = "text_stage_test",
    srcs = ["text_stage_test.cc"],
    deps = [
        ":fan_out_stage",
        ":io_types",
        ":janus_pro_model_config",
        ":janus_pro_test_utils",
        ":text_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:mock_litert_runner",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:sequenced",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//support/tokenizer",
        "@litert_lm//support/util:convert_tensor_buffer",
        "@litert_lm//support/util:test_utils",
        "@litert_archive//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "image_stage",
    srcs = ["image_stage.cc"],
    hdrs = ["image_stage.h"],
    deps = [
        ":io_types",
        ":janus_pro_model_config",
        ":stage_utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
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
        "@litert_lm//runtime/components/preprocessor:image_preprocessor",
        "@litert_lm//runtime/components/preprocessor:image_preprocessor_impl",
    ],
)

cc_test(
    name = "image_stage_test",
    srcs = ["image_stage_test.cc"],
    deps = [
        ":fan_out_stage",
        ":image_stage",
        ":janus_pro_model_config",
        ":janus_pro_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:mock_litert_runner",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//support/util:convert_tensor_buffer",
        "@litert_lm//support/util:test_utils",
        "@litert_archive//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "frontend_stage",
    srcs = ["frontend_stage.cc"],
    hdrs = ["frontend_stage.h"],
    deps = [
        ":io_types",
        ":janus_pro_model_config",
        ":text_stage",
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
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/base:fan_in_stage",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:sequenced",
        "@litert_lm//omni/base:stage",
        "@litert_lm//support/tokenizer",
    ],
)

cc_test(
    name = "frontend_stage_test",
    srcs = ["frontend_stage_test.cc"],
    deps = [
        ":fan_out_stage",
        ":frontend_stage",
        ":image_stage",
        ":io_types",
        ":janus_pro_model_config",
        ":text_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/base:mock_litert_runner",
        "@litert_lm//support/tokenizer",
        "@litert_lm//support/util:convert_tensor_buffer",
        "@litert_lm//support/util:test_utils",
    ],
)

cc_library(
    name = "llm_stage",
    srcs = ["llm_stage.cc"],
    hdrs = ["llm_stage.h"],
    deps = [
        ":io_types",
        ":janus_pro_model_config",
        ":stage_utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_element_type",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_ranked_tensor_type",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/any2any:any2any_sampler",
        "@litert_lm//omni/base:litert_lm_runner",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//runtime/executor:llm_executor_io_types",
        "@litert_lm//support/tokenizer",
        "@litert_lm//support/util:convert_tensor_buffer",
        "@litert_archive//tflite/types:half",
    ],
)

cc_test(
    name = "llm_stage_test",
    srcs = ["llm_stage_test.cc"],
    deps = [
        ":io_types",
        ":janus_pro_model_config",
        ":janus_pro_test_utils",
        ":llm_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_expected",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:mock_litert_lm_runner",
        "@litert_lm//omni/base:mock_litert_runner",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//runtime/executor:llm_executor_io_types",
        "@litert_lm//support/tokenizer",
        "@litert_lm//support/util:convert_tensor_buffer",
        "@litert_lm//support/util:test_utils",
        "@litert_archive//tflite/schema:schema_fbs",
        "@litert_archive//tflite/types:half",
    ],
)

cc_library(
    name = "image_gen_stage",
    srcs = ["image_gen_stage.cc"],
    hdrs = ["image_gen_stage.h"],
    deps = [
        ":io_types",
        ":janus_pro_model_config",
        ":stage_utils",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_element_type",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:stage",
    ],
)

cc_test(
    name = "image_gen_stage_test",
    srcs = ["image_gen_stage_test.cc"],
    deps = [
        ":image_gen_stage",
        ":io_types",
        ":janus_pro_model_config",
        ":janus_pro_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_expected",
        "@litert_archive//litert/cc:litert_macros",
        "@litert_archive//litert/cc:litert_tensor_buffer",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:mock_litert_runner",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//support/util:convert_tensor_buffer",
        "@litert_lm//support/util:test_utils",
        "@litert_archive//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "janus_pro_factory",
    srcs = ["janus_pro_factory.cc"],
    hdrs = ["janus_pro_factory.h"],
    deps = [
        ":fan_out_stage",
        ":frontend_stage",
        ":image_gen_stage",
        ":image_stage",
        ":janus_pro_model_config",
        ":llm_stage",
        ":text_stage",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert_archive//litert/cc:litert_compiled_model",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/any2any:any2any_resources",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_lm_runner",
        "@litert_lm//omni/base:litert_runner",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//runtime/components:model_resources",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//runtime/proto:any2any_metadata_cc_proto",
        "@litert_lm//support/tokenizer",
    ],
)

cc_library(
    name = "janus_pro",
    srcs = ["janus_pro.cc"],
    deps = [
        ":janus_pro_cc_proto",
        ":janus_pro_factory",
        ":janus_pro_model_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_options",
        "@litert_lm//omni:multi_staged_session",
        "@litert_lm//omni:omni_engine_header",
        "@litert_lm//omni:omni_registry",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/any2any:any2any_session",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//omni/base:stage",
        "@litert_lm//runtime/components:model_resources",
        "@litert_lm//runtime/framework:threadpool",
        "@litert_lm//runtime/proto:any2any_metadata_cc_proto",
        "@litert_lm//runtime/proto:any2any_model_type_cc_proto",
        "@litert_lm//runtime/proto:any2any_session_config_cc_proto",
    ],
    alwayslink = 1,
)

omni_runner_binary(
    name = "janus_pro_runner",
    models = [":janus_pro"],
)

cc_library(
    name = "janus_pro_test_utils",
    testonly = 1,
    srcs = ["janus_pro_test_utils.cc"],
    hdrs = ["janus_pro_test_utils.h"],
    deps = [
        ":janus_pro_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@litert_lm//runtime/proto:any2any_metadata_cc_proto",
        "@litert_lm//runtime/proto:any2any_model_type_cc_proto",
        "@litert_lm//schema/core:litertlm_header",
        "@litert_lm//schema/core:litertlm_header_schema",
        "@litert_lm//schema/core:litertlm_section",
        "@stb//:stb_image_write",
        "@litert_archive//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "janus_pro_test",
    srcs = ["janus_pro_test.cc"],
    data = [
        "@litert_lm//runtime/testdata:checksum_fake_lm_models",
    ],
    deps = [
        ":janus_pro",
        ":janus_pro_cc_proto",
        ":janus_pro_factory",
        ":janus_pro_model_config",
        ":janus_pro_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@litert_archive//litert/cc:litert_environment",
        "@litert_archive//litert/cc:litert_expected",
        "@litert_lm//omni:omni_engine",
        "@litert_lm//omni:omni_registry",
        "@litert_lm//omni:omni_session",
        "@litert_lm//omni/any2any:any2any_session",
        "@litert_lm//omni/base:io_types",
        "@litert_lm//omni/base:litert_lm_runner",
        "@litert_lm//omni/base:model_resources",
        "@litert_lm//omni/base:model_utils",
        "@litert_lm//runtime/components:model_resources",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//runtime/proto:any2any_metadata_cc_proto",
        "@litert_lm//runtime/proto:any2any_model_type_cc_proto",
        "@litert_lm//runtime/proto:any2any_session_config_cc_proto",
        "@litert_lm//runtime/proto:llm_metadata_cc_proto",
        "@litert_lm//support/util:test_utils",
    ],
)

cc_test(
    name = "janus_pro_runner_test",
    srcs = ["janus_pro_runner_test.cc"],
    data = [
        ":janus_pro_runner",
        "@litert_lm//runtime/testdata:checksum_fake_lm_models",
    ],
    deps = [
        ":janus_pro",
        ":janus_pro_test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert_lm//omni:omni_runner_lib",
        "@litert_lm//runtime/components/preprocessor:image_preprocessor",
        "@litert_lm//runtime/components/preprocessor:image_preprocessor_impl",
        "@litert_lm//support/util:test_utils",
    ],
)
