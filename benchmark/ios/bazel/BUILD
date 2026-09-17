# Copyright 2026 The AI Edge LiteRT Authors.
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

# iOS static framework around litert/tools:benchmark_model (BenchmarkLiteRtModel), so an app can run
# the same benchmark the Android and desktop benchmark_model binaries run.
# bazel build --config=ios_arm64 -c opt --copt=-DBENCHMARK_MODEL_NO_MAIN //litert/tools/ios_benchmark:LiteRtBenchmark_framework

load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "litert_benchmark_shim",
    srcs = ["litert_benchmark_shim.cc"],
    hdrs = ["litert_benchmark_shim.h"],
    deps = ["//litert/tools:benchmark_model_main"],
    alwayslink = 1,
)

ios_static_framework(
    name = "LiteRtBenchmark_framework",
    hdrs = ["litert_benchmark_shim.h"],
    bundle_name = "LiteRtBenchmark",
    minimum_os_version = "15.0",
    deps = [":litert_benchmark_shim"],
)
