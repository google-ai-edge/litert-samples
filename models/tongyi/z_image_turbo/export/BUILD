# Copyright 2026 The LiteRT Torch Authors.
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

load("@pypi//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_visibility = [
        "//models:__subpackages__",
    ],
)

py_library(
    name = "modeling",
    srcs = ["modeling.py"],
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        "@litert_torch//litert_torch",
        "@litert_torch//litert_torch/generative/export_hf/model_ext/bonsai_flux2:modeling_flux2",
        requirement("torch"),
    ],
)

py_library(
    name = "z_image_turbo_exportable",
    srcs = ["z_image_turbo.py"],
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        ":modeling",
        "//models/tongyi/z_image_turbo:z_image_turbo_py_proto",
        "@litert_lm//runtime/proto:text2image_metadata_py_proto",
        "@litert_torch//litert_torch",
        "@litert_torch//litert_torch/generative/export_hf/core:exportable_module_config",
        "@litert_torch//litert_torch/generative/export_hf/core:quantization",
        "@litert_torch//litert_torch/generative/export_hf/model_ext/bonsai_flux2:bonsai_flux2_exportable",
        "@litert_torch//litert_torch/generative/export_hf/model_ext/bonsai_flux2:modeling_flux2",
        requirement("safetensors"),
        requirement("torch"),
        requirement("transformers"),
    ],
)

py_library(
    name = "extension",
    srcs = ["extension.py"],
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        ":z_image_turbo_exportable",
        "@litert_torch//litert_torch/generative/export_hf/model_ext:extension",
    ],
)

py_library(
    name = "export_main_lib",
    srcs = ["export_main.py"],
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        ":extension",
        requirement("fire"),
        "@litert_torch//litert_torch/generative/export_hf:export",
    ],
)

py_binary(
    name = "export_main",
    srcs = ["export_main.py"],
    main = "export_main.py",
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        ":export_main_lib",
    ],
)

py_test(
    name = "test_z_image_turbo",
    size = "large",
    srcs = ["test_z_image_turbo.py"],
    shard_count = 3,
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        ":export_main_lib",
        ":extension",
        ":modeling",
        ":z_image_turbo_exportable",
        "//models/tongyi/z_image_turbo:z_image_turbo_py_proto",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@litert_archive//litert/python/litert_wrapper/compiled_model_wrapper:compiled_model",
        "@litert_lm//python/litert_lm_builder:litertlm_peek",
        "@litert_lm//runtime/proto:text2image_metadata_py_proto",
        "@com_google_protobuf//:protobuf_python",
        "@litert_torch//litert_torch",
        "@litert_torch//litert_torch/generative/export_hf:export",
        "@litert_torch//litert_torch/generative/export_hf/core:exportable_module_config",
        "@litert_torch//litert_torch/generative/export_hf/core:quantization",
        "@litert_torch//litert_torch/generative/export_hf/model_ext:extension",
        "@litert_torch//litert_torch/generative/export_hf/model_ext/bonsai_flux2:bonsai_flux2_exportable",
        "@litert_torch//litert_torch/generative/export_hf/model_ext/bonsai_flux2:modeling_flux2",
        requirement("numpy"),
        requirement("safetensors"),
        requirement("torch"),
        requirement("transformers"),
        "@litert_archive//tflite/python:schema_py",
    ],
)

py_test(
    name = "export_and_run_test",
    size = "large",
    srcs = ["export_and_run_test.py"],
    data = [
        ":export_main",
        "//models/tongyi/z_image_turbo:z_image_turbo_runner",
    ],
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        "@absl_py//absl/testing:absltest",
        requirement("safetensors"),
        requirement("torch"),
    ],
)
