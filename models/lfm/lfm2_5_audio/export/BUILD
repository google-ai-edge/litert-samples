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
    name = "lfm2_audio_exportable",
    srcs = [
        "lfm2_audio.py",
        "modules.py",
    ],
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        "//models/lfm/lfm2_5_audio:lfm2_audio_py_proto",
        "@litert_lm//runtime/proto:any2any_metadata_py_proto",
        requirement("huggingface-hub"),
        "@litert_torch//litert_torch",
        "@litert_torch//litert_torch:_convert",
        "@litert_torch//litert_torch/generative/export_hf/core:attention",
        "@litert_torch//litert_torch/generative/export_hf/core:cache",
        "@litert_torch//litert_torch/generative/export_hf/core:exportable_module_config",
        "@litert_torch//litert_torch/generative/export_hf/core:quantization",
        "@litert_torch//litert_torch/generative/export_hf/core/external_emb:exportable_module",
        "@litert_torch//litert_torch/generative/export_hf/model_ext:patches",
        "@litert_torch//litert_torch/generative/export_hf/model_ext/lfm2:patch",
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
        ":lfm2_audio_exportable",
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
    name = "test_lfm2_audio",
    size = "medium",
    srcs = ["test_lfm2_audio.py"],
    shard_count = 3,
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        ":export_main_lib",
        ":extension",
        ":lfm2_audio_exportable",
        "//models/lfm/lfm2_5_audio:lfm2_audio_py_proto",
        "@litert_archive//litert/python/litert_wrapper/compiled_model_wrapper:compiled_model",
        "@litert_lm//python/litert_lm_builder:litertlm_peek",
        "@litert_lm//runtime/proto:any2any_metadata_py_proto",
        "@absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
        "@litert_torch//litert_torch/generative/export_hf:export",
        "@litert_torch//litert_torch/generative/export_hf/core:exportable_module_config",
        "@litert_torch//litert_torch/generative/export_hf/model_ext:exportables",
        "@litert_torch//litert_torch/generative/export_hf/model_ext:extension",
        "@litert_torch//litert_torch/generative/export_hf/model_ext:patches",
        requirement("numpy"),
        requirement("safetensors"),
        requirement("torch"),
        requirement("transformers"),
    ],
)

py_test(
    name = "export_and_run_test",
    size = "large",
    srcs = ["export_and_run_test.py"],
    data = [
        ":export_main",
        "//models/lfm/lfm2_5_audio:lfm2_audio_runner",
    ],
    # PyTorch / litert_torch transitively depends on Linux-only CUDA targets.
    tags = ["nomacos"],
    deps = [
        "@absl_py//absl/testing:absltest",
        requirement("safetensors"),
        requirement("torch"),
    ],
)
