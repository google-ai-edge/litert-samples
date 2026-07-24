// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// LLaDA-8B-Base checkpoint loading. See diffusion_weights.h.

#include "models/llada/llada_8b/tensor_api/diffusion_weights.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::diffusion {

namespace {

struct WeightSpec {
  std::string graph_name;   // name BuildDenoiseStep looks up
  std::string source_name;  // name in the checkpoint
  std::vector<int> shape;   // expected shape after any vocab slice
  int full_rows = 0;        // checkpoint dim-0 when slicing (0 = no slice)
};

std::vector<WeightSpec> GetWeightSpecs(const DiffusionConfig& config,
                                       int checkpoint_vocab) {
  const int emb = config.emb_dim;
  const int qkv = config.qkv_out_dim();
  const int kv = config.kv_out_dim();
  const int ffn = config.hidden_dim;
  const bool slice = config.vocab < checkpoint_vocab;

  std::vector<WeightSpec> specs;
  specs.push_back({"model.embed_tokens.weight", "model.transformer.wte.weight",
                   {config.vocab, emb}, slice ? checkpoint_vocab : 0});
  for (int i = 0; i < config.n_layers; ++i) {
    const std::string g = absl::StrCat("model.layers.", i);
    const std::string s = absl::StrCat("model.transformer.blocks.", i);
    specs.push_back({g + ".input_layernorm.weight", s + ".attn_norm.weight",
                     {emb}});
    specs.push_back({g + ".self_attn.q_proj.weight", s + ".q_proj.weight",
                     {qkv, emb}});
    specs.push_back({g + ".self_attn.k_proj.weight", s + ".k_proj.weight",
                     {kv, emb}});
    specs.push_back({g + ".self_attn.v_proj.weight", s + ".v_proj.weight",
                     {kv, emb}});
    specs.push_back({g + ".self_attn.o_proj.weight", s + ".attn_out.weight",
                     {emb, qkv}});
    specs.push_back({g + ".post_attention_layernorm.weight",
                     s + ".ff_norm.weight", {emb}});
    specs.push_back({g + ".mlp.gate_proj.weight", s + ".ff_proj.weight",
                     {ffn, emb}});
    specs.push_back({g + ".mlp.up_proj.weight", s + ".up_proj.weight",
                     {ffn, emb}});
    specs.push_back({g + ".mlp.down_proj.weight", s + ".ff_out.weight",
                     {emb, ffn}});
  }
  specs.push_back({"model.norm.weight", "model.transformer.ln_f.weight",
                   {emb}});
  // Untied head: the MODEL-level ff_out (weight_tying=false in config.json).
  specs.push_back({"lm_head.weight", "model.transformer.ff_out.weight",
                   {config.vocab, emb}, slice ? checkpoint_vocab : 0});
  return specs;
}

// The gemma3 loader adds +1.0 to anything matching its is_layernorm
// predicate (contains "layernorm" or ends with "norm.weight"). LLaDA's RMS
// norms carry plain affine weights, so undo it. Source names attn_norm/
// ff_norm match the predicate; ln_f does not.
bool LoaderAddsGemmaNormOffset(const std::string& name) {
  return absl::StrContains(name, "layernorm") ||
         absl::EndsWith(name, "norm.weight");
}

// Blockwise-32 int4 RTN with fp16-rounded scales — same layout as
// talker_weights.cc so gptq_quantize.py companions stay compatible later.
constexpr int kInt4BlockSize = 32;

bool IsQuantizableFcWeight(const std::string& graph_name) {
  return absl::StrContains(graph_name, "_proj.weight") ||
         graph_name == "lm_head.weight";
}

absl::StatusOr<TfTensor> QuantizeFcWeightInt4(const TfTensor& tensor,
                                              const std::string& name) {
  const auto& shape = tensor.GetShape();
  if (shape.size() != 2) {
    return absl::FailedPreconditionError(
        absl::StrCat(name, ": expected 2-D weight for quantization"));
  }
  const int out_channels = shape[0];
  const int in_dim = shape[1];
  if (in_dim % kInt4BlockSize != 0) {
    return absl::FailedPreconditionError(
        absl::StrCat(name, ": in_dim not divisible by block"));
  }
  auto buffer = tensor.GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* w = reinterpret_cast<const float*>(lock.data());

  const int blocks_per_row = in_dim / kInt4BlockSize;
  std::vector<float> scales(static_cast<size_t>(out_channels) *
                            blocks_per_row);
  const size_t count = static_cast<size_t>(out_channels) * in_dim;
  std::vector<char> packed((count + 1) / 2, 0);
  for (int c = 0; c < out_channels; ++c) {
    const float* row = w + static_cast<size_t>(c) * in_dim;
    for (int b = 0; b < blocks_per_row; ++b) {
      float amax = 0.0f;
      for (int i = b * kInt4BlockSize; i < (b + 1) * kInt4BlockSize; ++i)
        amax = std::max(amax, std::fabs(row[i]));
      float scale = amax > 0.0f ? amax / 7.0f : 1.0f;
      scale = static_cast<float>(fp16_t(scale));
      scales[static_cast<size_t>(c) * blocks_per_row + b] = scale;
      for (int i = b * kInt4BlockSize; i < (b + 1) * kInt4BlockSize; ++i) {
        float v = std::round(row[i] / scale);
        int8_t q = static_cast<int8_t>(std::min(7.0f, std::max(-7.0f, v)));
        const size_t flat = static_cast<size_t>(c) * in_dim + i;
        char nib = static_cast<char>(q & 0xF);
        packed[flat / 2] |= (flat % 2 == 0) ? nib : static_cast<char>(nib << 4);
      }
    }
  }

  auto q_buffer = OwningCpuBuffer::Copy(packed.data(), packed.size());
  TfTensor quantized({.name = name,
                      .type = Type::kI4,
                      .shape = {out_channels, in_dim},
                      .buffer = std::move(q_buffer)});
  quantized.SetQuantization(std::make_shared<BlockwiseQuantization>(
      std::move(scales), std::vector<int64_t>(), kInt4BlockSize,
      /*quantized_dimension=*/0));
  return quantized;
}

absl::StatusOr<TfTensor> QuantizeFcWeightInt8(const TfTensor& tensor,
                                              const std::string& name) {
  const auto& shape = tensor.GetShape();
  if (shape.size() != 2) {
    return absl::FailedPreconditionError(
        absl::StrCat(name, ": expected 2-D weight for quantization"));
  }
  const int out_channels = shape[0];
  const int in_dim = shape[1];
  auto buffer = tensor.GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* w = reinterpret_cast<const float*>(lock.data());

  std::vector<float> scales(out_channels);
  std::vector<int8_t> q(static_cast<size_t>(out_channels) * in_dim);
  for (int c = 0; c < out_channels; ++c) {
    const float* row = w + static_cast<size_t>(c) * in_dim;
    float amax = 0.0f;
    for (int i = 0; i < in_dim; ++i) amax = std::max(amax, std::fabs(row[i]));
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    scales[c] = scale;
    int8_t* q_row = q.data() + static_cast<size_t>(c) * in_dim;
    for (int i = 0; i < in_dim; ++i) {
      float v = std::round(row[i] / scale);
      q_row[i] = static_cast<int8_t>(std::min(127.0f, std::max(-127.0f, v)));
    }
  }
  auto q_buffer = OwningCpuBuffer::Copy(
      reinterpret_cast<const char*>(q.data()), q.size());
  TfTensor quantized({.name = name,
                      .type = Type::kI8,
                      .shape = {out_channels, in_dim},
                      .buffer = std::move(q_buffer)});
  quantized.SetQuantization(std::make_shared<PerChannelAffineQuantization>(
      std::move(scales), std::vector<int64_t>(out_channels, 0),
      /*quantized_dimension=*/0));
  return quantized;
}

absl::Status SubtractOne(TfTensor& tensor) {
  auto buffer = tensor.GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->LockMutable();
  float* data = reinterpret_cast<float*>(lock.data());
  size_t count = lock.size() / sizeof(float);
  for (size_t i = 0; i < count; ++i) data[i] -= 1.0f;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<WeightMap> LoadLladaWeights(const DiffusionConfig& config,
                                           const std::string& checkpoint_dir,
                                           WeightQuantMode quant_mode) {
  std::vector<std::string> shard_paths;
  for (const auto& entry :
       std::filesystem::directory_iterator(checkpoint_dir)) {
    const std::string p = entry.path().string();
    if (absl::EndsWith(p, ".safetensors")) shard_paths.push_back(p);
  }
  std::sort(shard_paths.begin(), shard_paths.end());
  if (shard_paths.empty()) {
    return absl::NotFoundError(
        absl::StrCat("no .safetensors shards in ", checkpoint_dir));
  }
  std::vector<SafetensorLoader> shards;
  for (const auto& p : shard_paths) {
    auto loader_or = SafetensorLoader::Load(p);
    if (!loader_or.ok()) return loader_or.status();
    shards.push_back(std::move(*loader_or));
  }

  // Checkpoint vocab (untied head rows) for the optional slice.
  int checkpoint_vocab = 0;
  for (const auto& shard : shards) {
    auto info_or = shard.GetTensorInfo("model.transformer.ff_out.weight");
    if (info_or.ok()) {
      checkpoint_vocab = static_cast<int>(info_or->shape[0]);
      break;
    }
  }
  if (checkpoint_vocab == 0) {
    return absl::NotFoundError(
        "model.transformer.ff_out.weight (untied head) not found in shards");
  }

  WeightMap weights;
  for (const WeightSpec& spec : GetWeightSpecs(config, checkpoint_vocab)) {
    const SafetensorLoader* shard = nullptr;
    for (const auto& candidate : shards) {
      if (candidate.GetTensorInfo(spec.source_name).ok()) {
        shard = &candidate;
        break;
      }
    }
    if (shard == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("checkpoint missing ", spec.source_name));
    }
    auto handle_or = shard->LoadTensor(
        spec.source_name, SafetensorLoader::QuantizedLoadMode::kDequantizeToFp32);
    if (!handle_or.ok()) return handle_or.status();
    TfTensor tensor(*handle_or);

    if (spec.full_rows > 0) {
      // Vocab slice: keep the first shape[0] rows of [full_rows, emb].
      const int rows = spec.shape[0];
      const int cols = spec.shape[1];
      auto buffer = tensor.GetBuffer();
      if (!buffer.ok()) return buffer.status();
      auto lock = buffer->Lock();
      const float* src = reinterpret_cast<const float*>(lock.data());
      const size_t count = static_cast<size_t>(rows) * cols;
      std::vector<float> sliced(src, src + count);
      tensor = TfTensor({.name = spec.graph_name,
                         .type = Type::kFP32,
                         .shape = spec.shape,
                         .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                             std::move(sliced))});
    }

    const auto& loaded_shape = tensor.GetShape();
    std::vector<int> loaded(loaded_shape.begin(), loaded_shape.end());
    if (loaded != spec.shape) {
      return absl::FailedPreconditionError(absl::StrCat(
          spec.source_name, ": checkpoint shape [",
          absl::StrJoin(loaded, ","), "] != expected [",
          absl::StrJoin(spec.shape, ","), "]"));
    }

    if (LoaderAddsGemmaNormOffset(spec.source_name)) {
      auto status = SubtractOne(tensor);
      if (!status.ok()) return status;
    }

    if (quant_mode != WeightQuantMode::kNone &&
        IsQuantizableFcWeight(spec.graph_name)) {
      auto quantized_or = quant_mode == WeightQuantMode::kInt4
                              ? QuantizeFcWeightInt4(tensor, spec.graph_name)
                              : QuantizeFcWeightInt8(tensor, spec.graph_name);
      if (!quantized_or.ok()) return quantized_or.status();
      weights[spec.graph_name] = std::move(*quantized_or);
      continue;
    }

    tensor.SetName(spec.graph_name);
    weights[spec.graph_name] = std::move(tensor);
  }
  return weights;
}

}  // namespace litert::tensor::examples::diffusion
