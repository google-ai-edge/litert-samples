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

// Real-checkpoint weight loading. See talker_weights.h.

#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_config.h"
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_graph.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::talker {

namespace {

// Mirrors the loader's is_layernorm predicate (safetensor_loader.cc), which
// adds the Gemma-style +1.0 these Qwen3 norms must not have.
bool LoaderAddsGemmaNormOffset(const std::string& name) {
  return absl::StrContains(name, "layernorm") ||
         absl::EndsWith(name, "norm.weight");
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

// 2-D [out, in] matmul weights get quantized; norms and embedding tables
// (Gather sources) stay fp32.
bool IsQuantizableFcWeight(const std::string& name) {
  return absl::StrContains(name, "_proj.weight") ||
         absl::EndsWith(name, "codec_head.weight") ||
         absl::StrContains(name, ".lm_head.");
}

// Symmetric weight quantization. int8: per-channel along dim 0
// (PerChannelAffineQuantization). int4: BLOCKWISE along the input dim
// (block kInt4BlockSize, fp16 scales row-major [out, in/block], bit-packed
// kI4 storage) — per-channel int4 measurably destroys this model (frame-0
// logits decorrelate to cos~0), matching why production int4 uses
// calibrated/blockwise schemes.
constexpr int kInt4BlockSize = 32;

// Bit-packs int4 codes (values in [-7,7], one int8 each) into the TFLite
// kI4 layout (even index in the low nibble) and attaches blockwise-32
// quantization. Shared by the in-process RTN path and the pre-quantized
// (GPTQ) companion-file path.
TfTensor BuildInt4Tensor(const int8_t* q, std::vector<float> scales,
                         int out_channels, int in_dim,
                         const std::string& name) {
  const size_t count =
      static_cast<size_t>(out_channels) * static_cast<size_t>(in_dim);
  std::vector<char> packed((count + 1) / 2, 0);
  for (size_t i = 0; i < count; ++i) {
    char nib = static_cast<char>(q[i] & 0xF);
    packed[i / 2] |= (i % 2 == 0) ? nib : static_cast<char>(nib << 4);
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

absl::StatusOr<TfTensor> QuantizeFcWeight(const TfTensor& tensor,
                                          const std::string& name,
                                          WeightQuantMode mode) {
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

  const bool int4 = mode == WeightQuantMode::kInt4;
  const float qmax = int4 ? 7.0f : 127.0f;
  const int block = int4 ? kInt4BlockSize : in_dim;
  if (in_dim % block != 0) {
    return absl::FailedPreconditionError(
        absl::StrCat(name, ": in_dim ", in_dim, " not divisible by block ",
                     block));
  }
  const int blocks_per_row = in_dim / block;

  std::vector<float> scales(static_cast<size_t>(out_channels) *
                            blocks_per_row);
  std::vector<int8_t> q(static_cast<size_t>(out_channels) * in_dim);
  for (int c = 0; c < out_channels; ++c) {
    const float* row = w + static_cast<size_t>(c) * in_dim;
    int8_t* q_row = q.data() + static_cast<size_t>(c) * in_dim;
    for (int b = 0; b < blocks_per_row; ++b) {
      float amax = 0.0f;
      for (int i = b * block; i < (b + 1) * block; ++i)
        amax = std::max(amax, std::fabs(row[i]));
      float scale = amax > 0.0f ? amax / qmax : 1.0f;
      if (int4) {
        // Blockwise scales are serialized as fp16; quantize against the
        // fp16-rounded value the runtime will actually use.
        scale = static_cast<float>(fp16_t(scale));
      }
      scales[static_cast<size_t>(c) * blocks_per_row + b] = scale;
      for (int i = b * block; i < (b + 1) * block; ++i) {
        float v = std::round(row[i] / scale);
        q_row[i] = static_cast<int8_t>(std::min(qmax, std::max(-qmax, v)));
      }
    }
  }

  if (int4) return BuildInt4Tensor(q.data(), std::move(scales),
                                   out_channels, in_dim, name);

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

// Loads one pre-quantized int4 weight from a companion safetensors
// (gemma3 side-tensor convention: `<name>` I8 [out,in] codes in [-7,7],
// `<name>.scale` F32 [out, in/block] fp16-rounded blockwise scales) —
// e.g. the GPTQ output of verify/gptq_quantize.py. Bit-packs into the
// same kI4 + BlockwiseQuantization layout as the in-process RTN path.
absl::StatusOr<TfTensor> LoadPrequantInt4Weight(const SafetensorLoader& ploader,
                                                const std::string& name,
                                                int out_channels, int in_dim) {
  if (in_dim % kInt4BlockSize != 0) {
    return absl::FailedPreconditionError(
        absl::StrCat(name, ": in_dim not divisible by block"));
  }
  const int blocks_per_row = in_dim / kInt4BlockSize;

  auto qinfo_or = ploader.GetTensorInfo(name);
  if (!qinfo_or.ok()) return qinfo_or.status();
  const auto& qinfo = *qinfo_or;
  const size_t q_count =
      static_cast<size_t>(out_channels) * static_cast<size_t>(in_dim);
  if (qinfo.dtype != safetensors::kINT8 ||
      qinfo.data_end - qinfo.data_start != q_count) {
    return absl::FailedPreconditionError(absl::StrCat(
        name, ": prequant body must be I8 [", out_channels, ",", in_dim,
        "]"));
  }

  auto sinfo_or = ploader.GetTensorInfo(name + ".scale");
  if (!sinfo_or.ok()) return sinfo_or.status();
  const auto& sinfo = *sinfo_or;
  const size_t s_count =
      static_cast<size_t>(out_channels) * static_cast<size_t>(blocks_per_row);
  if (sinfo.dtype != safetensors::kFLOAT32 ||
      sinfo.data_end - sinfo.data_start != s_count * sizeof(float)) {
    return absl::FailedPreconditionError(absl::StrCat(
        name, ".scale: must be F32 [", out_channels, ",", blocks_per_row,
        "]"));
  }

  const int8_t* q = reinterpret_cast<const int8_t*>(
      qinfo.storage->data_base + qinfo.data_start);
  const float* s = reinterpret_cast<const float*>(
      sinfo.storage->data_base + sinfo.data_start);
  return BuildInt4Tensor(q, std::vector<float>(s, s + s_count), out_channels,
                         in_dim, name);
}

}  // namespace

absl::StatusOr<WeightMap> LoadCheckpointWeights(const TalkerConfig& config,
                                                const std::string& path,
                                                WeightQuantMode quant_mode,
                                                const std::string& prequant_path) {
  auto loader_or = SafetensorLoader::Load(path);
  if (!loader_or.ok()) return loader_or.status();
  auto loader = std::move(*loader_or);

  std::optional<SafetensorLoader> prequant;
  if (!prequant_path.empty()) {
    auto ploader_or = SafetensorLoader::Load(prequant_path);
    if (!ploader_or.ok()) return ploader_or.status();
    prequant = std::move(*ploader_or);
  }

  WeightMap weights;
  for (const WeightSpec& spec : GetWeightSpecs(config)) {
    if (prequant.has_value() && IsQuantizableFcWeight(spec.name)) {
      if (spec.shape.size() != 2) {
        return absl::FailedPreconditionError(
            absl::StrCat(spec.name, ": prequant weight must be 2-D"));
      }
      auto tensor_or = LoadPrequantInt4Weight(*prequant, spec.name,
                                              spec.shape[0], spec.shape[1]);
      if (!tensor_or.ok()) return tensor_or.status();
      weights[spec.name] = std::move(*tensor_or);
      continue;
    }
    auto handle_or = loader.LoadTensor(
        spec.name, SafetensorLoader::QuantizedLoadMode::kDequantizeToFp32);
    if (!handle_or.ok()) {
      return absl::NotFoundError(absl::StrCat(
          "checkpoint missing ", spec.name, ": ",
          handle_or.status().message()));
    }
    TfTensor tensor(*handle_or);

    const auto& loaded_shape = tensor.GetShape();
    std::vector<int> loaded(loaded_shape.begin(), loaded_shape.end());
    if (loaded != spec.shape) {
      return absl::FailedPreconditionError(absl::StrCat(
          spec.name, ": checkpoint shape [", absl::StrJoin(loaded, ","),
          "] != expected [", absl::StrJoin(spec.shape, ","), "]"));
    }

    if (LoaderAddsGemmaNormOffset(spec.name)) {
      auto status = SubtractOne(tensor);
      if (!status.ok()) return status;
    }

    if (quant_mode != WeightQuantMode::kNone &&
        IsQuantizableFcWeight(spec.name)) {
      auto quantized_or = QuantizeFcWeight(tensor, spec.name, quant_mode);
      if (!quantized_or.ok()) return quantized_or.status();
      weights[spec.name] = std::move(*quantized_or);
      continue;
    }
    tensor.SetName(spec.name);
    weights[spec.name] = std::move(tensor);
  }
  return weights;
}

}  // namespace litert::tensor::examples::talker
