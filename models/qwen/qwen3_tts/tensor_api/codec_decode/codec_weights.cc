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

// Codec decoder weight loading. See codec_weights.h.

#include "models/qwen/qwen3_tts/tensor_api/codec_decode/codec_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::codec {

namespace {

using ::litert::tensor::examples::SafetensorLoader;

// Mirrors the loader's is_layernorm predicate (safetensor_loader.cc), which
// adds a Gemma-style +1.0 these norms must not have.
bool LoaderAddsGemmaNormOffset(const std::string& name) {
  return absl::StrContains(name, "layernorm") ||
         absl::EndsWith(name, "norm.weight");
}

struct RawTensor {
  std::vector<float> data;
  std::vector<int> shape;
};

absl::StatusOr<RawTensor> ReadTensor(const SafetensorLoader& loader,
                                     const std::string& name) {
  auto handle_or = loader.LoadTensor(
      name, SafetensorLoader::QuantizedLoadMode::kDequantizeToFp32);
  if (!handle_or.ok()) {
    return absl::NotFoundError(
        absl::StrCat("missing ", name, ": ", handle_or.status().message()));
  }
  TfTensor t(*handle_or);
  RawTensor raw;
  const auto& s = t.GetShape();
  raw.shape.assign(s.begin(), s.end());
  auto buffer = t.GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* p = reinterpret_cast<const float*>(lock.data());
  raw.data.assign(p, p + lock.size() / sizeof(float));
  if (LoaderAddsGemmaNormOffset(name)) {
    for (float& v : raw.data) v -= 1.0f;
  }
  return raw;
}

TfTensor MakeConst(const std::string& name, const std::vector<int>& shape,
                   std::vector<float> data) {
  return TfTensor({.name = name,
                   .type = Type::kFP32,
                   .shape = shape,
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>(data)});
}

// torch Conv1d [out, in, k] -> TFLite Conv2D filter [out, 1, k, in].
TfTensor ConvFilter(const std::string& name, const RawTensor& w) {
  int out = w.shape[0], in = w.shape[1], k = w.shape[2];
  std::vector<float> data(w.data.size());
  for (int o = 0; o < out; ++o)
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < in; ++i)
        data[(static_cast<size_t>(o) * k + j) * in + i] =
            w.data[(static_cast<size_t>(o) * in + i) * k + j];
  return MakeConst(name, {out, 1, k, in}, std::move(data));
}

// torch depthwise Conv1d [C, 1, k] -> TFLite DepthwiseConv2D [1, 1, k, C].
TfTensor DwConvFilter(const std::string& name, const RawTensor& w) {
  int c = w.shape[0], k = w.shape[2];
  std::vector<float> data(w.data.size());
  for (int j = 0; j < k; ++j)
    for (int ch = 0; ch < c; ++ch)
      data[static_cast<size_t>(j) * c + ch] =
          w.data[static_cast<size_t>(ch) * k + j];
  return MakeConst(name, {1, 1, k, c}, std::move(data));
}

// torch ConvTranspose1d [in, out, k] -> TFLite TransposeConv [out, 1, k, in].
TfTensor TransConvFilter(const std::string& name, const RawTensor& w) {
  int in = w.shape[0], out = w.shape[1], k = w.shape[2];
  std::vector<float> data(w.data.size());
  for (int o = 0; o < out; ++o)
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < in; ++i)
        data[(static_cast<size_t>(o) * k + j) * in + i] =
            w.data[(static_cast<size_t>(i) * out + o) * k + j];
  return MakeConst(name, {out, 1, k, in}, std::move(data));
}

}  // namespace

absl::StatusOr<WeightMap> LoadCodecWeights(const CodecConfig& config,
                                           const std::string& path) {
  auto loader_or = SafetensorLoader::Load(path);
  if (!loader_or.ok()) return loader_or.status();
  auto loader = std::move(*loader_or);

  WeightMap weights;
  auto read = [&](const std::string& name) { return ReadTensor(loader, name); };

  // Pass a raw tensor through unchanged (norm fixup already applied).
  auto passthrough = [&](const std::string& name) -> absl::Status {
    auto raw = read(name);
    if (!raw.ok()) return raw.status();
    weights[name] = MakeConst(name, raw->shape, std::move(raw->data));
    return absl::OkStatus();
  };
  // Snake params: store exp(alpha) and 1/(exp(beta)+1e-9).
  auto snake = [&](const std::string& base) -> absl::Status {
    auto alpha = read(base + ".alpha");
    if (!alpha.ok()) return alpha.status();
    auto beta = read(base + ".beta");
    if (!beta.ok()) return beta.status();
    std::vector<float> ae(alpha->data.size()), bi(beta->data.size());
    for (size_t i = 0; i < ae.size(); ++i) ae[i] = std::exp(alpha->data[i]);
    for (size_t i = 0; i < bi.size(); ++i)
      bi[i] = 1.0f / (std::exp(beta->data[i]) + 1e-9f);
    weights[base + ".alpha_exp"] =
        MakeConst(base + ".alpha_exp", alpha->shape, std::move(ae));
    weights[base + ".beta_inv"] =
        MakeConst(base + ".beta_inv", beta->shape, std::move(bi));
    return absl::OkStatus();
  };
  auto conv = [&](const std::string& name) -> absl::Status {
    auto w = read(name + ".weight");
    if (!w.ok()) return w.status();
    weights[name + ".weight"] = ConvFilter(name + ".weight", *w);
    return passthrough(name + ".bias");
  };
  auto transconv = [&](const std::string& name) -> absl::Status {
    auto w = read(name + ".weight");
    if (!w.ok()) return w.status();
    weights[name + ".weight"] = TransConvFilter(name + ".weight", *w);
    return passthrough(name + ".bias");
  };

#define RETURN_IF_ERROR(expr)                \
  do {                                       \
    absl::Status _st = (expr);               \
    if (!_st.ok()) return _st;               \
  } while (false)

  // --- RVQ: EMA tables + squeezed output projections ---
  for (int q = 0; q < config.num_quantizers; ++q) {
    std::string base =
        q == 0 ? "decoder.quantizer.rvq_first.vq.layers.0._codebook"
               : absl::StrCat("decoder.quantizer.rvq_rest.vq.layers.", q - 1,
                              "._codebook");
    auto sum = read(base + ".embedding_sum");
    if (!sum.ok()) return sum.status();
    auto usage = read(base + ".cluster_usage");
    if (!usage.ok()) return usage.status();
    int rows = sum->shape[0], dim = sum->shape[1];
    std::vector<float> table(sum->data.size());
    for (int r = 0; r < rows; ++r) {
      float u = std::max(usage->data[r], config.vq_eps);
      for (int d = 0; d < dim; ++d)
        table[static_cast<size_t>(r) * dim + d] =
            sum->data[static_cast<size_t>(r) * dim + d] / u;
    }
    std::string tname = absl::StrCat("codebook_table_", q);
    weights[tname] = MakeConst(tname, {rows, dim}, std::move(table));
  }
  for (const std::string side : {"rvq_first", "rvq_rest"}) {
    std::string name = absl::StrCat("decoder.quantizer.", side, ".output_proj");
    auto w = read(name + ".weight");  // [512, 256, 1]
    if (!w.ok()) return w.status();
    weights[name] = MakeConst(name, {w->shape[0], w->shape[1]},
                              std::move(w->data));
  }

  // --- pre_conv + pre_transformer ---
  RETURN_IF_ERROR(conv("decoder.pre_conv.conv"));
  {
    const std::string base = "decoder.pre_transformer";
    RETURN_IF_ERROR(passthrough(base + ".input_proj.weight"));
    RETURN_IF_ERROR(passthrough(base + ".input_proj.bias"));
    RETURN_IF_ERROR(passthrough(base + ".output_proj.weight"));
    RETURN_IF_ERROR(passthrough(base + ".output_proj.bias"));
    RETURN_IF_ERROR(passthrough(base + ".norm.weight"));
    for (int l = 0; l < config.tf_layers; ++l) {
      std::string p = absl::StrCat(base, ".layers.", l);
      for (const char* suffix :
           {".input_layernorm.weight", ".post_attention_layernorm.weight",
            ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
            ".self_attn_layer_scale.scale", ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight", ".mlp.down_proj.weight",
            ".mlp_layer_scale.scale"}) {
        RETURN_IF_ERROR(passthrough(p + suffix));
      }
    }
  }

  // --- upsample stages ---
  for (size_t i = 0; i < config.upsampling_ratios.size(); ++i) {
    std::string p = absl::StrCat("decoder.upsample.", i);
    RETURN_IF_ERROR(transconv(p + ".0.conv"));
    {
      std::string c = p + ".1";
      auto w = read(c + ".dwconv.conv.weight");
      if (!w.ok()) return w.status();
      weights[c + ".dwconv.conv.weight"] =
          DwConvFilter(c + ".dwconv.conv.weight", *w);
      RETURN_IF_ERROR(passthrough(c + ".dwconv.conv.bias"));
      for (const char* suffix :
           {".norm.weight", ".norm.bias", ".pwconv1.weight", ".pwconv1.bias",
            ".pwconv2.weight", ".pwconv2.bias", ".gamma"}) {
        RETURN_IF_ERROR(passthrough(c + suffix));
      }
    }
  }

  // --- decoder stack ---
  RETURN_IF_ERROR(conv("decoder.decoder.0.conv"));
  for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
    std::string p = absl::StrCat("decoder.decoder.", i + 1, ".block");
    RETURN_IF_ERROR(snake(p + ".0"));
    RETURN_IF_ERROR(transconv(p + ".1.conv"));
    for (int u = 2; u <= 4; ++u) {
      std::string r = absl::StrCat(p, ".", u);
      RETURN_IF_ERROR(snake(r + ".act1"));
      RETURN_IF_ERROR(conv(r + ".conv1.conv"));
      RETURN_IF_ERROR(snake(r + ".act2"));
      RETURN_IF_ERROR(conv(r + ".conv2.conv"));
    }
  }
  int last = static_cast<int>(config.upsample_rates.size()) + 1;
  RETURN_IF_ERROR(snake(absl::StrCat("decoder.decoder.", last)));
  RETURN_IF_ERROR(conv(absl::StrCat("decoder.decoder.", last + 1, ".conv")));

#undef RETURN_IF_ERROR
  return weights;
}

}  // namespace litert::tensor::examples::codec
