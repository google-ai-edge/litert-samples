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

// Codec decoder step-graph. See codec_graph.h.

#include "models/qwen/qwen3_tts/tensor_api/codec_decode/codec_graph.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "models/qwen/qwen3_tts/tensor_api/codec_decode/codec_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::codec {

namespace {

constexpr float kMaskFloor = -30000.0f;  // fp16-safe additive mask floor

TfTensor Const1(float value) {
  return TfTensor({.type = Type::kFP32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>({value})});
}

const TfTensor& W(const WeightMap& weights, const std::string& name) {
  auto it = weights.find(name);
  ABSL_CHECK(it != weights.end()) << "missing weight: " << name;
  return it->second;
}

// ---- generic pieces ------------------------------------------------------

TfTensor RmsNorm(const TfTensor& input, const TfTensor& scale, float eps) {
  TfTensor x2 = Mul(input, input);
  int last = static_cast<int>(input.GetShape().size()) - 1;
  TfTensor ms = Mean(x2, {last}, /*keep_dims=*/true);
  return Mul(Mul(input, Rsqrt(Add(ms, Const1(eps)))), scale);
}

TfTensor Silu(const TfTensor& x) { return Mul(x, Logistic(x)); }

// LayerNorm over the last axis with gamma+beta (ConvNeXt norm).
TfTensor LayerNorm(const TfTensor& x, const TfTensor& gamma,
                   const TfTensor& beta, float eps) {
  int last = static_cast<int>(x.GetShape().size()) - 1;
  TfTensor m = Mean(x, {last}, /*keep_dims=*/true);
  TfTensor d = Sub(x, m);
  TfTensor v = Mean(Mul(d, d), {last}, /*keep_dims=*/true);
  return Add(Mul(Mul(d, Rsqrt(Add(v, Const1(eps)))), gamma), beta);
}

// SnakeBeta on NHWC [1,1,T,C]: x + beta_inv * sin(x * alpha_exp)^2, with
// alpha_exp = exp(alpha), beta_inv = 1/(exp(beta)+1e-9) precomputed at load.
TfTensor Snake(const TfTensor& x, const TfTensor& alpha_exp,
               const TfTensor& beta_inv) {
  TfTensor s = Sin(Mul(x, alpha_exp));
  return Add(x, Mul(beta_inv, Mul(s, s)));
}

// Left-pad the width axis of NHWC [1,1,T,C] by `left` zeros.
TfTensor PadLeftW(const TfTensor& x, int left) {
  if (left == 0) return x;
  TfTensor paddings(
      {.type = Type::kI32,
       .shape = {4, 2},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(
           std::vector<int32_t>{0, 0, 0, 0, left, 0, 0, 0})});
  return Pad(x, paddings);
}

// Causal Conv1d on NHWC [1,1,T,C]: left-pad (k-1)*dilation, VALID conv.
// filter [out, 1, k, in] (+bias [out]) already in TFLite layout.
TfTensor CausalConv(const TfTensor& x, const TfTensor& filter,
                    const TfTensor& bias, int dilation = 1) {
  int k = filter.GetShape()[2];
  TfTensor padded = PadLeftW(x, (k - 1) * dilation);
  return Conv2D(padded, filter, bias, /*stride_h=*/1, /*stride_w=*/1,
                kPaddingValid, /*dilation_h=*/1, /*dilation_w=*/dilation);
}

// Causal depthwise Conv1d k=7 on NHWC; filter [1, 1, k, C].
TfTensor CausalDwConv(const TfTensor& x, const TfTensor& filter,
                      const TfTensor& bias) {
  int k = filter.GetShape()[2];
  TfTensor padded = PadLeftW(x, k - 1);
  return DepthwiseConv2D(padded, filter, bias, /*stride_h=*/1, /*stride_w=*/1,
                         kPaddingValid);
}

// Causal ConvTranspose1d: VALID transpose-conv (out = (T-1)*s + k), then crop
// (k - s) from the right. filter [out, 1, k, in], bias [out].
TfTensor CausalTransConv(const TfTensor& x, const TfTensor& filter,
                         const TfTensor& bias, int stride) {
  const auto& s = x.GetShape();  // [1,1,T,in]
  int T = s[2];
  int k = filter.GetShape()[2];
  int out_ch = filter.GetShape()[0];
  int full = (T - 1) * stride + k;
  TfTensor y = TransposeConv(filter, x, bias, {1, 1, full, out_ch},
                             kPaddingValid, /*stride_h=*/1,
                             /*stride_w=*/stride);
  int crop = k - stride;
  if (crop == 0) return y;
  return Slice(y, {0, 0, 0, 0}, {1, 1, full - crop, out_ch});
}

// ---- RVQ -----------------------------------------------------------------

// codes [16, T] -> latent [1, 1, T, codebook_dim].
TfTensor RvqDecode(const CodecConfig& config, const TfTensor& codes, int T,
                   const WeightMap& weights) {
  auto gather_q = [&](int q) {
    TfTensor row = Slice(codes, {q, 0}, {1, T});
    TfTensor idx = Reshape(row, {T});  // 1-D indices (Metal-safe)
    return Gather(W(weights, absl::StrCat("codebook_table_", q)), idx,
                  /*axis=*/0);  // [T, vq_dim]
  };

  TfTensor semantic = FullyConnected(
      gather_q(0), W(weights, "decoder.quantizer.rvq_first.output_proj"));
  TfTensor acoustic_sum = gather_q(1);
  for (int q = 2; q < config.num_quantizers; ++q) {
    acoustic_sum = Add(acoustic_sum, gather_q(q));
  }
  TfTensor acoustic = FullyConnected(
      acoustic_sum, W(weights, "decoder.quantizer.rvq_rest.output_proj"));
  TfTensor latent = Add(semantic, acoustic);  // [T, codebook_dim]
  return Reshape(latent, {1, 1, T, config.codebook_dim});
}

// ---- pre_transformer -----------------------------------------------------

std::pair<TfTensor, TfTensor> ConstRope(int seq, int head_dim, float base) {
  std::vector<float> cos_v(static_cast<size_t>(seq) * head_dim);
  std::vector<float> sin_v(static_cast<size_t>(seq) * head_dim);
  int half = head_dim / 2;
  for (int t = 0; t < seq; ++t) {
    for (int i = 0; i < half; ++i) {
      float freq = std::pow(base, -2.0f * static_cast<float>(i) / head_dim);
      float angle = static_cast<float>(t) * freq;
      cos_v[t * head_dim + i] = std::cos(angle);
      cos_v[t * head_dim + half + i] = std::cos(angle);
      sin_v[t * head_dim + i] = std::sin(angle);
      sin_v[t * head_dim + half + i] = std::sin(angle);
    }
  }
  TfTensor cos_t({.type = Type::kFP32,
                  .shape = {1, 1, seq, head_dim},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(cos_v)});
  TfTensor sin_t({.type = Type::kFP32,
                  .shape = {1, 1, seq, head_dim},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(sin_v)});
  return {cos_t, sin_t};
}

TfTensor ApplyRope(const TfTensor& x, const TfTensor& cos,
                   const TfTensor& sin) {
  const auto& s = x.GetShape();
  int half = s[3] / 2;
  TfTensor x1 = Slice(x, {0, 0, 0, 0}, {s[0], s[1], s[2], half});
  TfTensor x2 = Slice(x, {0, 0, 0, half}, {s[0], s[1], s[2], half});
  TfTensor rot = Concatenation({Mul(x2, Const1(-1.0f)), x1}, /*axis=*/3);
  return Add(Mul(x, cos), Mul(rot, sin));
}

// Baked causal + sliding-window additive mask [1,1,T,T]:
// key j visible from query i iff i-window < j <= i.
TfTensor SlidingCausalMask(int T, int window) {
  std::vector<float> mask(static_cast<size_t>(T) * T, kMaskFloor);
  for (int i = 0; i < T; ++i) {
    for (int j = std::max(0, i - window + 1); j <= i; ++j) {
      mask[static_cast<size_t>(i) * T + j] = 0.0f;
    }
  }
  return TfTensor({.type = Type::kFP32,
                   .shape = {1, 1, T, T},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>(mask)});
}

// x [1, T, hidden] -> [1, T, hidden]; MHA (heads == kv heads), no qk-norm.
TfTensor PreTransformer(const CodecConfig& config, TfTensor x, int T,
                        const WeightMap& weights) {
  const std::string base = "decoder.pre_transformer";
  x = FullyConnected(x, W(weights, base + ".input_proj.weight"),
                     W(weights, base + ".input_proj.bias"));

  auto [cos, sin] = ConstRope(T, config.tf_head_dim, config.rope_theta);
  TfTensor mask = SlidingCausalMask(T, config.sliding_window);
  float scale = 1.0f / std::sqrt(static_cast<float>(config.tf_head_dim));

  for (int l = 0; l < config.tf_layers; ++l) {
    std::string p = absl::StrCat(base, ".layers.", l);
    TfTensor h = RmsNorm(x, W(weights, p + ".input_layernorm.weight"),
                         config.tf_eps);
    TfTensor q = FullyConnected(h, W(weights, p + ".self_attn.q_proj.weight"));
    TfTensor k = FullyConnected(h, W(weights, p + ".self_attn.k_proj.weight"));
    TfTensor v = FullyConnected(h, W(weights, p + ".self_attn.v_proj.weight"));
    q = Transpose(Reshape(q, {1, T, config.tf_heads, config.tf_head_dim}),
                  {0, 2, 1, 3});
    k = Transpose(Reshape(k, {1, T, config.tf_heads, config.tf_head_dim}),
                  {0, 2, 1, 3});
    v = Transpose(Reshape(v, {1, T, config.tf_heads, config.tf_head_dim}),
                  {0, 2, 1, 3});
    q = ApplyRope(q, cos, sin);
    k = ApplyRope(k, cos, sin);
    TfTensor scores = BatchMatMul(q, k, /*adj_x=*/false, /*adj_y=*/true);
    scores = Add(Mul(scores, Const1(scale)), mask);
    TfTensor attn = BatchMatMul(Softmax(scores), v);
    attn = Reshape(Transpose(attn, {0, 2, 1, 3}),
                   {1, T, config.tf_heads * config.tf_head_dim});
    attn = FullyConnected(attn, W(weights, p + ".self_attn.o_proj.weight"));
    x = Add(x, Mul(attn, W(weights, p + ".self_attn_layer_scale.scale")));

    TfTensor f = RmsNorm(x, W(weights, p + ".post_attention_layernorm.weight"),
                         config.tf_eps);
    TfTensor gate = FullyConnected(f, W(weights, p + ".mlp.gate_proj.weight"));
    TfTensor up = FullyConnected(f, W(weights, p + ".mlp.up_proj.weight"));
    TfTensor ffn = FullyConnected(Mul(Silu(gate), up),
                                  W(weights, p + ".mlp.down_proj.weight"));
    x = Add(x, Mul(ffn, W(weights, p + ".mlp_layer_scale.scale")));
  }

  x = RmsNorm(x, W(weights, base + ".norm.weight"), config.tf_eps);
  return FullyConnected(x, W(weights, base + ".output_proj.weight"),
                        W(weights, base + ".output_proj.bias"));
}

// ---- ConvNeXt / decoder blocks ------------------------------------------

TfTensor ConvNeXtBlock(const TfTensor& x, const std::string& p,
                       const WeightMap& weights) {
  TfTensor h = CausalDwConv(x, W(weights, p + ".dwconv.conv.weight"),
                            W(weights, p + ".dwconv.conv.bias"));
  h = LayerNorm(h, W(weights, p + ".norm.weight"),
                W(weights, p + ".norm.bias"), 1e-6f);
  h = FullyConnected(h, W(weights, p + ".pwconv1.weight"),
                     W(weights, p + ".pwconv1.bias"));
  h = Gelu(h);
  h = FullyConnected(h, W(weights, p + ".pwconv2.weight"),
                     W(weights, p + ".pwconv2.bias"));
  h = Mul(h, W(weights, p + ".gamma"));
  return Add(x, h);
}

TfTensor ResidualUnit(const TfTensor& x, const std::string& p, int dilation,
                      const WeightMap& weights) {
  TfTensor h = Snake(x, W(weights, p + ".act1.alpha_exp"),
                     W(weights, p + ".act1.beta_inv"));
  h = CausalConv(h, W(weights, p + ".conv1.conv.weight"),
                 W(weights, p + ".conv1.conv.bias"), dilation);
  h = Snake(h, W(weights, p + ".act2.alpha_exp"),
            W(weights, p + ".act2.beta_inv"));
  h = CausalConv(h, W(weights, p + ".conv2.conv.weight"),
                 W(weights, p + ".conv2.conv.bias"));
  return Add(x, h);
}

}  // namespace

TfTensor MakeCodesInput(const CodecConfig& config, int frames) {
  return TfTensor({.name = "codes",
                   .type = Type::kI32,
                   .shape = {config.num_quantizers, frames}});
}

TfTensor BuildCodecDecode(const CodecConfig& config, const TfTensor& codes,
                          int frames, const WeightMap& weights) {
  const int T = frames;

  // RVQ -> [1,1,T,512] -> causal pre_conv k3 -> [1,1,T,1024].
  TfTensor h = RvqDecode(config, codes, T, weights);
  h = CausalConv(h, W(weights, "decoder.pre_conv.conv.weight"),
                 W(weights, "decoder.pre_conv.conv.bias"));

  // pre_transformer on [1,T,latent].
  h = Reshape(h, {1, T, config.latent_dim});
  h = PreTransformer(config, h, T, weights);
  h = Reshape(h, {1, 1, T, config.latent_dim});

  // upsample stages: TransConv(k=s=r) + ConvNeXt.
  for (size_t i = 0; i < config.upsampling_ratios.size(); ++i) {
    int r = config.upsampling_ratios[i];
    std::string p = absl::StrCat("decoder.upsample.", i);
    h = CausalTransConv(h, W(weights, p + ".0.conv.weight"),
                        W(weights, p + ".0.conv.bias"), r);
    h = ConvNeXtBlock(h, p + ".1", weights);
  }

  // decoder stack: conv k7 -> 4 blocks -> Snake -> conv k7 -> clamp.
  h = CausalConv(h, W(weights, "decoder.decoder.0.conv.weight"),
                 W(weights, "decoder.decoder.0.conv.bias"));
  for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
    int rate = config.upsample_rates[i];
    std::string p = absl::StrCat("decoder.decoder.", i + 1, ".block");
    h = Snake(h, W(weights, p + ".0.alpha_exp"),
              W(weights, p + ".0.beta_inv"));
    h = CausalTransConv(h, W(weights, p + ".1.conv.weight"),
                        W(weights, p + ".1.conv.bias"), rate);
    const int dilations[3] = {1, 3, 9};
    for (int u = 0; u < 3; ++u) {
      h = ResidualUnit(h, absl::StrCat(p, ".", u + 2), dilations[u], weights);
    }
  }
  int last = static_cast<int>(config.upsample_rates.size()) + 1;
  h = Snake(h, W(weights, absl::StrCat("decoder.decoder.", last, ".alpha_exp")),
            W(weights, absl::StrCat("decoder.decoder.", last, ".beta_inv")));
  h = CausalConv(
      h, W(weights, absl::StrCat("decoder.decoder.", last + 1, ".conv.weight")),
      W(weights, absl::StrCat("decoder.decoder.", last + 1, ".conv.bias")));

  h = Maximum(Minimum(h, Const1(1.0f)), Const1(-1.0f));
  TfTensor wav = Reshape(h, {1, T * config.total_upsample()});
  wav.SetName("wav");
  return wav;
}

}  // namespace litert::tensor::examples::codec
