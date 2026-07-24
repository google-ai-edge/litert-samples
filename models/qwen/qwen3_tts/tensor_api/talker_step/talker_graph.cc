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

// Talker step-graph — Stage 2. See talker_graph.h.

#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_graph.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::talker {

namespace {

// fp16-safe additive mask floor (fp32 -1e9 overflows fp16 on GPU).
constexpr float kMaskFloor = -30000.0f;

TfTensor Const1(float value) {
  return TfTensor({.type = Type::kFP32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>({value})});
}

// Deterministic LCG so runs are reproducible without an RNG dependency.
std::vector<float> SyntheticData(size_t count, unsigned& state, float scale) {
  std::vector<float> data(count);
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    data[i] = scale * (static_cast<float>(state >> 8) /
                           static_cast<float>(1u << 24) * 2.0f -
                       1.0f);
  }
  return data;
}

TfTensor SyntheticWeight(const std::string& name, const std::vector<int>& shape,
                         unsigned& state, float scale = 0.02f) {
  size_t count = 1;
  for (int d : shape) count *= static_cast<size_t>(d);
  return TfTensor({.name = name,
                   .type = Type::kFP32,
                   .shape = shape,
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                       SyntheticData(count, state, scale))});
}

const TfTensor& GetWeight(const WeightMap& weights, const std::string& name) {
  auto it = weights.find(name);
  ABSL_CHECK(it != weights.end()) << "missing weight: " << name;
  return it->second;
}

// Toggled per BuildTalkerStep call from TalkerConfig::use_rms_norm_composite
// (single-threaded graph construction).
bool g_rms_norm_composite = false;

// RMS normalization composed from primitives.
TfTensor RmsNormRaw(const TfTensor& input, const TfTensor& scale, float eps) {
  TfTensor x_squared = Mul(input, input);
  int last_axis = static_cast<int>(input.GetShape().size()) - 1;
  TfTensor mean_squared = Mean(x_squared, {last_axis}, /*keep_dims=*/true);
  TfTensor inv_rms = Rsqrt(Add(mean_squared, Const1(eps)));
  return Mul(Mul(input, inv_rms), scale);
}

std::vector<uint8_t> RmsNormAttributes(float eps) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("epsilon", eps); });
  fbb.Finish();
  return fbb.GetBuffer();
}

// RMS norm, optionally as the odml.rms_norm composite (fused ML Drift
// kernels on Metal and CL; the decomposition is the raw math, so CPU
// execution is identical either way).
TfTensor RmsNorm(const TfTensor& input, const TfTensor& scale, float eps) {
  if (!g_rms_norm_composite) return RmsNormRaw(input, scale, eps);
  StableHLOCompositeOptions opts{.name = "odml.rms_norm",
                                 .composite_attributes =
                                     RmsNormAttributes(eps)};
  return StableHLOComposite(
      opts,
      [eps](TfTensor x, TfTensor s) { return RmsNormRaw(x, s, eps); }, input,
      scale);
}

TfTensor Silu(const TfTensor& x) { return Mul(x, Logistic(x)); }

// Rotate-half RoPE with cos/sin provided as tensors (host input for the
// talker; baked constants for the code predictor).
TfTensor ApplyRope(const TfTensor& x, const TfTensor& cos,
                   const TfTensor& sin) {
  const auto& s = x.GetShape();  // [1, heads, seq, head_dim]
  int half = s[3] / 2;
  TfTensor x1 = Slice(x, {0, 0, 0, 0}, {s[0], s[1], s[2], half});
  TfTensor x2 = Slice(x, {0, 0, 0, half}, {s[0], s[1], s[2], half});
  TfTensor rotated = Concatenation({Mul(x2, Const1(-1.0f)), x1}, /*axis=*/3);
  return Add(Mul(x, cos), Mul(rotated, sin));
}

// HF-pairing GQA attention core without materializing repeated K/V: fold the
// kv groups into the batch axis and each group's q heads into the sequence
// axis, so q head h reads kv group h / (heads/kv) — repeat_interleave
// semantics. (A plain Tile pairs head h with group h % kv instead — wrong
// whenever 1 < kv < heads; the gemma3 example only supports kv == 1, where
// the two coincide. Gather-axis-1 repeat is rejected by the Metal delegate.)
// q [1, heads, seq, hd], k/v [1, kv, kv_seq, hd], mask [1, 1|heads, seq,
// kv_seq] additive -> output [1, heads, seq, hd].
TfTensor GqaAttention(const TfTensor& q, const TfTensor& k, const TfTensor& v,
                      const TfTensor& mask, int n_heads, int n_kv_groups,
                      int head_dim) {
  int g = n_heads / n_kv_groups;
  int seq = q.GetShape()[2];
  int kv_seq = k.GetShape()[2];
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  TfTensor qg = Reshape(q, {n_kv_groups, 1, g * seq, head_dim});
  TfTensor kg = Reshape(k, {n_kv_groups, 1, kv_seq, head_dim});
  TfTensor vg = Reshape(v, {n_kv_groups, 1, kv_seq, head_dim});

  TfTensor scores = BatchMatMul(qg, kg, /*adj_x=*/false, /*adj_y=*/true);
  scores = Reshape(scores, {1, n_heads, seq, kv_seq});
  scores = Mul(scores, Const1(scale));
  scores = Add(scores, mask);
  TfTensor attn = Softmax(scores);
  attn = Reshape(attn, {n_kv_groups, 1, g * seq, kv_seq});
  TfTensor out = BatchMatMul(attn, vg);
  return Reshape(out, {1, n_heads, seq, head_dim});
}

std::vector<uint8_t> RuntimeBmmAttributes(bool is_src) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Bool("is_global", true);
    fbb.Bool("is_src", is_src);
    fbb.Bool("rhs_cache_update", false);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

// GQA attention via the odml.runtime_bmm composite pair in the "avbound"
// composition: the QK side runs UNBOUNDED (runtime_param_full carries
// max_seq) so every tail score is deterministically written and the fused
// scale/mask epilogue covers the full width; only the scores x V side is
// runtime-bounded (runtime_param carries the active length). A dst-bounded
// QK would leave its output tail unwritten, which the full-width Softmax in
// between would then read - see the attn_bench findings. The decompositions
// are plain BatchMatMuls, so CPU execution equals the raw-op path exactly.
// q [1, heads, seq, hd], k [1, kv, max_seq, hd], v_t [1, kv, hd, max_seq],
// mask [1, 1|heads, seq, max_seq] additive -> output [1, heads, seq, hd].
TfTensor RbmmGqaAttention(const TfTensor& q, const TfTensor& k,
                          const TfTensor& v_t, const TfTensor& mask,
                          const TfTensor& runtime_param,
                          const TfTensor& runtime_param_full, int n_heads,
                          int n_kv_groups, int head_dim, int max_seq) {
  int g = n_heads / n_kv_groups;
  int seq = q.GetShape()[2];
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  TfTensor qg = Reshape(q, {1, n_kv_groups, g * seq, head_dim});
  StableHLOCompositeOptions qk_opts{
      .name = "odml.runtime_bmm",
      .composite_attributes = RuntimeBmmAttributes(/*is_src=*/false)};
  TfTensor scores = StableHLOComposite(
      qk_opts,
      [](TfTensor l, TfTensor r, TfTensor /*p*/) {
        return BatchMatMul(l, r, /*adj_x=*/false, /*adj_y=*/true);
      },
      qg, k, runtime_param_full);  // [1, kv, g*seq, max_seq]
  scores = Reshape(scores, {1, n_heads, seq, max_seq});
  scores = Mul(scores, Const1(scale));
  scores = Add(scores, mask);
  TfTensor attn = Softmax(scores);
  attn = Reshape(attn, {1, n_kv_groups, g * seq, max_seq});
  StableHLOCompositeOptions av_opts{
      .name = "odml.runtime_bmm",
      .composite_attributes = RuntimeBmmAttributes(/*is_src=*/true)};
  TfTensor out = StableHLOComposite(
      av_opts,
      [](TfTensor l, TfTensor r, TfTensor /*p*/) {
        return BatchMatMul(l, r, /*adj_x=*/false, /*adj_y=*/true);
      },
      attn, v_t, runtime_param);  // [1, kv, g*seq, hd]
  return Reshape(out, {1, n_heads, seq, head_dim});
}

// Generic pre-norm GQA transformer layer used by both the talker backbone and
// the code predictor. `k_ext`/`v_ext` (optional) provide externally updated
// caches for attention; when null, attention runs over this call's k/v only.
struct LayerWeights {
  std::string prefix;
};

struct AttentionResult {
  TfTensor output;
  TfTensor key_cache;
  TfTensor value_cache;
};

// Talker self-attention with a FIXED-SIZE cache updated in-graph via
// DynamicUpdateSlice at the runtime cache_position (both signatures pass the
// position as an input — the Metal delegate rejects constant start_indices).
AttentionResult TalkerAttention(const TalkerConfig& config,
                                const TfTensor& input, const std::string& prefix,
                                const TalkerInputs& inputs, int layer_idx,
                                const WeightMap& weights) {
  const auto& in_shape = input.GetShape();
  int seq = in_shape[1];

  TfTensor q = FullyConnected(input, GetWeight(weights, prefix + ".q_proj.weight"));
  TfTensor k = FullyConnected(input, GetWeight(weights, prefix + ".k_proj.weight"));
  TfTensor v = FullyConnected(input, GetWeight(weights, prefix + ".v_proj.weight"));

  q = Reshape(q, {1, seq, config.n_heads, config.head_dim});
  q = Transpose(q, {0, 2, 1, 3});
  k = Reshape(k, {1, seq, config.n_kv_groups, config.head_dim});
  k = Transpose(k, {0, 2, 1, 3});
  v = Reshape(v, {1, seq, config.n_kv_groups, config.head_dim});
  v = Transpose(v, {0, 2, 1, 3});

  q = RmsNorm(q, GetWeight(weights, prefix + ".q_norm.weight"),
              config.rms_norm_eps);
  k = RmsNorm(k, GetWeight(weights, prefix + ".k_norm.weight"),
              config.rms_norm_eps);

  q = ApplyRope(q, inputs.rope_cos, inputs.rope_sin);
  k = ApplyRope(k, inputs.rope_cos, inputs.rope_sin);

  TfTensor key_cache_new = DynamicUpdateSlice(inputs.key_caches[layer_idx], k,
                                              inputs.cache_position);
  key_cache_new.SetName(absl::StrCat("key_cache_", layer_idx, "_new"));

  TfTensor value_cache_new;
  TfTensor out;
  if (config.use_runtime_bmm) {
    // Value cache lives transposed ([1, kv, hd, max_seq]); write this call's
    // rows as columns at the runtime position on the last axis.
    TfTensor v_t = Transpose(v, {0, 1, 3, 2});  // [1, kv, hd, seq]
    value_cache_new = DynamicUpdateSlice(inputs.value_caches[layer_idx], v_t,
                                         inputs.cache_position_vt);
    value_cache_new.SetName(absl::StrCat("value_cache_", layer_idx, "_new"));
    out = RbmmGqaAttention(q, key_cache_new, value_cache_new,
                           inputs.attention_mask, inputs.runtime_param,
                           inputs.runtime_param_full, config.n_heads,
                           config.n_kv_groups, config.head_dim,
                           config.max_seq);
  } else {
    value_cache_new = DynamicUpdateSlice(inputs.value_caches[layer_idx], v,
                                         inputs.cache_position);
    value_cache_new.SetName(absl::StrCat("value_cache_", layer_idx, "_new"));
    out =
        GqaAttention(q, key_cache_new, value_cache_new, inputs.attention_mask,
                     config.n_heads, config.n_kv_groups, config.head_dim);
  }

  out = Transpose(out, {0, 2, 1, 3});
  out = Reshape(out, {1, seq, config.qkv_out_dim()});
  out = FullyConnected(out, GetWeight(weights, prefix + ".o_proj.weight"));
  return {out, key_cache_new, value_cache_new};
}

TfTensor FeedForward(const TfTensor& input, const std::string& prefix,
                     const WeightMap& weights) {
  TfTensor gate = FullyConnected(input, GetWeight(weights, prefix + ".gate_proj.weight"));
  TfTensor up = FullyConnected(input, GetWeight(weights, prefix + ".up_proj.weight"));
  return FullyConnected(Mul(Silu(gate), up),
                        GetWeight(weights, prefix + ".down_proj.weight"));
}

// Code predictor incremental stepper — the KV-CACHED in-graph form of the
// production folded MTP (export_mtp_folded.py): step t processes ONE token
// at rope position t, appending per-layer K/V onto growing Concatenation
// caches (all shapes static in the unroll; causal by construction, no mask;
// no DynamicUpdateSlice needed). Replaces the earlier naive full re-forward
// per group, whose ~8.4x extra CP FLOPs put the one-call graph BEHIND the
// production host-loop floor on phone CPUs (see notes/pixel8a-ab.md).
struct CpCacheState {
  // Per layer: [1, kv, t+1, hd] roped keys / values accumulated so far.
  std::vector<TfTensor> k, v;
};

// Per-position rope constants [1, 1, 1, head_dim].
std::pair<TfTensor, TfTensor> ConstRopeAt(int pos, int head_dim, float base) {
  std::vector<float> cos_v(head_dim), sin_v(head_dim);
  int half = head_dim / 2;
  for (int i = 0; i < half; ++i) {
    float freq = std::pow(base, -2.0f * static_cast<float>(i) / head_dim);
    float angle = static_cast<float>(pos) * freq;
    cos_v[i] = cos_v[half + i] = std::cos(angle);
    sin_v[i] = sin_v[half + i] = std::sin(angle);
  }
  TfTensor cos_t({.type = Type::kFP32,
                  .shape = {1, 1, 1, head_dim},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(cos_v)});
  TfTensor sin_t({.type = Type::kFP32,
                  .shape = {1, 1, 1, head_dim},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(sin_v)});
  return {cos_t, sin_t};
}

// One CP step: x [1, 1, emb] at position t -> final-normed hidden [1, 1, emb].
TfTensor CpStep(const TalkerConfig& config, TfTensor x, int t,
                CpCacheState& state, const WeightMap& weights) {
  auto [cos, sin] = ConstRopeAt(t, config.head_dim, config.rope_base);
  TfTensor zero_mask(
      {.type = Type::kFP32,
       .shape = {1, 1, 1, t + 1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
           std::vector<float>(t + 1, 0.0f))});

  for (int i = 0; i < config.cp_layers; ++i) {
    std::string p = absl::StrCat("talker.code_predictor.model.layers.", i);
    TfTensor h = RmsNorm(x, GetWeight(weights, p + ".input_layernorm.weight"),
                         config.rms_norm_eps);
    TfTensor q =
        FullyConnected(h, GetWeight(weights, p + ".self_attn.q_proj.weight"));
    TfTensor k =
        FullyConnected(h, GetWeight(weights, p + ".self_attn.k_proj.weight"));
    TfTensor v =
        FullyConnected(h, GetWeight(weights, p + ".self_attn.v_proj.weight"));
    q = Transpose(Reshape(q, {1, 1, config.n_heads, config.head_dim}),
                  {0, 2, 1, 3});
    k = Transpose(Reshape(k, {1, 1, config.n_kv_groups, config.head_dim}),
                  {0, 2, 1, 3});
    v = Transpose(Reshape(v, {1, 1, config.n_kv_groups, config.head_dim}),
                  {0, 2, 1, 3});
    q = RmsNorm(q, GetWeight(weights, p + ".self_attn.q_norm.weight"),
                config.rms_norm_eps);
    k = RmsNorm(k, GetWeight(weights, p + ".self_attn.k_norm.weight"),
                config.rms_norm_eps);
    q = ApplyRope(q, cos, sin);
    k = ApplyRope(k, cos, sin);

    if (t == 0) {
      state.k.push_back(k);
      state.v.push_back(v);
    } else {
      state.k[i] = Concatenation({state.k[i], k}, /*axis=*/2);
      state.v[i] = Concatenation({state.v[i], v}, /*axis=*/2);
    }

    TfTensor out = GqaAttention(q, state.k[i], state.v[i], zero_mask,
                                config.n_heads, config.n_kv_groups,
                                config.head_dim);
    out = Reshape(Transpose(out, {0, 2, 1, 3}), {1, 1, config.qkv_out_dim()});
    out = FullyConnected(out, GetWeight(weights, p + ".self_attn.o_proj.weight"));
    x = Add(x, out);

    TfTensor f = RmsNorm(
        x, GetWeight(weights, p + ".post_attention_layernorm.weight"),
        config.rms_norm_eps);
    x = Add(x, FeedForward(f, p + ".mlp", weights));
  }
  return RmsNorm(x,
                 GetWeight(weights, "talker.code_predictor.model.norm.weight"),
                 config.rms_norm_eps);
}

// 1-D-indices Gather of one embedding row: ids [1,1] -> [1,1,emb].
TfTensor GatherRow(const TfTensor& table, const TfTensor& id, int emb_dim) {
  TfTensor flat = Reshape(id, {1});
  TfTensor row = Gather(table, flat, /*axis=*/0);
  return Reshape(row, {1, 1, emb_dim});
}

}  // namespace

std::vector<TfTensor> TalkerInputs::AsList(bool is_decode) const {
  std::vector<TfTensor> list;
  if (is_decode) {
    list.push_back(feedback_emb);
    list.push_back(text_track_emb);
  } else {
    list.push_back(embedded_input);
  }
  list.push_back(cb0_bias);
  list.push_back(cache_position);
  list.push_back(rope_cos);
  list.push_back(rope_sin);
  list.push_back(attention_mask);
  if (has_runtime_bmm) {
    list.push_back(runtime_param);
    list.push_back(runtime_param_full);
    list.push_back(cache_position_vt);
  }
  for (size_t i = 0; i < key_caches.size(); ++i) {
    list.push_back(key_caches[i]);
    list.push_back(value_caches[i]);
  }
  return list;
}

std::vector<TfTensor> TalkerOutputs::AsList() const {
  std::vector<TfTensor> list;
  list.push_back(codec_frame);
  list.push_back(cb0_logits);
  list.push_back(frame_feedback_emb);
  for (size_t i = 0; i < key_caches.size(); ++i) {
    list.push_back(key_caches[i]);
    list.push_back(value_caches[i]);
  }
  return list;
}

TalkerInputs MakeTalkerInputs(const TalkerConfig& config, bool is_decode) {
  TalkerInputs inputs;
  int seq = is_decode ? 1 : config.prefill_len;

  if (is_decode) {
    inputs.feedback_emb = TfTensor({.name = "feedback_emb",
                                    .type = Type::kFP32,
                                    .shape = {1, 1, config.emb_dim}});
    inputs.text_track_emb = TfTensor({.name = "text_track_emb",
                                      .type = Type::kFP32,
                                      .shape = {1, 1, config.emb_dim}});
  } else {
    inputs.embedded_input =
        TfTensor({.name = "embedded_input",
                  .type = Type::kFP32,
                  .shape = {1, seq, config.emb_dim}});
  }
  inputs.cb0_bias = TfTensor({.name = "cb0_bias",
                              .type = Type::kFP32,
                              .shape = {1, config.talker_vocab}});
  inputs.cache_position =
      TfTensor({.name = "cache_position", .type = Type::kI32, .shape = {4}});
  inputs.rope_cos = TfTensor({.name = "rope_cos",
                              .type = Type::kFP32,
                              .shape = {1, 1, seq, config.head_dim}});
  inputs.rope_sin = TfTensor({.name = "rope_sin",
                              .type = Type::kFP32,
                              .shape = {1, 1, seq, config.head_dim}});
  inputs.attention_mask = TfTensor({.name = "attention_mask",
                                    .type = Type::kFP32,
                                    .shape = {1, 1, seq, config.max_seq}});
  if (config.use_runtime_bmm) {
    inputs.has_runtime_bmm = true;
    inputs.runtime_param = TfTensor({.name = "runtime_param",
                                     .type = Type::kI32,
                                     .shape = {1, 1, 1, 7}});
    inputs.runtime_param_full = TfTensor({.name = "runtime_param_full",
                                          .type = Type::kI32,
                                          .shape = {1, 1, 1, 7}});
    inputs.cache_position_vt = TfTensor(
        {.name = "cache_position_vt", .type = Type::kI32, .shape = {4}});
  }
  inputs.key_caches.reserve(config.n_layers);
  inputs.value_caches.reserve(config.n_layers);
  for (int i = 0; i < config.n_layers; ++i) {
    inputs.key_caches.push_back(TfTensor(
        {.name = absl::StrCat("key_cache_", i),
         .type = Type::kFP32,
         .shape = {1, config.n_kv_groups, config.max_seq, config.head_dim}}));
    std::vector<int> v_shape =
        config.use_runtime_bmm
            ? std::vector<int>{1, config.n_kv_groups, config.head_dim,
                               config.max_seq}
            : std::vector<int>{1, config.n_kv_groups, config.max_seq,
                               config.head_dim};
    inputs.value_caches.push_back(TfTensor({.name = absl::StrCat(
                                                "value_cache_", i),
                                            .type = Type::kFP32,
                                            .shape = v_shape}));
  }
  return inputs;
}

std::vector<WeightSpec> GetWeightSpecs(const TalkerConfig& config) {
  std::vector<WeightSpec> specs;
  auto add = [&](const std::string& name, const std::vector<int>& shape,
                 float scale = 0.02f) {
    specs.push_back({name, shape, scale});
  };
  auto add_backbone = [&](const std::string& base, int n_layers) {
    for (int i = 0; i < n_layers; ++i) {
      std::string p = absl::StrCat(base, ".layers.", i);
      add(p + ".input_layernorm.weight", {config.emb_dim}, 1.0f);
      add(p + ".self_attn.q_proj.weight", {config.qkv_out_dim(), config.emb_dim});
      add(p + ".self_attn.k_proj.weight", {config.kv_out_dim(), config.emb_dim});
      add(p + ".self_attn.v_proj.weight", {config.kv_out_dim(), config.emb_dim});
      add(p + ".self_attn.o_proj.weight", {config.emb_dim, config.qkv_out_dim()});
      add(p + ".self_attn.q_norm.weight", {config.head_dim}, 1.0f);
      add(p + ".self_attn.k_norm.weight", {config.head_dim}, 1.0f);
      add(p + ".post_attention_layernorm.weight", {config.emb_dim}, 1.0f);
      add(p + ".mlp.gate_proj.weight", {config.hidden_dim, config.emb_dim});
      add(p + ".mlp.up_proj.weight", {config.hidden_dim, config.emb_dim});
      add(p + ".mlp.down_proj.weight", {config.emb_dim, config.hidden_dim});
    }
    add(base + ".norm.weight", {config.emb_dim}, 1.0f);
  };

  // Talker backbone + heads (real checkpoint names/shapes).
  add_backbone("talker.model", config.n_layers);
  add("talker.model.codec_embedding.weight",
      {config.talker_vocab, config.emb_dim});
  add("talker.codec_head.weight", {config.talker_vocab, config.emb_dim});

  // Code predictor: shared 5-layer backbone + per-group embeddings and heads.
  add_backbone("talker.code_predictor.model", config.cp_layers);
  for (int g = 0; g < config.num_sub_groups(); ++g) {
    add(absl::StrCat("talker.code_predictor.model.codec_embedding.", g,
                     ".weight"),
        {config.codec_vocab, config.emb_dim});
    add(absl::StrCat("talker.code_predictor.lm_head.", g, ".weight"),
        {config.codec_vocab, config.emb_dim});
  }
  return specs;
}

WeightMap MakeSyntheticWeights(const TalkerConfig& config, unsigned seed) {
  WeightMap weights;
  unsigned state = seed;
  for (const WeightSpec& spec : GetWeightSpecs(config)) {
    weights[spec.name] =
        SyntheticWeight(spec.name, spec.shape, state, spec.init_scale);
  }
  return weights;
}

TalkerOutputs BuildTalkerStep(const TalkerConfig& config,
                              const TalkerInputs& inputs,
                              const WeightMap& weights, bool is_decode) {
  g_rms_norm_composite = config.use_rms_norm_composite;
  const TfTensor& codec_table =
      GetWeight(weights, "talker.model.codec_embedding.weight");

  // --- Talker input ---
  TfTensor hidden;
  if (is_decode) {
    // Dual-track aggregation in-graph: prev-frame feedback + text-track row.
    hidden = Add(inputs.feedback_emb, inputs.text_track_emb);
  } else {
    hidden = inputs.embedded_input;
  }

  // --- Talker backbone ---
  TalkerOutputs outputs;
  outputs.key_caches.reserve(config.n_layers);
  outputs.value_caches.reserve(config.n_layers);
  for (int i = 0; i < config.n_layers; ++i) {
    std::string p = absl::StrCat("talker.model.layers.", i);
    TfTensor attn_in = RmsNorm(
        hidden, GetWeight(weights, p + ".input_layernorm.weight"),
        config.rms_norm_eps);
    AttentionResult attn = TalkerAttention(config, attn_in, p + ".self_attn",
                                           inputs, i, weights);
    hidden = Add(hidden, attn.output);
    outputs.key_caches.push_back(attn.key_cache);
    outputs.value_caches.push_back(attn.value_cache);

    TfTensor ffn_in = RmsNorm(
        hidden, GetWeight(weights, p + ".post_attention_layernorm.weight"),
        config.rms_norm_eps);
    hidden = Add(hidden, FeedForward(ffn_in, p + ".mlp", weights));
  }
  hidden = RmsNorm(hidden, GetWeight(weights, "talker.model.norm.weight"),
                   config.rms_norm_eps);
  if (!is_decode) {
    hidden = Slice(hidden, {0, config.prefill_len - 1, 0},
                   {1, 1, config.emb_dim});
  }
  // `hidden` is the talker's last hidden state [1, 1, emb].

  // --- cb0: talker codec head + host bias (suppression / penalties) ---
  TfTensor cb0_logits =
      FullyConnected(hidden, GetWeight(weights, "talker.codec_head.weight"));
  outputs.cb0_logits = cb0_logits;
  outputs.cb0_logits.SetName("cb0_logits");
  TfTensor cb0_biased = Add(cb0_logits, Reshape(inputs.cb0_bias,
                                                {1, 1, config.talker_vocab}));
  TfTensor cb0 = ArgMax(cb0_biased, -1, Type::kI32);  // [1, 1]
  TfTensor cb0_emb = GatherRow(codec_table, cb0, config.emb_dim);

  // --- Code predictor, KV-cached incremental unroll (16 steps) ---
  // Mirrors the production folded MTP: step 0 consumes the talker hidden,
  // step 1 the cb0 embed; step t>=1 emits code t-1 via lm_head[t-1] and
  // feeds its per-group embedding into step t+1.
  std::vector<TfTensor> frame_ids;
  frame_ids.reserve(config.num_code_groups);
  frame_ids.push_back(cb0);

  CpCacheState cp_state;
  TfTensor cp_in = hidden;
  TfTensor feedback_sum = cb0_emb;
  for (int t = 0; t < config.num_code_groups; ++t) {
    TfTensor cp_hidden = CpStep(config, cp_in, t, cp_state, weights);
    if (t == 0) {
      cp_in = cb0_emb;
      continue;
    }
    int g = t - 1;
    TfTensor logits = FullyConnected(
        cp_hidden,
        GetWeight(weights,
                  absl::StrCat("talker.code_predictor.lm_head.", g, ".weight")));
    TfTensor id = ArgMax(logits, -1, Type::kI32);  // [1, 1]
    frame_ids.push_back(id);
    TfTensor emb = GatherRow(
        GetWeight(weights,
                  absl::StrCat("talker.code_predictor.model.codec_embedding.",
                               g, ".weight")),
        id, config.emb_dim);
    feedback_sum = Add(feedback_sum, emb);
    if (t + 1 < config.num_code_groups) cp_in = emb;
  }

  outputs.codec_frame =
      Concatenation(absl::MakeSpan(frame_ids), /*axis=*/1);  // [1, 16] kI32
  outputs.codec_frame.SetName("codec_frame");
  outputs.frame_feedback_emb = feedback_sum;
  outputs.frame_feedback_emb.SetName("frame_feedback_emb");
  return outputs;
}

}  // namespace litert::tensor::examples::talker
