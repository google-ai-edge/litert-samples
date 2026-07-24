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

// Talker step-graph — Stage 2 (structure aligned with the real
// Qwen/Qwen3-TTS-12Hz-0.6B-Base checkpoint; synthetic weights until the
// safetensors loader is wired).
//
// One decode signature = one full 12Hz frame, all in-graph:
//   talker backbone (28L, fixed-size KV cache via DynamicUpdateSlice at a
//   runtime position) -> cb0 logits (+ host-provided additive bias for
//   suppression/penalties) -> greedy cb0 -> the 5-layer code predictor
//   unrolled in-graph as a KV-CACHED 16-step incremental pass (per-layer
//   growing Concatenation caches, one token per step — the in-graph form of
//   the production folded MTP) -> frame codes [1,16] -> next-frame feedback
//   embedding (codec_emb[cb0] + sum of sub embeds) emitted as an output for
//   rebinding into the next call.
//
// vs the existing production-style pipeline (hostloop_e2e): talker step and
// folded-MTP are separate graphs and the dual-track frame aggregation
// (codec_emb[cb0] + sum mtp_emb + text track) happens on the host with numpy
// tables. Here talker + predictor + aggregation are ONE signature call; the
// host only supplies the text-track row, the cb0 bias vector, rope rows, and
// the cache position.
//
// The sequencing loop itself stays host/Engine-side (.tflite has no control
// flow); no Custom() anywhere. Greedy decode in-graph; sampling would take a
// host noise input (no RNG op — known API feedback item).

#ifndef SPEECHLM_STAGE1_TALKER_STEP_TALKER_GRAPH_H_
#define SPEECHLM_STAGE1_TALKER_STEP_TALKER_GRAPH_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::talker {

using TfTensor = ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>;
using WeightMap = absl::flat_hash_map<std::string, TfTensor>;

struct TalkerInputs {
  // prefill: [1, prefill_len, emb_dim] kFP32 — precomputed dual-track prompt
  // embeddings (text embedding + projection stay host-side in this stage).
  TfTensor embedded_input;
  // decode: [1, 1, emb_dim] kFP32 — previous frame's feedback embedding
  // (rebound from the previous call's frame_feedback_emb output).
  TfTensor feedback_emb;
  // decode: [1, 1, emb_dim] kFP32 — text-track row (trailing text embed or
  // tts_pad), host-provided per step.
  TfTensor text_track_emb;
  // [1, talker_vocab] kFP32 additive bias on cb0 logits — host encodes
  // suppression of non-codec ids, EOS gating and repetition penalties here.
  TfTensor cb0_bias;
  // [4] kI32 {0, 0, position, 0} — runtime KV write position.
  TfTensor cache_position;
  // [1, 1, seq, head_dim] kFP32 each.
  TfTensor rope_cos;
  TfTensor rope_sin;
  // decode: [1, 1, 1, max_seq]; prefill: [1, 1, prefill_len, max_seq].
  TfTensor attention_mask;
  // Per layer: [1, n_kv_groups, max_seq, head_dim] kFP32. With
  // use_runtime_bmm the VALUE caches are stored transposed,
  // [1, n_kv_groups, head_dim, max_seq], so valid positions sit on channels
  // for the src-bounded scores x V composite.
  std::vector<TfTensor> key_caches;
  std::vector<TfTensor> value_caches;
  // use_runtime_bmm only: odml.runtime_bmm control tensors, int32 [1,1,1,7]
  // with element 2 = active token count (runtime_param) / max_seq
  // (runtime_param_full, keeps the QK side unbounded), plus the value-cache
  // write position as [4] {0,0,0,position} (cache_position_vt).
  TfTensor runtime_param;
  TfTensor runtime_param_full;
  TfTensor cache_position_vt;
  bool has_runtime_bmm = false;

  std::vector<TfTensor> AsList(bool is_decode) const;
};

struct TalkerOutputs {
  // [1, num_code_groups] kI32 — cb0 + 15 sub codes for this frame.
  TfTensor codec_frame;
  // [1, 1, talker_vocab] kFP32 — cb0 logits BEFORE the host bias (the
  // in-graph greedy pick uses the biased ones). For host-side sampling and
  // the numerical equivalence check.
  TfTensor cb0_logits;
  // [1, 1, emb_dim] kFP32 — codec_emb[cb0] + sum(sub embeds); feed back as
  // feedback_emb next call (text track is added in-graph there).
  TfTensor frame_feedback_emb;
  // Updated fixed-size caches, same shapes as inputs.
  std::vector<TfTensor> key_caches;
  std::vector<TfTensor> value_caches;

  std::vector<TfTensor> AsList() const;
};

TalkerInputs MakeTalkerInputs(const TalkerConfig& config, bool is_decode);

// Every weight tensor the step graphs consume, under the REAL checkpoint
// names (talker.model.layers.N..., talker.codec_head.weight,
// talker.code_predictor.model.layers.N..., talker.code_predictor.lm_head.N...)
// with the expected shape. Single source of truth for both the synthetic
// generator and the safetensors loader.
struct WeightSpec {
  std::string name;
  std::vector<int> shape;
  float init_scale;  // synthetic init only: 1.0 for norms, 0.02 elsewhere
};
std::vector<WeightSpec> GetWeightSpecs(const TalkerConfig& config);

// Synthetic weights for all GetWeightSpecs entries (safetensors swap is a
// pure loader change).
WeightMap MakeSyntheticWeights(const TalkerConfig& config, unsigned seed);

TalkerOutputs BuildTalkerStep(const TalkerConfig& config,
                              const TalkerInputs& inputs,
                              const WeightMap& weights, bool is_decode);

}  // namespace litert::tensor::examples::talker

#endif  // SPEECHLM_STAGE1_TALKER_STEP_TALKER_GRAPH_H_
