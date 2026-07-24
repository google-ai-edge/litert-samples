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

// Diffusion-LM denoise-step scaffold — Stage 0 (synthetic weights).
//
// One masked-diffusion denoise step (LLaDA-style) authored on the LiteRT
// Tensor API TfLite backend. The whole per-step schedule is IN-GRAPH:
// bidirectional transformer over the full sequence (no causal mask, no KV
// cache) -> logits -> greedy predictions + confidence -> pick the top-k most
// confident still-masked positions (TopK + OneHot + SelectV2 — no ScatterNd
// needed) -> emit the updated token ids. The host loop only feeds ids back
// for T = masked/k steps.
//
// Everything uses ops with tflite emission today; no Custom().

#ifndef SPEECHLM_STAGE1_DIFFUSION_STEP_DIFFUSION_GRAPH_H_
#define SPEECHLM_STAGE1_DIFFUSION_STEP_DIFFUSION_GRAPH_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::diffusion {

using TfTensor = ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>;
using WeightMap = absl::flat_hash_map<std::string, TfTensor>;

struct DiffusionConfig {
  // Backbone (Qwen3-0.6B-shaped placeholder; LLaDA-style models are larger —
  // Stage 1 aligns dims with the chosen checkpoint).
  int emb_dim = 1024;
  int n_layers = 28;
  int n_heads = 16;
  int n_kv_groups = 8;
  int head_dim = 128;
  int hidden_dim = 3072;
  float rms_norm_eps = 1e-6f;
  float rope_base = 1000000.0f;
  // Qwen3 applies per-head RMSNorm to q/k; LLaMA/OLMo-family (LLaDA) does not.
  bool use_qk_norm = true;
  // Emit RMS norms as the odml.rms_norm composite (fused ML Drift kernels;
  // decomposition = the raw math, so CPU execution is unchanged).
  bool use_rms_norm_composite = false;
  // Emit the attention core as odml.scaled_dot_product_attention (BSND
  // inputs, no mask — bidirectional). Decomposition = the same GqaAttention
  // fold as the raw path, so CPU execution is numerically identical.
  // MHA (n_kv_groups == n_heads) presents plain BSND. GQA presents the
  // group-fold as batched MHA ([G, g*S, 1, D] q vs [G, S, 1, D] k/v):
  // Metal CLAIMS un-folded grouped K/V at q_seq>1 but computes wrong
  // results (2026-07-22 probe: every generated id differs from the CPU
  // reference while raw GPU matches it exactly), so the un-folded form is
  // kept only behind sdpa_nofold as the repro.
  bool use_sdpa_composite = false;
  // Repro switch for the wrong-result claim above: emit GQA sdpa WITHOUT
  // the pre-fold. Ignored for MHA shapes (identical to use_sdpa_composite).
  bool sdpa_skip_gqa_fold = false;
  // Debug: run the folded-sdpa body as plain ops (no composite wrapper).
  bool sdpa_fold_direct = false;

  int seq_len = 256;    // full (fixed) generation window
  int vocab = 32000;    // placeholder vocab
  int mask_id = 31999;  // [MASK] token id (placeholder)
  int unmask_k = 8;     // tokens revealed per step (static per signature)

  int qkv_out_dim() const { return n_heads * head_dim; }
  int kv_out_dim() const { return n_kv_groups * head_dim; }
};

struct DiffusionInputs {
  TfTensor token_ids;  // [1, seq_len] kI32, mask_id at still-masked positions
  TfTensor rope_cos;   // [1, 1, seq_len, head_dim]
  TfTensor rope_sin;   // [1, 1, seq_len, head_dim]

  std::vector<TfTensor> AsList() const;
};

struct DiffusionOutputs {
  TfTensor new_token_ids;  // [1, seq_len] kI32, k more positions revealed
  TfTensor masked_count;   // [1] kI32 — remaining masked positions (host stop)

  std::vector<TfTensor> AsList() const;
};

DiffusionInputs MakeDiffusionInputs(const DiffusionConfig& config);

WeightMap MakeSyntheticWeights(const DiffusionConfig& config, unsigned seed);

// Traces one denoise step over `inputs`.
DiffusionOutputs BuildDenoiseStep(const DiffusionConfig& config,
                                  const DiffusionInputs& inputs,
                                  const WeightMap& weights);

}  // namespace litert::tensor::examples::diffusion

#endif  // SPEECHLM_STAGE1_DIFFUSION_STEP_DIFFUSION_GRAPH_H_
