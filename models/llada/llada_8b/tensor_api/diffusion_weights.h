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

// Real-checkpoint weight loading for the diffusion denoise-step graph —
// Stage 2 (LLaDA-8B-Base).
//
// Maps the checkpoint's OLMo-style tensor names
// (model.transformer.blocks.N.{q,k,v}_proj / attn_out / ff_proj / up_proj /
// ff_out, wte, ln_f, untied model-level ff_out head) onto the graph's
// LLaMA-style names (model.layers.N.self_attn.* / mlp.*, embed_tokens,
// lm_head). Handles a sharded checkpoint directory (model-XXXXX-of-XXXXX
// .safetensors) by probing each shard per tensor.
//
// config.n_layers may be smaller than the checkpoint's 32 (loads a prefix of
// the stack) and config.vocab smaller than 126464 (loads the first vocab rows
// of wte / the head) — both for cheap loader/graph equivalence runs against
// the numpy reference before committing to the full model.

#ifndef SPEECHLM_STAGE1_DIFFUSION_STEP_DIFFUSION_WEIGHTS_H_
#define SPEECHLM_STAGE1_DIFFUSION_STEP_DIFFUSION_WEIGHTS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "models/llada/llada_8b/tensor_api/diffusion_graph.h"

namespace litert::tensor::examples::diffusion {

// kInt4 quantizes the 2-D matmul weights (q/k/v/o, SwiGLU, lm_head) to
// blockwise-32 int4 with fp16 scales — the same layout as the talker; the
// embedding (Gather source) and norms stay fp32.
enum class WeightQuantMode { kNone, kInt4, kInt8 };

absl::StatusOr<WeightMap> LoadLladaWeights(
    const DiffusionConfig& config, const std::string& checkpoint_dir,
    WeightQuantMode quant_mode = WeightQuantMode::kNone);

}  // namespace litert::tensor::examples::diffusion

#endif  // SPEECHLM_STAGE1_DIFFUSION_STEP_DIFFUSION_WEIGHTS_H_
