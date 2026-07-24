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

// Real-checkpoint weight loading for the talker step-graph.
//
// Loads Qwen/Qwen3-TTS-12Hz-0.6B-Base model.safetensors (bf16 -> fp32) into
// the WeightMap. Only the tensors the step graphs consume are loaded (talker
// backbone + codec embedding/head + code predictor); text_embedding,
// text_projection and the speaker encoder stay host-side.
//
// Reuses the gemma3 example's SafetensorLoader but undoes its Gemma-only
// RMSNorm convention: that loader adds +1.0 to every tensor named
// *layernorm* / *norm.weight (Gemma stores w-1); Qwen3 norms use the raw
// weight, so the offset is subtracted back here.

#ifndef SPEECHLM_STAGE1_TALKER_STEP_TALKER_WEIGHTS_H_
#define SPEECHLM_STAGE1_TALKER_STEP_TALKER_WEIGHTS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_config.h"
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_graph.h"

namespace litert::tensor::examples::talker {

// Optional in-process weight quantization applied while loading: per-channel
// symmetric (zero_point 0, dim 0) int8 (±127) or int4 (±7, bit-packed kI4)
// on the 2-D matmul weights (q/k/v/o, mlp, codec_head, lm_heads). Norms and
// embedding tables (Gather sources) stay fp32.
enum class WeightQuantMode { kNone, kInt8, kInt4 };

// If `prequant_path` is non-empty, the quantizable 2-D matmul weights are
// read pre-quantized from that companion safetensors (`<name>` I8 codes in
// [-7,7] + `<name>.scale` F32 blockwise-32 scales — the output format of
// verify/gptq_quantize.py) instead of being quantized in-process;
// `quant_mode` then only applies to nothing (all quantizables must be in
// the companion). Norms/embeddings still come from `path`.
absl::StatusOr<WeightMap> LoadCheckpointWeights(
    const TalkerConfig& config, const std::string& path,
    WeightQuantMode quant_mode = WeightQuantMode::kNone,
    const std::string& prequant_path = "");

}  // namespace litert::tensor::examples::talker

#endif  // SPEECHLM_STAGE1_TALKER_STEP_TALKER_WEIGHTS_H_
