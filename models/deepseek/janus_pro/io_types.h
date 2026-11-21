// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MODELS_DEEPSEEK_JANUS_PRO_IO_TYPES_H_
#define MODELS_DEEPSEEK_JANUS_PRO_IO_TYPES_H_

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace litert::omni::any2any {

// Intermediate modality embedding representation produced by modality stages
// (`JanusProTextStage` and `JanusProImageStage`) for a single sequenced input
// item.
struct JanusProEmbeddingSegment {
  // Flattened token embeddings in LLM hidden dimension space with shape
  // `[num_tokens, llm_hidden_size]`.
  std::vector<float> embeddings;
  // Corresponding token IDs for text/control tokens, or
  // `janus_pro::kImageEmbedPlaceholderTokenId` (`-1`) for continuous vision
  // embeddings.
  std::vector<int> token_ids;
  // Number of embedding tokens represented in `embeddings`.
  int num_tokens = 0;
  // Raw user text associated with this segment (preserved for diagnostics and
  // prompt inspection).
  std::string raw_text;
  // Whether this segment marks the end of the current user input turn
  // (`EndOfInputTurn` or `EndOfInput`).
  bool is_turn_end = false;
  // Whether this segment was produced from an explicit `EndOfInputTurn` item.
  bool is_explicit_eot = false;
};

// Output of Stage 1 (`JanusProFrontendStage`): multimodal prompt embeddings in
// LLM hidden dimension space (`[prompt_len, llm_hidden_size]`), plus optional
// unconditional CFG prompt embeddings for visual generation mode.
struct JanusProFrontendChunk {
  // Flattened conditional multimodal prompt embeddings in LLM hidden dimension
  // space with shape `[prompt_len, llm_hidden_size]`.
  std::vector<float> cond_embeddings;
  // Flattened unconditional CFG prompt embeddings in LLM hidden dimension space
  // with shape `[uncond_prompt_len, llm_hidden_size]` (populated when
  // `generate_image == true`).
  std::vector<float> uncond_embeddings;
  // Number of tokens in `cond_embeddings`.
  int prompt_len = 0;
  // Number of tokens in `uncond_embeddings`.
  int uncond_prompt_len = 0;
  // Concatenated token IDs from text and control token segments in the
  // conditional prompt.
  std::vector<int> token_ids;
  // Concatenated token IDs in the unconditional CFG prompt.
  std::vector<int> uncond_token_ids;
  // Whether this turn runs in autoregressive visual generation mode (`Text ->
  // 576 VQ codes -> ImageOutput`) instead of multimodal understanding mode.
  bool generate_image = false;
};

// Grid of discrete VQ-16 codebook token IDs (`576` tokens for `24x24` patches)
// produced by `JanusProLlmStage` in visual generation mode and consumed by
// `JanusProImageGenStage`.
struct JanusProVqCodeGrid {
  // Generated VQ-16 codebook indices of length `num_image_tokens` (`576`),
  // each in `[0, vq_codebook_size)`.
  std::vector<int32_t> vq_codes;
  // Target decoded image width in pixels (`384`).
  int image_width = 384;
  // Target decoded image height in pixels (`384`).
  int image_height = 384;
};

// Output item emitted by `JanusProLlmStage` to `JanusProImageGenStage`:
// - `std::string`: Generated assistant text response in understanding mode.
// - `JanusProVqCodeGrid`: Generated VQ-16 code grid in visual generation mode.
struct JanusProLlmOutput {
  std::variant<std::string, JanusProVqCodeGrid> payload;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_IO_TYPES_H_
