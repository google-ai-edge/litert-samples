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

#ifndef MODELS_LFM_LFM2_5_AUDIO_IO_TYPES_H_
#define MODELS_LFM_LFM2_5_AUDIO_IO_TYPES_H_

#include <vector>

#include "models/lfm/lfm2_5_audio/lfm2_audio_model_config.h"

namespace litert::omni::any2any {

// Represents a processed multimodal input segment or assembled turn prompt for
// `LiquidAI/LFM2.5-Audio-1.5B`.
struct Lfm2AudioEmbeddingSegment {
  // Flattened continuous embeddings of shape `[num_tokens, llm_hidden_size]`.
  std::vector<float> embeddings;

  // Token IDs (`>= 0` for text/control tokens, or
  // `config.audio_embed_placeholder_token_id` for continuous `AUDIO_IN`
  // embeddings).
  std::vector<int> token_ids;

  // Number of token positions in this segment (`token_ids.size()`).
  int num_tokens = 0;

  // True when this segment marks `EndOfInputTurn`.
  bool is_end_of_turn = false;
};

// Assembled LFM2.5-Audio turn prompt produced by `Lfm2AudioFrontendStage`.
struct Lfm2AudioFrontendOutput {
  // Flattened prompt embeddings of shape `[prompt_len, llm_hidden_size]`.
  std::vector<float> prompt_embeddings;

  // Prompt token IDs (`>= 0` for text/special tokens, or
  // `config.audio_embed_placeholder_token_id` for `AUDIO_IN` frames).
  std::vector<int> prompt_token_ids;

  // Total prompt token count.
  int prompt_len = 0;

  // Whether speech generation (`AUDIO_OUT`) is enabled for this turn.
  bool generate_audio = true;

  // Resolved generation mode (`kInterleaved` or `kSequential`) for this turn.
  Lfm2AudioGenerationMode generation_mode =
      Lfm2AudioGenerationMode::kInterleaved;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_IO_TYPES_H_
