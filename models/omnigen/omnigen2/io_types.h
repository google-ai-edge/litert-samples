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

#ifndef MODELS_OMNIGEN_OMNIGEN2_IO_TYPES_H_
#define MODELS_OMNIGEN_OMNIGEN2_IO_TYPES_H_

#include <vector>

#include "omni/base/io_types.h"

namespace litert::omni::text2image::omnigen2 {

// Conditioning embeddings produced by `OmniGen2TextEncoderStage`, preserving
// generation metadata and active token lengths for conditional and optional
// unconditional (CFG) branches.
struct OmniGen2TextEncoderOutput {
  ImageGenInputMetadata metadata;
  std::vector<float> prompt_embeds;
  // Number of active (unpadded) tokens in `prompt_embeds` (`1 .. seq_len`).
  int active_seq_len = 0;
  // Optional negative/unconditional prompt embeddings for Classifier-Free
  // Guidance when `encode_negative_prompt` is enabled.
  std::vector<float> negative_prompt_embeds;
  // Number of active (unpadded) tokens in `negative_prompt_embeds`.
  int negative_active_seq_len = 0;
};

// Output of `OmniGen2DenoiserStage`: denoised 2x2-patchified latents of shape
// `[1, grid_h * grid_w, packed_ch]` after FlowMatch Euler integration.
struct OmniGen2DenoiserOutput {
  ImageGenInputMetadata metadata;
  std::vector<float> packed_latents;
};

}  // namespace litert::omni::text2image::omnigen2

#endif  // MODELS_OMNIGEN_OMNIGEN2_IO_TYPES_H_
