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

#ifndef MODELS_OMNIGEN_OMNIGEN2_RUNNER_TEXT_ENCODER_STAGE_H_
#define MODELS_OMNIGEN_OMNIGEN2_RUNNER_TEXT_ENCODER_STAGE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "models/omnigen/omnigen2/runner/io_types.h"
#include "models/omnigen/omnigen2/runner/omnigen2_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/prompt_source.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image::omnigen2 {

// Stage 1 for `OmniGen2`: tokenizes input prompts, applies Qwen2.5-VL ChatML
// framing and right-padding, and runs the text encoder `LiteRtRunner` to
// produce conditioning embeddings (and optional negative prompt embeddings for
// Classifier-Free Guidance).
class OmniGen2TextEncoderStage
    : public SingleThreadedStageWithDeque<OmniGen2TextEncoderOutput> {
 public:
  // Configuration for `OmniGen2TextEncoderStage`.
  // Default token framing encodes the OmniGen2 Qwen2.5-VL ChatML system/user
  // template: `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{prompt}
  // <|im_end|>\n`.
  struct Config {
    // Input buffer indices for the text encoder runner.
    struct InputIndices {
      size_t input_ids = 0;
      size_t attention_mask = 1;
    };

    // Padded sequence length expected by the text encoder model. User prompt
    // tokens exceeding `seq_len - (prefix_token_ids.size() +
    // suffix_token_ids.size())` are truncated while preserving framing tokens.
    int seq_len = 256;
    // Token ID used to right-pad `input_ids` to `seq_len`.
    int32_t pad_token_id = kOmniGen2PadTokenId;
    // Optional text prefix prepended to the user prompt before tokenization.
    std::string prompt_prefix = "user\n";
    // Token IDs prepended before the tokenized prompt.
    std::vector<int32_t> prefix_token_ids = DefaultOmniGen2PrefixTokenIds();
    // Token IDs appended after the tokenized prompt before padding.
    std::vector<int32_t> suffix_token_ids = DefaultOmniGen2SuffixTokenIds();
    // Whether to also encode `negative_prompt` into
    // `OmniGen2TextEncoderOutput::negative_prompt_embeds` for CFG.
    bool encode_negative_prompt = false;
    // Default negative prompt text encoded when `encode_negative_prompt` is
    // true.
    std::string negative_prompt;
    // Input buffer indices for the runner.
    InputIndices input_indices;
  };

  using InputIndices = Config::InputIndices;

  static absl::StatusOr<std::unique_ptr<OmniGen2TextEncoderStage>> Create(
      Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
      std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner, const Config& config);

  static absl::StatusOr<std::unique_ptr<OmniGen2TextEncoderStage>> Create(
      Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
      std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner);

  ~OmniGen2TextEncoderStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  OmniGen2TextEncoderStage(
      Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
      std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      std::vector<TensorBuffer> input_buffers,
      std::vector<TensorBuffer> output_buffers, Config config);

  Stage<Text2ImagePrompt>& prompt_source_;
  const std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer_;
  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
  const Config config_;
};

}  // namespace litert::omni::text2image::omnigen2

#endif  // MODELS_OMNIGEN_OMNIGEN2_RUNNER_TEXT_ENCODER_STAGE_H_
