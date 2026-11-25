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

#ifndef MODELS_LFM_LFM2_5_AUDIO_RUNNER_TEXT_STAGE_H_
#define MODELS_LFM_LFM2_5_AUDIO_RUNNER_TEXT_STAGE_H_

#include <memory>
#include <vector>

#include "models/lfm/lfm2_5_audio/runner/io_types.h"
#include "models/lfm/lfm2_5_audio/runner/lfm2_audio_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace lfm2_audio {

// Validates that every token ID in `token_ids` lies in `[0, vocab_size)` and
// embeds the sequence with `runner` using pre-allocated `input_buffers` and
// `output_buffers`.
//
// Returns flattened `[token_ids.size(), hidden_dim]` embeddings, or an error
// status if any token ID is out of range or `runner` is null / fails.
absl::StatusOr<std::vector<float>> EmbedTokenSequence(
    absl::Span<const int> token_ids, int vocab_size, int hidden_dim,
    LiteRtRunner* absl_nullable runner,
    std::vector<TensorBuffer>& input_buffers,
    std::vector<TensorBuffer>& output_buffers, absl::string_view caller_name);

}  // namespace lfm2_audio

// Tokenizes trusted control templates `text` while preserving LFM2.5-Audio
// special tokens (`<|pad|>`, `<|startoftext|>`, `<|im_start|>`, `<|im_end|>`,
// `<|audio_start|>`, `<|text_end|>`). Untrusted user text is tokenized
// separately in `Lfm2AudioTextStage` with `<|` splitting to prevent special
// token injection.
//
// Args:
//   tokenizer: Tokenizer instance.
//   text: Input string possibly containing special token literals.
//   config: LFM2.5-Audio model configuration with special token IDs.
//
// Returns:
//   Token ID sequence for `text`.
absl::StatusOr<std::vector<int>> TokenizeLfm2AudioTextWithSpecialTokens(
    support::Tokenizer& tokenizer, absl::string_view text,
    const Lfm2AudioModelConfig& config);

// Text tokenization and embedding stage for `LiquidAI/LFM2.5-Audio-1.5B`.
// Rejects any out-of-range token ID (`< 0` or `>= config.llm_vocab_size`) with
// `absl::InvalidArgumentError` without clamping.
class Lfm2AudioTextStage : public SingleThreadedStageWithDeque<
                               Sequenced<Lfm2AudioEmbeddingSegment>> {
 public:
  // Creates an `Lfm2AudioTextStage`.
  //
  // Args:
  //   text_queue: Non-null queue providing `Sequenced<TextInput>` items.
  //   end_of_turn_queue: Non-null queue providing
  //     `Sequenced<EndOfInputTurn>` items.
  //   tokenizer: Non-null tokenizer.
  //   text_embedder_runner: Optional runner executing `tf_lite_text_embedder`.
  //   config: LFM2.5-Audio model configuration.
  //
  // Returns:
  //   Initialized `Lfm2AudioTextStage`, or an error status on invalid config.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioTextStage>> Create(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner =
          nullptr,
      const Lfm2AudioModelConfig& config = {});

  ~Lfm2AudioTextStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  Lfm2AudioTextStage(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
      std::vector<TensorBuffer> embed_input_buffers,
      std::vector<TensorBuffer> embed_output_buffers,
      const Lfm2AudioModelConfig& config);

  absl::StatusOr<std::vector<float>> EmbedTokens(
      absl::Span<const int> token_ids);

  FanOutOutputQueue<OmniSession::Input, TextInput>& text_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>& end_of_turn_queue_;
  Lfm2AudioModelConfig config_;
  int hidden_dim_;
  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtRunner> text_embedder_runner_;
  std::vector<TensorBuffer> embed_input_buffers_;
  std::vector<TensorBuffer> embed_output_buffers_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_RUNNER_TEXT_STAGE_H_
