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

#ifndef MODELS_DEEPSEEK_JANUS_PRO_RUNNER_TEXT_STAGE_H_
#define MODELS_DEEPSEEK_JANUS_PRO_RUNNER_TEXT_STAGE_H_

#include <memory>
#include <vector>

#include "models/deepseek/janus_pro/runner/io_types.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
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
namespace janus_pro {

// Resolves `token_str` through `tokenizer.TokenToId`, falling back to
// `fallback_id` when the tokenizer does not register it as a single token.
int ResolveSpecialTokenId(support::Tokenizer& tokenizer,
                          absl::string_view token_str, int fallback_id);

// Splits `text` around Janus-Pro chat and modality control tokens, mapping
// special tokens via `tokenizer.TokenToId` (with fallback IDs) and delegating
// remaining text spans to `tokenizer.TextToTokenIds`.
absl::StatusOr<std::vector<int>> TokenizeControlText(
    support::Tokenizer& tokenizer, absl::string_view text);

// Validates that `text_embedder_runner` input and output buffers match
// `[..., max_batch]` Int32 and `[..., max_batch, llm_hidden_size]` Float32,
// returning `max_batch` on success.
absl::StatusOr<int> ValidateTextEmbedderBuffers(
    absl::Span<const TensorBuffer> input_buffers,
    absl::Span<const TensorBuffer> output_buffers, int llm_hidden_size,
    absl::string_view caller_name);

// Validates token IDs against `[0, vocab_size)` and looks up LLM embeddings for
// `token_ids` using `runner` in batches of `max_batch`.
absl::StatusOr<std::vector<float>> EmbedTokenSequence(
    absl::Span<const int> token_ids, int vocab_size, int hidden_dim,
    int max_batch, LiteRtRunner& runner, absl::Span<TensorBuffer> input_buffers,
    absl::Span<TensorBuffer> output_buffers, absl::string_view caller_name);

}  // namespace janus_pro

// Text and turn-control embedding stage for Janus-Pro:
// - Consumes `Sequenced<TextInput>`, `Sequenced<EndOfInputTurn>`, and
//   `Sequenced<EndOfInput>` from `JanusProInputFanOutStage` in ascending
//   `seqnum` order among its input queues.
// - Escapes control-token delimiters in untrusted user text before delegating
//   to `Tokenizer::TextToTokenIds()` and embeds token IDs into LLM hidden
//   dimension space via `text_embedder_runner`.
class JanusProTextStage
    : public SingleThreadedStageWithDeque<Sequenced<JanusProEmbeddingSegment>> {
 public:
  // Creates a `JanusProTextStage` consuming sequenced text, end-of-turn, and
  // end-of-input items and embedding tokens via `text_embedder_runner`.
  static absl::StatusOr<std::unique_ptr<JanusProTextStage>> Create(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
          end_of_input_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      const JanusProModelConfig& config = {});

  ~JanusProTextStage() override;

  // Tokenizes `text` recognizing Janus-Pro control tokens and embeds the
  // resulting token IDs into LLM hidden dimension space.
  absl::StatusOr<JanusProEmbeddingSegment> EmbedControlText(
      absl::string_view text);

  // Embeds `token_ids` directly via `text_embedder_runner_`.
  absl::StatusOr<std::vector<float>> EmbedTokens(
      absl::Span<const int> token_ids);

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  JanusProTextStage(
      FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
          end_of_turn_queue,
      FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
          end_of_input_queue,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      std::vector<TensorBuffer> text_emb_input_buffers,
      std::vector<TensorBuffer> text_emb_output_buffers, int max_batch,
      const JanusProModelConfig& config);

  // Tokenizes untrusted user prompt text and projects it to LLM embeddings.
  absl::StatusOr<JanusProEmbeddingSegment> ProcessText(absl::string_view text);

  FanOutOutputQueue<OmniSession::Input, TextInput>& text_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>& end_of_turn_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInput>& end_of_input_queue_;
  JanusProModelConfig config_;
  int hidden_dim_;
  int max_batch_;

  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtRunner> text_embedder_runner_;
  std::vector<TensorBuffer> text_emb_input_buffers_;
  std::vector<TensorBuffer> text_emb_output_buffers_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_RUNNER_TEXT_STAGE_H_
