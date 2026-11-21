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

#include "models/deepseek/janus_pro/text_stage.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/io_types.h"
#include "models/deepseek/janus_pro/janus_pro_model_config.h"
#include "models/deepseek/janus_pro/stage_utils.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace {

struct SpecialTokenEntry {
  absl::string_view literal;
  int fallback_id;
};

constexpr std::array<SpecialTokenEntry, 8> kSpecialTokens = {{
    {janus_pro::kBosToken, janus_pro::kBosTokenId},
    {janus_pro::kEosToken, janus_pro::kEosTokenId},
    {janus_pro::kPadToken, janus_pro::kPadTokenId},
    {janus_pro::kBeginOfImageToken, janus_pro::kBeginOfImageTokenId},
    {janus_pro::kEndOfImageToken, janus_pro::kEndOfImageTokenId},
    {janus_pro::kImagePlaceholderToken, janus_pro::kImagePlaceholderTokenId},
    {janus_pro::kUserToken, janus_pro::kUserTokenId},
    {janus_pro::kAssistantToken, janus_pro::kAssistantTokenId},
}};

// Delimiter prefixes that could match Janus-Pro `added_tokens` control IDs if
// passed verbatim inside untrusted user prompts.
constexpr std::array<absl::string_view, 5> kControlTokenPrefixes = {{
    "<|",
    "<｜",
    janus_pro::kBeginOfImageToken,
    janus_pro::kEndOfImageToken,
    janus_pro::kImagePlaceholderToken,
}};

absl::Status ValidateTokenIdsInVocab(absl::Span<const int> token_ids,
                                     int vocab_size,
                                     absl::string_view caller_name) {
  if (vocab_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        caller_name, ": llm_vocab_size must be positive, got ", vocab_size));
  }
  for (size_t i = 0; i < token_ids.size(); ++i) {
    const int token_id = token_ids[i];
    if (token_id < 0 || token_id >= vocab_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          caller_name, ": token_id at index ", i, " (", token_id,
          ") is out of valid vocabulary range [0, ", vocab_size, ")."));
    }
  }
  return absl::OkStatus();
}

// Tokenizes untrusted user text without letting `HuggingFaceTokenizer` match
// Janus-Pro control `added_tokens` as raw special token IDs. Splitting
// immediately after `<` breaks every control token literal across two
// tokenization pieces while leaving ordinary text untouched in a single call.
absl::StatusOr<std::vector<int>> TokenizeUntrustedText(
    support::Tokenizer& tokenizer, absl::string_view text) {
  std::vector<int> token_ids;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t next_delim = absl::string_view::npos;
    for (absl::string_view prefix : kControlTokenPrefixes) {
      const size_t found = text.find(prefix, pos);
      if (found != absl::string_view::npos &&
          (next_delim == absl::string_view::npos || found < next_delim)) {
        next_delim = found;
      }
    }
    if (next_delim == absl::string_view::npos) {
      ABSL_ASSIGN_OR_RETURN(support::TokenIds part_ids,
                            tokenizer.TextToTokenIds(text.substr(pos)));
      token_ids.insert(token_ids.end(), part_ids.begin(), part_ids.end());
      break;
    }
    // Include `<` in the current piece and leave the next byte at the start of
    // the following piece so neither piece contains a control token literal.
    ABSL_ASSIGN_OR_RETURN(
        support::TokenIds part_ids,
        tokenizer.TextToTokenIds(text.substr(pos, next_delim - pos + 1)));
    token_ids.insert(token_ids.end(), part_ids.begin(), part_ids.end());
    pos = next_delim + 1;
  }
  return token_ids;
}

}  // namespace

namespace janus_pro {

int ResolveSpecialTokenId(support::Tokenizer& tokenizer,
                          absl::string_view token_str, int fallback_id) {
  absl::StatusOr<int> tok_id = tokenizer.TokenToId(token_str);
  return tok_id.ok() ? *tok_id : fallback_id;
}

absl::StatusOr<std::vector<int>> TokenizeControlText(
    support::Tokenizer& tokenizer, absl::string_view text) {
  std::vector<int> all_tokens;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t next_special_pos = absl::string_view::npos;
    const SpecialTokenEntry* matched = nullptr;

    for (const SpecialTokenEntry& entry : kSpecialTokens) {
      const size_t found = text.find(entry.literal, pos);
      if (found != absl::string_view::npos &&
          (next_special_pos == absl::string_view::npos ||
           found < next_special_pos)) {
        next_special_pos = found;
        matched = &entry;
      }
    }

    if (matched == nullptr) {
      ABSL_ASSIGN_OR_RETURN(support::TokenIds part_ids,
                            tokenizer.TextToTokenIds(text.substr(pos)));
      all_tokens.insert(all_tokens.end(), part_ids.begin(), part_ids.end());
      break;
    }

    if (next_special_pos > pos) {
      ABSL_ASSIGN_OR_RETURN(
          support::TokenIds part_ids,
          tokenizer.TextToTokenIds(text.substr(pos, next_special_pos - pos)));
      all_tokens.insert(all_tokens.end(), part_ids.begin(), part_ids.end());
    }

    all_tokens.push_back(ResolveSpecialTokenId(tokenizer, matched->literal,
                                               matched->fallback_id));
    pos = next_special_pos + matched->literal.size();
  }
  return all_tokens;
}

absl::StatusOr<int> ValidateTextEmbedderBuffers(
    absl::Span<const TensorBuffer> input_buffers,
    absl::Span<const TensorBuffer> output_buffers, int llm_hidden_size,
    absl::string_view caller_name) {
  if (llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        caller_name, ": config.llm_hidden_size must be positive."));
  }
  if (input_buffers.empty() || output_buffers.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat(caller_name,
                     ": text_embedder_runner must provide non-empty input and "
                     "output buffers."));
  }
  LITERT_ASSIGN_OR_RETURN(RankedTensorType in_type,
                          input_buffers[0].TensorType());
  if (in_type.ElementType() != ElementType::Int32) {
    return absl::InvalidArgumentError(
        absl::StrCat(caller_name,
                     ": text_embedder input buffer must have Int32 element "
                     "type."));
  }
  LITERT_ASSIGN_OR_RETURN(size_t input_elements,
                          in_type.Layout().NumElements());
  if (input_elements == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(caller_name,
                     ": text_embedder input buffer must have at least 1 "
                     "element."));
  }
  ABSL_RETURN_IF_ERROR(ValidateTensorBuffer(
      output_buffers[0], ElementType::Float32,
      input_elements * static_cast<size_t>(llm_hidden_size),
      absl::StrCat(caller_name, " text_embedder output")));
  return static_cast<int>(input_elements);
}

absl::StatusOr<std::vector<float>> EmbedTokenSequence(
    absl::Span<const int> token_ids, int vocab_size, int hidden_dim,
    int max_batch, LiteRtRunner& runner, absl::Span<TensorBuffer> input_buffers,
    absl::Span<TensorBuffer> output_buffers, absl::string_view caller_name) {
  ABSL_RETURN_IF_ERROR(
      ValidateTokenIdsInVocab(token_ids, vocab_size, caller_name));
  if (token_ids.empty()) {
    return std::vector<float>{};
  }
  if (hidden_dim <= 0 || max_batch <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(caller_name, ": hidden_dim and max_batch must be > 0."));
  }
  std::vector<float> embeddings(token_ids.size() *
                                static_cast<size_t>(hidden_dim));
  std::vector<int32_t> batch_ids(static_cast<size_t>(max_batch), 0);
  std::vector<float> batch_out(static_cast<size_t>(max_batch) *
                               static_cast<size_t>(hidden_dim));

  size_t offset = 0;
  while (offset < token_ids.size()) {
    const size_t count =
        std::min(static_cast<size_t>(max_batch), token_ids.size() - offset);
    absl::c_fill(batch_ids, 0);
    for (size_t i = 0; i < count; ++i) {
      batch_ids[i] = static_cast<int32_t>(token_ids[offset + i]);
    }
    LITERT_RETURN_IF_ERROR(
        input_buffers[0].Write<int32_t>(absl::MakeConstSpan(batch_ids)));
    ABSL_RETURN_IF_ERROR(
        runner.Run(/*signature_name=*/"", input_buffers, output_buffers));
    LITERT_RETURN_IF_ERROR(
        output_buffers[0].Read<float>(absl::MakeSpan(batch_out)));
    const size_t copy_elems = count * static_cast<size_t>(hidden_dim);
    absl::c_copy(
        absl::MakeConstSpan(batch_out).first(copy_elems),
        embeddings.begin() +
            static_cast<ptrdiff_t>(offset * static_cast<size_t>(hidden_dim)));
    offset += count;
  }
  return embeddings;
}

}  // namespace janus_pro

JanusProTextStage::JanusProTextStage(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
        end_of_input_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
    std::vector<TensorBuffer> text_emb_input_buffers,
    std::vector<TensorBuffer> text_emb_output_buffers, int max_batch,
    const JanusProModelConfig& config)
    : text_queue_(*text_queue),
      end_of_turn_queue_(*end_of_turn_queue),
      end_of_input_queue_(*end_of_input_queue),
      config_(config),
      hidden_dim_(config.llm_hidden_size),
      max_batch_(max_batch),
      tokenizer_(std::move(tokenizer)),
      text_embedder_runner_(std::move(text_embedder_runner)),
      text_emb_input_buffers_(std::move(text_emb_input_buffers)),
      text_emb_output_buffers_(std::move(text_emb_output_buffers)) {}

JanusProTextStage::~JanusProTextStage() = default;

absl::StatusOr<std::unique_ptr<JanusProTextStage>> JanusProTextStage::Create(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
        end_of_input_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
    const JanusProModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "JanusProTextStage config.llm_hidden_size and config.llm_vocab_size "
        "must be positive.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::vector<TensorBuffer> input_buffers,
      text_embedder_runner->CreateInputBuffers(/*signature_name=*/""));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<TensorBuffer> output_buffers,
      text_embedder_runner->CreateOutputBuffers(/*signature_name=*/""));
  ABSL_ASSIGN_OR_RETURN(
      int max_batch, janus_pro::ValidateTextEmbedderBuffers(
                         input_buffers, output_buffers, config.llm_hidden_size,
                         "JanusProTextStage"));
  return absl::WrapUnique(new JanusProTextStage(
      text_queue, end_of_turn_queue, end_of_input_queue, std::move(tokenizer),
      std::move(text_embedder_runner), std::move(input_buffers),
      std::move(output_buffers), max_batch, config));
}

absl::StatusOr<std::vector<float>> JanusProTextStage::EmbedTokens(
    absl::Span<const int> token_ids) {
  return janus_pro::EmbedTokenSequence(
      token_ids, config_.llm_vocab_size, hidden_dim_, max_batch_,
      *text_embedder_runner_, absl::MakeSpan(text_emb_input_buffers_),
      absl::MakeSpan(text_emb_output_buffers_), "JanusProTextStage");
}

absl::StatusOr<JanusProEmbeddingSegment> JanusProTextStage::EmbedControlText(
    absl::string_view text) {
  ABSL_ASSIGN_OR_RETURN(std::vector<int> token_ids,
                        janus_pro::TokenizeControlText(*tokenizer_, text));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> embeddings, EmbedTokens(token_ids));
  const int num_tokens = static_cast<int>(token_ids.size());
  return JanusProEmbeddingSegment{
      .embeddings = std::move(embeddings),
      .token_ids = std::move(token_ids),
      .num_tokens = num_tokens,
      .raw_text = std::string(text),
  };
}

absl::StatusOr<JanusProEmbeddingSegment> JanusProTextStage::ProcessText(
    absl::string_view text) {
  std::vector<int> token_ids;
  if (!text.empty()) {
    ABSL_ASSIGN_OR_RETURN(token_ids, TokenizeUntrustedText(*tokenizer_, text));
  }

  std::vector<float> embeddings;
  if (!token_ids.empty()) {
    ABSL_ASSIGN_OR_RETURN(embeddings, EmbedTokens(token_ids));
  }

  const int num_tokens = static_cast<int>(token_ids.size());
  return JanusProEmbeddingSegment{
      .embeddings = std::move(embeddings),
      .token_ids = std::move(token_ids),
      .num_tokens = num_tokens,
      .raw_text = std::string(text),
  };
}

bool JanusProTextStage::NeedScheduleInternal() const {
  return !text_queue_.Empty() || !end_of_turn_queue_.Empty() ||
         !end_of_input_queue_.Empty();
}

absl::Status JanusProTextStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (!NeedScheduleInternal()) {
    return absl::NotFoundError("JanusProTextStage has no ready input.");
  }

  enum class Source { kNone, kText, kEndOfTurn, kEndOfInput };
  while (NeedScheduleInternal()) {
    Source min_source = Source::kNone;
    uint64_t min_seqnum = 0;

    if (!text_queue_.Empty()) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<TextInput>* peeked_text,
                            text_queue_.PeekFront());
      min_source = Source::kText;
      min_seqnum = peeked_text->seqnum;
    }
    if (!end_of_turn_queue_.Empty()) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<EndOfInputTurn>* peeked_eot,
                            end_of_turn_queue_.PeekFront());
      if (min_source == Source::kNone || peeked_eot->seqnum < min_seqnum) {
        min_source = Source::kEndOfTurn;
        min_seqnum = peeked_eot->seqnum;
      }
    }
    if (!end_of_input_queue_.Empty()) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<EndOfInput>* peeked_eoi,
                            end_of_input_queue_.PeekFront());
      if (min_source == Source::kNone || peeked_eoi->seqnum < min_seqnum) {
        min_source = Source::kEndOfInput;
        min_seqnum = peeked_eoi->seqnum;
      }
    }

    JanusProEmbeddingSegment segment;
    switch (min_source) {
      case Source::kText: {
        ABSL_ASSIGN_OR_RETURN(const Sequenced<TextInput>* peeked_text,
                              text_queue_.PeekFront());
        ABSL_ASSIGN_OR_RETURN(segment, ProcessText(peeked_text->item.text));
        ABSL_RETURN_IF_ERROR(text_queue_.PopFront().status());
        break;
      }
      case Source::kEndOfTurn: {
        segment = JanusProEmbeddingSegment{
            .is_turn_end = true,
            .is_explicit_eot = true,
        };
        ABSL_RETURN_IF_ERROR(end_of_turn_queue_.PopFront().status());
        break;
      }
      case Source::kEndOfInput: {
        segment = JanusProEmbeddingSegment{
            .is_turn_end = true,
            .is_explicit_eot = false,
        };
        ABSL_RETURN_IF_ERROR(end_of_input_queue_.PopFront().status());
        break;
      }
      case Source::kNone:
        return absl::OkStatus();
    }
    this->PushOutput(Sequenced<JanusProEmbeddingSegment>{
        .seqnum = min_seqnum,
        .item = std::move(segment),
    });
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
