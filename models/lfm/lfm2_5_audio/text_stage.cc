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

#include "models/lfm/lfm2_5_audio/text_stage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "models/lfm/lfm2_5_audio/io_types.h"
#include "models/lfm/lfm2_5_audio/lfm2_audio_model_config.h"
#include "models/lfm/lfm2_5_audio/stage_utils.h"
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

absl::Status ValidateTokenIdsInVocab(absl::Span<const int> token_ids,
                                     int vocab_size,
                                     absl::string_view caller_name) {
  if (vocab_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        caller_name, ": llm_vocab_size must be positive, got ", vocab_size));
  }
  for (size_t i = 0; i < token_ids.size(); ++i) {
    const int id = token_ids[i];
    if (id < 0 || id >= vocab_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          caller_name, ": token_id at index ", i, " (", id,
          ") is out of valid vocabulary range [0, ", vocab_size, ")."));
    }
  }
  return absl::OkStatus();
}

// Tokenizes untrusted user text without letting `HuggingFaceTokenizer` match
// LFM2.5-Audio `<|...|>` `added_tokens` as raw special control token IDs.
// Splitting between `<` and `|` ensures no piece passed to `TextToTokenIds()`
// contains `<|`, while preserving literal characters for byte-level BPE.
absl::StatusOr<std::vector<int>> TokenizeUntrustedText(
    support::Tokenizer& tokenizer, absl::string_view text) {
  std::vector<int> token_ids;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t special_delim = text.find("<|", pos);
    if (special_delim == absl::string_view::npos) {
      ABSL_ASSIGN_OR_RETURN(std::vector<int> part_ids,
                            tokenizer.TextToTokenIds(text.substr(pos)));
      token_ids.insert(token_ids.end(), part_ids.begin(), part_ids.end());
      break;
    }
    // Include `<` in the current piece and leave `|` at the start of the next
    // piece so neither piece contains `<|`.
    ABSL_ASSIGN_OR_RETURN(
        std::vector<int> part_ids,
        tokenizer.TextToTokenIds(text.substr(pos, special_delim - pos + 1)));
    token_ids.insert(token_ids.end(), part_ids.begin(), part_ids.end());
    pos = special_delim + 1;
  }
  return token_ids;
}

}  // namespace

namespace lfm2_audio {

absl::StatusOr<std::vector<float>> EmbedTokenSequence(
    absl::Span<const int> token_ids, int vocab_size, int hidden_dim,
    LiteRtRunner* absl_nullable runner,
    std::vector<TensorBuffer>& input_buffers,
    std::vector<TensorBuffer>& output_buffers, absl::string_view caller_name) {
  ABSL_RETURN_IF_ERROR(
      ValidateTokenIdsInVocab(token_ids, vocab_size, caller_name));
  if (token_ids.empty()) {
    return std::vector<float>();
  }
  if (runner == nullptr || input_buffers.empty() || output_buffers.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat(caller_name, ": text_embedder_runner must not be null."));
  }

  std::vector<float> out(token_ids.size() * hidden_dim, 0.0f);
  std::vector<float> step_emb(hidden_dim, 0.0f);
  for (size_t i = 0; i < token_ids.size(); ++i) {
    const int32_t tok = static_cast<int32_t>(token_ids[i]);
    LITERT_RETURN_IF_ERROR(
        input_buffers[0].Write<int32_t>(absl::MakeConstSpan(&tok, 1)));
    ABSL_RETURN_IF_ERROR(
        runner->Run(/*signature_name=*/"", input_buffers, output_buffers));
    LITERT_RETURN_IF_ERROR(
        output_buffers[0].Read<float>(absl::MakeSpan(step_emb)));
    absl::c_copy(step_emb, out.begin() + i * hidden_dim);
  }
  return out;
}

}  // namespace lfm2_audio

absl::StatusOr<std::vector<int>> TokenizeLfm2AudioTextWithSpecialTokens(
    support::Tokenizer& tokenizer, absl::string_view text,
    const Lfm2AudioModelConfig& config) {
  if (text.empty()) {
    return std::vector<int>();
  }

  struct SpecialEntry {
    absl::string_view literal;
    int fallback_id;
  };
  const std::array<SpecialEntry, 6> specials = {{
      {lfm2_audio::kPadToken, config.pad_token_id},
      {lfm2_audio::kBosToken, config.bos_token_id},
      {lfm2_audio::kImStartToken, config.im_start_token_id},
      {lfm2_audio::kImEndToken, config.im_end_token_id},
      {lfm2_audio::kAudioStartToken, config.audio_start_token_id},
      {lfm2_audio::kTextEndToken, config.text_end_token_id},
  }};

  std::vector<int> token_ids;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t best_pos = absl::string_view::npos;
    size_t best_idx = 0;
    for (size_t i = 0; i < specials.size(); ++i) {
      const size_t found = text.find(specials[i].literal, pos);
      if (found != absl::string_view::npos &&
          (best_pos == absl::string_view::npos || found < best_pos)) {
        best_pos = found;
        best_idx = i;
      }
    }

    if (best_pos == absl::string_view::npos) {
      absl::string_view rem = text.substr(pos);
      ABSL_ASSIGN_OR_RETURN(std::vector<int> ids,
                            tokenizer.TextToTokenIds(rem));
      token_ids.insert(token_ids.end(), ids.begin(), ids.end());
      break;
    }

    if (best_pos > pos) {
      absl::string_view prefix = text.substr(pos, best_pos - pos);
      ABSL_ASSIGN_OR_RETURN(std::vector<int> ids,
                            tokenizer.TextToTokenIds(prefix));
      token_ids.insert(token_ids.end(), ids.begin(), ids.end());
    }

    const auto& sp = specials[best_idx];
    auto tok_id = tokenizer.TokenToId(sp.literal);
    token_ids.push_back(tok_id.ok() ? *tok_id : sp.fallback_id);
    pos = best_pos + sp.literal.size();
  }

  return token_ids;
}

Lfm2AudioTextStage::Lfm2AudioTextStage(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
    std::vector<TensorBuffer> embed_input_buffers,
    std::vector<TensorBuffer> embed_output_buffers,
    const Lfm2AudioModelConfig& config)
    : text_queue_(*text_queue),
      end_of_turn_queue_(*end_of_turn_queue),
      config_(config),
      hidden_dim_(config_.llm_hidden_size),
      tokenizer_(std::move(tokenizer)),
      text_embedder_runner_(std::move(text_embedder_runner)),
      embed_input_buffers_(std::move(embed_input_buffers)),
      embed_output_buffers_(std::move(embed_output_buffers)) {}

Lfm2AudioTextStage::~Lfm2AudioTextStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioTextStage>> Lfm2AudioTextStage::Create(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
    const Lfm2AudioModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "Lfm2AudioTextStage config.llm_hidden_size and config.llm_vocab_size "
        "must be positive.");
  }

  std::vector<TensorBuffer> embed_inputs;
  std::vector<TensorBuffer> embed_outputs;
  if (text_embedder_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        embed_inputs,
        text_embedder_runner->CreateInputBuffers(/*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        embed_outputs,
        text_embedder_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (embed_inputs.size() != 1 || embed_outputs.size() != 1) {
      return absl::InvalidArgumentError(
          "Lfm2AudioTextStage text_embedder_runner must have 1 input and 1 "
          "output buffer.");
    }
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        embed_inputs[0], ElementType::Int32, /*expected_elements=*/1,
        "Lfm2AudioTextStage text_embedder input"));
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        embed_outputs[0], ElementType::Float32,
        static_cast<size_t>(config.llm_hidden_size),
        "Lfm2AudioTextStage text_embedder output"));
  }

  return absl::WrapUnique(new Lfm2AudioTextStage(
      text_queue, end_of_turn_queue, std::move(tokenizer),
      std::move(text_embedder_runner), std::move(embed_inputs),
      std::move(embed_outputs), config));
}

bool Lfm2AudioTextStage::NeedScheduleInternal() const {
  return !text_queue_.Empty() || !end_of_turn_queue_.Empty();
}

absl::Status Lfm2AudioTextStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (!NeedScheduleInternal()) {
    return absl::NotFoundError(
        "No text or end-of-turn items ready to schedule.");
  }

  while (!text_queue_.Empty() || !end_of_turn_queue_.Empty()) {
    bool take_text = false;
    if (!text_queue_.Empty() && !end_of_turn_queue_.Empty()) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<TextInput>* t_peek,
                            text_queue_.PeekFront());
      ABSL_ASSIGN_OR_RETURN(const Sequenced<EndOfInputTurn>* e_peek,
                            end_of_turn_queue_.PeekFront());
      take_text = (t_peek->seqnum < e_peek->seqnum);
    } else {
      take_text = !text_queue_.Empty();
    }

    if (take_text) {
      ABSL_ASSIGN_OR_RETURN(Sequenced<TextInput> seq_text,
                            text_queue_.PopFront());
      Lfm2AudioEmbeddingSegment seg;
      if (!seq_text.item.text.empty()) {
        ABSL_ASSIGN_OR_RETURN(
            seg.token_ids,
            TokenizeUntrustedText(*tokenizer_, seq_text.item.text));
        seg.num_tokens = static_cast<int>(seg.token_ids.size());
        ABSL_ASSIGN_OR_RETURN(seg.embeddings, EmbedTokens(seg.token_ids));
      }
      PushOutput(Sequenced<Lfm2AudioEmbeddingSegment>{
          .seqnum = seq_text.seqnum,
          .item = std::move(seg),
      });
    } else {
      ABSL_ASSIGN_OR_RETURN(Sequenced<EndOfInputTurn> seq_eot,
                            end_of_turn_queue_.PopFront());
      Lfm2AudioEmbeddingSegment seg;
      seg.is_end_of_turn = true;
      PushOutput(Sequenced<Lfm2AudioEmbeddingSegment>{
          .seqnum = seq_eot.seqnum,
          .item = std::move(seg),
      });
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> Lfm2AudioTextStage::EmbedTokens(
    absl::Span<const int> token_ids) {
  return lfm2_audio::EmbedTokenSequence(
      token_ids, config_.llm_vocab_size, hidden_dim_,
      text_embedder_runner_.get(), embed_input_buffers_, embed_output_buffers_,
      "Lfm2AudioTextStage");
}

}  // namespace litert::omni::any2any
