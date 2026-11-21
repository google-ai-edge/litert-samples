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

#include "models/deepseek/janus_pro/frontend_stage.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/io_types.h"
#include "models/deepseek/janus_pro/janus_pro_model_config.h"
#include "models/deepseek/janus_pro/text_stage.h"
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
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/fan_in_stage.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace {

bool IsImageSegment(const JanusProEmbeddingSegment& segment) {
  return absl::c_linear_search(segment.token_ids,
                               janus_pro::kImageEmbedPlaceholderTokenId);
}

absl::StatusOr<JanusProEmbeddingSegment> EmbedControlSnippet(
    support::Tokenizer& tokenizer, LiteRtRunner& runner,
    absl::Span<TensorBuffer> input_buffers,
    absl::Span<TensorBuffer> output_buffers, int vocab_size, int hidden_dim,
    int max_batch, absl::string_view text) {
  ABSL_ASSIGN_OR_RETURN(std::vector<int> token_ids,
                        janus_pro::TokenizeControlText(tokenizer, text));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> embeddings,
      janus_pro::EmbedTokenSequence(token_ids, vocab_size, hidden_dim,
                                    max_batch, runner, input_buffers,
                                    output_buffers, "JanusProFrontendStage"));
  const int num_tokens = static_cast<int>(token_ids.size());
  return JanusProEmbeddingSegment{
      .embeddings = std::move(embeddings),
      .token_ids = std::move(token_ids),
      .num_tokens = num_tokens,
      .raw_text = std::string(text),
  };
}

}  // namespace

JanusProFrontendStage::JanusProFrontendStage(
    SingleThreadedStageWithDeque<
        Sequenced<JanusProEmbeddingSegment>>* absl_nonnull text_stage,
    SingleThreadedStageWithDeque<
        Sequenced<JanusProEmbeddingSegment>>* absl_nullable image_stage,
    PrecomputedTemplates templates, const JanusProModelConfig& config)
    : templates_(std::move(templates)),
      hidden_dim_(config.llm_hidden_size),
      generate_image_(config.generate_image) {
  fan_in_.AddInputStage(text_stage);
  if (image_stage != nullptr) {
    fan_in_.AddInputStage(image_stage);
  }
}

JanusProFrontendStage::~JanusProFrontendStage() = default;

absl::StatusOr<std::unique_ptr<JanusProFrontendStage>>
JanusProFrontendStage::Create(
    SingleThreadedStageWithDeque<
        Sequenced<JanusProEmbeddingSegment>>* absl_nonnull text_stage,
    SingleThreadedStageWithDeque<
        Sequenced<JanusProEmbeddingSegment>>* absl_nullable image_stage,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
    const JanusProModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "JanusProFrontendStage config.llm_hidden_size and "
        "config.llm_vocab_size must be positive.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<TensorBuffer> embed_inputs,
      text_embedder_runner->CreateInputBuffers(/*signature_name=*/""));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<TensorBuffer> embed_outputs,
      text_embedder_runner->CreateOutputBuffers(/*signature_name=*/""));
  ABSL_ASSIGN_OR_RETURN(int max_batch,
                        janus_pro::ValidateTextEmbedderBuffers(
                            embed_inputs, embed_outputs, config.llm_hidden_size,
                            "JanusProFrontendStage"));

  auto embed_control =
      [&](absl::string_view text) -> absl::StatusOr<JanusProEmbeddingSegment> {
    return EmbedControlSnippet(
        *tokenizer, *text_embedder_runner, absl::MakeSpan(embed_inputs),
        absl::MakeSpan(embed_outputs), config.llm_vocab_size,
        config.llm_hidden_size, max_batch, text);
  };

  PrecomputedTemplates templates;
  if (config.generate_image) {
    // Visual generation mode uses `system_prompt = ""` and starts with BOS +
    // `<|User|>: `.
    ABSL_ASSIGN_OR_RETURN(
        templates.first_turn_prefix,
        embed_control(
            absl::StrCat(janus_pro::kBosToken, janus_pro::kUserToken, ": ")));
    templates.next_turn_prefix = templates.first_turn_prefix;
    ABSL_ASSIGN_OR_RETURN(
        templates.turn_end_suffix,
        embed_control(absl::StrCat("\n\n", janus_pro::kAssistantToken, ":",
                                   janus_pro::kBeginOfImageToken)));
  } else {
    std::string first_prefix(janus_pro::kBosToken);
    if (!config.system_prompt.empty()) {
      absl::StrAppend(&first_prefix, config.system_prompt, "\n\n");
    }
    absl::StrAppend(&first_prefix, janus_pro::kUserToken, ": ");
    ABSL_ASSIGN_OR_RETURN(templates.first_turn_prefix,
                          embed_control(first_prefix));
    ABSL_ASSIGN_OR_RETURN(
        templates.next_turn_prefix,
        embed_control(
            absl::StrCat(janus_pro::kEosToken, janus_pro::kUserToken, ": ")));
    ABSL_ASSIGN_OR_RETURN(
        templates.turn_end_suffix,
        embed_control(absl::StrCat("\n\n", janus_pro::kAssistantToken, ":")));
  }

  ABSL_ASSIGN_OR_RETURN(
      JanusProEmbeddingSegment boundary,
      embed_control(absl::StrCat(janus_pro::kBeginOfImageToken,
                                 janus_pro::kEndOfImageToken, "\n")));
  const size_t hidden = static_cast<size_t>(config.llm_hidden_size);
  if (boundary.token_ids.size() < 2 ||
      boundary.embeddings.size() != boundary.token_ids.size() * hidden) {
    return absl::InternalError(
        "JanusProFrontendStage failed to precompute image boundary tokens.");
  }
  templates.boi_token_id = boundary.token_ids[0];
  templates.eoi_token_id = boundary.token_ids[1];
  templates.boi_embedding.assign(boundary.embeddings.begin(),
                                 boundary.embeddings.begin() + hidden);
  templates.eoi_embedding.assign(boundary.embeddings.begin() + hidden,
                                 boundary.embeddings.begin() + 2 * hidden);
  templates.image_newline_token_ids.assign(boundary.token_ids.begin() + 2,
                                           boundary.token_ids.end());
  templates.image_newline_embeddings.assign(
      boundary.embeddings.begin() + 2 * hidden, boundary.embeddings.end());

  templates.bos_token_id = janus_pro::ResolveSpecialTokenId(
      *tokenizer, janus_pro::kBosToken, janus_pro::kBosTokenId);
  templates.pad_token_id = janus_pro::ResolveSpecialTokenId(
      *tokenizer, janus_pro::kPadToken, janus_pro::kPadTokenId);
  const std::vector<int> bos_pad_ids = {templates.bos_token_id,
                                        templates.pad_token_id};
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> bos_pad_embs,
      janus_pro::EmbedTokenSequence(
          bos_pad_ids, config.llm_vocab_size, config.llm_hidden_size, max_batch,
          *text_embedder_runner, absl::MakeSpan(embed_inputs),
          absl::MakeSpan(embed_outputs), "JanusProFrontendStage"));
  templates.bos_embedding.assign(bos_pad_embs.begin(),
                                 bos_pad_embs.begin() + hidden);
  templates.pad_embedding.assign(bos_pad_embs.begin() + hidden,
                                 bos_pad_embs.begin() + 2 * hidden);

  return absl::WrapUnique(new JanusProFrontendStage(
      text_stage, image_stage, std::move(templates), config));
}

bool JanusProFrontendStage::NeedScheduleInternal() const {
  return fan_in_.NeedSchedule() || fan_in_.HasOutput();
}

absl::Status JanusProFrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (fan_in_.NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(fan_in_.Schedule());
  }

  bool processed_any = false;
  while (fan_in_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(const JanusProEmbeddingSegment* segment,
                          fan_in_.PeekOutput());
    ABSL_RETURN_IF_ERROR(ProcessSegment(*segment));
    ABSL_RETURN_IF_ERROR(fan_in_.GetOutput().status());
    processed_any = true;
  }

  if (!processed_any && !this->HasOutput()) {
    return absl::NotFoundError(
        "JanusProFrontendStage has no ready sequence items.");
  }
  return absl::OkStatus();
}

absl::StatusOr<JanusProEmbeddingSegment>
JanusProFrontendStage::FrameImageSegment(
    const JanusProEmbeddingSegment& image) const {
  if (image.embeddings.empty() ||
      image.embeddings.size() % static_cast<size_t>(hidden_dim_) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("JanusProFrontendStage image segment embedding size (",
                     image.embeddings.size(),
                     ") must be a positive multiple of llm_hidden_size (",
                     hidden_dim_, ")."));
  }
  size_t placeholder_count = 0;
  for (int tok : image.token_ids) {
    if (tok == janus_pro::kImageEmbedPlaceholderTokenId) {
      ++placeholder_count;
    } else if (tok != janus_pro::kBeginOfImageTokenId &&
               tok != janus_pro::kEndOfImageTokenId) {
      return absl::InvalidArgumentError(
          absl::StrCat("JanusProFrontendStage image segment contains "
                       "unexpected token ID (",
                       tok, ")."));
    }
  }
  const size_t total_image_tokens =
      image.embeddings.size() / static_cast<size_t>(hidden_dim_);
  if (placeholder_count != total_image_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "JanusProFrontendStage image segment placeholder token count (",
        placeholder_count, ") does not match image embedding token count (",
        total_image_tokens, ")."));
  }

  std::vector<float> framed_embeddings;
  std::vector<int> framed_token_ids;
  size_t emb_offset = 0;
  for (int tok : image.token_ids) {
    if (tok == janus_pro::kBeginOfImageTokenId) {
      framed_embeddings.insert(framed_embeddings.end(),
                               templates_.boi_embedding.begin(),
                               templates_.boi_embedding.end());
      framed_token_ids.push_back(templates_.boi_token_id);
    } else if (tok == janus_pro::kEndOfImageTokenId) {
      framed_embeddings.insert(framed_embeddings.end(),
                               templates_.eoi_embedding.begin(),
                               templates_.eoi_embedding.end());
      framed_token_ids.push_back(templates_.eoi_token_id);
    } else {
      framed_embeddings.insert(
          framed_embeddings.end(), image.embeddings.begin() + emb_offset,
          image.embeddings.begin() + emb_offset + hidden_dim_);
      framed_token_ids.push_back(janus_pro::kImageEmbedPlaceholderTokenId);
      emb_offset += static_cast<size_t>(hidden_dim_);
    }
  }
  framed_embeddings.insert(framed_embeddings.end(),
                           templates_.image_newline_embeddings.begin(),
                           templates_.image_newline_embeddings.end());
  framed_token_ids.insert(framed_token_ids.end(),
                          templates_.image_newline_token_ids.begin(),
                          templates_.image_newline_token_ids.end());

  const int num_tokens = static_cast<int>(framed_token_ids.size());
  return JanusProEmbeddingSegment{
      .embeddings = std::move(framed_embeddings),
      .token_ids = std::move(framed_token_ids),
      .num_tokens = num_tokens,
  };
}

absl::Status JanusProFrontendStage::ProcessSegment(
    const JanusProEmbeddingSegment& segment) {
  if (segment.is_turn_end) {
    if (segment.is_explicit_eot) {
      if (!in_turn_) {
        AppendRawSegment(is_first_turn_ ? templates_.first_turn_prefix
                                        : templates_.next_turn_prefix);
        is_first_turn_ = false;
      }
      AppendRawSegment(templates_.turn_end_suffix);
      ABSL_RETURN_IF_ERROR(FlushTurn());
      in_turn_ = false;
    } else if (in_turn_) {
      AppendRawSegment(templates_.turn_end_suffix);
      ABSL_RETURN_IF_ERROR(FlushTurn());
      in_turn_ = false;
    }
    return absl::OkStatus();
  }

  if (segment.embeddings.empty() && segment.token_ids.empty()) {
    return absl::OkStatus();
  }

  if (IsImageSegment(segment)) {
    ABSL_ASSIGN_OR_RETURN(JanusProEmbeddingSegment framed_image,
                          FrameImageSegment(segment));
    if (!in_turn_) {
      AppendRawSegment(is_first_turn_ ? templates_.first_turn_prefix
                                      : templates_.next_turn_prefix);
      is_first_turn_ = false;
      in_turn_ = true;
    }
    AppendRawSegment(framed_image);
    return absl::OkStatus();
  }

  if (!in_turn_) {
    AppendRawSegment(is_first_turn_ ? templates_.first_turn_prefix
                                    : templates_.next_turn_prefix);
    is_first_turn_ = false;
    in_turn_ = true;
  }
  AppendRawSegment(segment);
  return absl::OkStatus();
}

void JanusProFrontendStage::AppendRawSegment(
    const JanusProEmbeddingSegment& segment) {
  prompt_embeddings_.insert(prompt_embeddings_.end(),
                            segment.embeddings.begin(),
                            segment.embeddings.end());
  token_ids_.insert(token_ids_.end(), segment.token_ids.begin(),
                    segment.token_ids.end());
}

absl::Status JanusProFrontendStage::FlushInternal() {
  ABSL_RETURN_IF_ERROR(fan_in_.Flush());
  while (fan_in_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(const JanusProEmbeddingSegment* segment,
                          fan_in_.PeekOutput());
    ABSL_RETURN_IF_ERROR(ProcessSegment(*segment));
    ABSL_RETURN_IF_ERROR(fan_in_.GetOutput().status());
  }
  if (in_turn_) {
    AppendRawSegment(templates_.turn_end_suffix);
  }
  ABSL_RETURN_IF_ERROR(FlushTurn());
  in_turn_ = false;
  return absl::OkStatus();
}

void JanusProFrontendStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  fan_in_.Reset();
  is_first_turn_ = true;
  in_turn_ = false;
  ResetTurn();
}

absl::Status JanusProFrontendStage::FlushTurn() {
  if (prompt_embeddings_.empty() && token_ids_.empty()) {
    ResetTurn();
    return absl::OkStatus();
  }
  const int prompt_len =
      hidden_dim_ > 0
          ? static_cast<int>(prompt_embeddings_.size() / hidden_dim_)
          : static_cast<int>(token_ids_.size());

  std::vector<int> uncond_token_ids;
  std::vector<float> uncond_embeddings;
  int uncond_prompt_len = 0;

  if (generate_image_ && prompt_len > 0) {
    uncond_token_ids.assign(prompt_len, templates_.pad_token_id);
    uncond_token_ids.front() = templates_.bos_token_id;
    if (prompt_len >= 2) {
      uncond_token_ids.back() = templates_.boi_token_id;
    }

    uncond_embeddings.reserve(static_cast<size_t>(prompt_len) * hidden_dim_);
    uncond_embeddings.insert(uncond_embeddings.end(),
                             templates_.bos_embedding.begin(),
                             templates_.bos_embedding.end());
    for (int i = 1; i + 1 < prompt_len; ++i) {
      uncond_embeddings.insert(uncond_embeddings.end(),
                               templates_.pad_embedding.begin(),
                               templates_.pad_embedding.end());
    }
    if (prompt_len >= 2) {
      uncond_embeddings.insert(uncond_embeddings.end(),
                               templates_.boi_embedding.begin(),
                               templates_.boi_embedding.end());
    }
    uncond_prompt_len = prompt_len;
  }

  this->PushOutput(JanusProFrontendChunk{
      .cond_embeddings = std::move(prompt_embeddings_),
      .uncond_embeddings = std::move(uncond_embeddings),
      .prompt_len = prompt_len,
      .uncond_prompt_len = uncond_prompt_len,
      .token_ids = std::move(token_ids_),
      .uncond_token_ids = std::move(uncond_token_ids),
      .generate_image = generate_image_,
  });
  ResetTurn();
  return absl::OkStatus();
}

void JanusProFrontendStage::ResetTurn() {
  prompt_embeddings_.clear();
  token_ids_.clear();
}

}  // namespace litert::omni::any2any
