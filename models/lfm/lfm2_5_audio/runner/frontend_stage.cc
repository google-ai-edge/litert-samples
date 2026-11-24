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

#include "models/lfm/lfm2_5_audio/runner/frontend_stage.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/lfm/lfm2_5_audio/runner/io_types.h"
#include "models/lfm/lfm2_5_audio/runner/lfm2_audio_model_config.h"
#include "models/lfm/lfm2_5_audio/runner/text_stage.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/utils/tensor_utils.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

Lfm2AudioFrontendStage::Lfm2AudioFrontendStage(
    SingleThreadedStageWithDeque<
        Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nonnull text_stage,
    SingleThreadedStageWithDeque<
        Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nullable audio_stage,
    Lfm2AudioEmbeddingSegment system_prefix_seg,
    Lfm2AudioEmbeddingSegment user_start_seg,
    Lfm2AudioEmbeddingSegment user_end_seg,
    Lfm2AudioGenerationMode resolved_mode, const Lfm2AudioModelConfig& config)
    : text_stage_(*text_stage),
      audio_stage_(audio_stage),
      config_(config),
      hidden_dim_(config_.llm_hidden_size),
      system_prefix_seg_(std::move(system_prefix_seg)),
      user_start_seg_(std::move(user_start_seg)),
      user_end_seg_(std::move(user_end_seg)),
      resolved_mode_(resolved_mode) {}

Lfm2AudioFrontendStage::~Lfm2AudioFrontendStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioFrontendStage>>
Lfm2AudioFrontendStage::Create(
    SingleThreadedStageWithDeque<
        Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nonnull text_stage,
    SingleThreadedStageWithDeque<
        Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nullable audio_stage,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner,
    const Lfm2AudioModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "Lfm2AudioFrontendStage config.llm_hidden_size and "
        "config.llm_vocab_size must be positive.");
  }
  if (text_embedder_runner == nullptr) {
    return absl::FailedPreconditionError(
        "Lfm2AudioFrontendStage: text_embedder_runner must not be null.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<TensorBuffer> embed_inputs,
      text_embedder_runner->CreateInputBuffers(/*signature_name=*/""));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<TensorBuffer> embed_outputs,
      text_embedder_runner->CreateOutputBuffers(/*signature_name=*/""));
  if (embed_inputs.size() != 1 || embed_outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "Lfm2AudioFrontendStage text_embedder_runner must have 1 input and 1 "
        "output buffer.");
  }
  ABSL_RETURN_IF_ERROR(ValidateTensorBuffer(
      embed_inputs[0], ElementType::Int32, /*expected_elements=*/1,
      "Lfm2AudioFrontendStage text_embedder input"));
  ABSL_RETURN_IF_ERROR(
      ValidateTensorBuffer(embed_outputs[0], ElementType::Float32,
                           static_cast<size_t>(config.llm_hidden_size),
                           "Lfm2AudioFrontendStage text_embedder output"));

  auto embed_control =
      [&](absl::string_view text) -> absl::StatusOr<Lfm2AudioEmbeddingSegment> {
    Lfm2AudioEmbeddingSegment seg;
    ABSL_ASSIGN_OR_RETURN(seg.token_ids, TokenizeLfm2AudioTextWithSpecialTokens(
                                             *tokenizer, text, config));
    seg.num_tokens = static_cast<int>(seg.token_ids.size());
    ABSL_ASSIGN_OR_RETURN(
        seg.embeddings,
        lfm2_audio::EmbedTokenSequence(
            seg.token_ids, config.llm_vocab_size, config.llm_hidden_size,
            text_embedder_runner.get(), embed_inputs, embed_outputs,
            "Lfm2AudioFrontendStage"));
    return seg;
  };

  std::string effective_system_prompt = config.system_prompt;
  if (effective_system_prompt == lfm2_audio::kDefaultInterleavedSystemPrompt) {
    if (!config.generate_audio) {
      effective_system_prompt =
          std::string(lfm2_audio::kDefaultSequentialTextSystemPrompt);
    } else if (config.generation_mode == Lfm2AudioGenerationMode::kSequential) {
      effective_system_prompt =
          std::string(lfm2_audio::kDefaultTtsSystemPrompt);
    }
  }

  absl::string_view voice_instruction;
  if (absl::EqualsIgnoreCase(config.voice_name, "us_male") ||
      absl::EqualsIgnoreCase(config.voice_name, "US male")) {
    voice_instruction = "Use the US male voice.";
  } else if (absl::EqualsIgnoreCase(config.voice_name, "us_female") ||
             absl::EqualsIgnoreCase(config.voice_name, "US female")) {
    voice_instruction = "Use the US female voice.";
  } else if (absl::EqualsIgnoreCase(config.voice_name, "uk_male") ||
             absl::EqualsIgnoreCase(config.voice_name, "UK male")) {
    voice_instruction = "Use the UK male voice.";
  } else if (absl::EqualsIgnoreCase(config.voice_name, "uk_female") ||
             absl::EqualsIgnoreCase(config.voice_name, "UK female")) {
    voice_instruction = "Use the UK female voice.";
  }
  if (config.generate_audio && !voice_instruction.empty() &&
      !effective_system_prompt.empty() &&
      !absl::StrContainsIgnoreCase(effective_system_prompt, "voice")) {
    absl::StrAppend(&effective_system_prompt, " ", voice_instruction);
  }

  Lfm2AudioGenerationMode resolved_mode = Lfm2AudioGenerationMode::kSequential;
  if (!config.generate_audio) {
    resolved_mode = Lfm2AudioGenerationMode::kSequential;
  } else if (config.generation_mode != Lfm2AudioGenerationMode::kAuto) {
    resolved_mode = config.generation_mode;
  } else if (absl::StrContainsIgnoreCase(effective_system_prompt,
                                         "interleaved")) {
    resolved_mode = Lfm2AudioGenerationMode::kInterleaved;
  } else {
    resolved_mode = Lfm2AudioGenerationMode::kSequential;
  }

  std::string prefix = std::string(lfm2_audio::kBosToken);
  if (!effective_system_prompt.empty()) {
    absl::StrAppend(&prefix, lfm2_audio::kImStartToken, "system\n",
                    effective_system_prompt, lfm2_audio::kImEndToken, "\n");
  }
  ABSL_ASSIGN_OR_RETURN(Lfm2AudioEmbeddingSegment system_prefix_seg,
                        embed_control(prefix));
  ABSL_ASSIGN_OR_RETURN(
      Lfm2AudioEmbeddingSegment user_start_seg,
      embed_control(absl::StrCat(lfm2_audio::kImStartToken, "user\n")));
  ABSL_ASSIGN_OR_RETURN(
      Lfm2AudioEmbeddingSegment user_end_seg,
      embed_control(absl::StrCat(lfm2_audio::kImEndToken, "\n",
                                 lfm2_audio::kImStartToken, "assistant\n")));

  return absl::WrapUnique(new Lfm2AudioFrontendStage(
      text_stage, audio_stage, std::move(system_prefix_seg),
      std::move(user_start_seg), std::move(user_end_seg), resolved_mode,
      config));
}

bool Lfm2AudioFrontendStage::NeedScheduleInternal() const {
  auto matches_next =
      [this](const SingleThreadedStageWithDeque<
             Sequenced<Lfm2AudioEmbeddingSegment>>* stage) -> bool {
    if (stage == nullptr || !stage->HasOutput()) {
      return false;
    }
    auto peeked = stage->PeekOutput();
    return peeked.ok() && (*peeked)->seqnum == next_expected_seqnum_;
  };
  return matches_next(&text_stage_) || matches_next(audio_stage_);
}

absl::Status Lfm2AudioFrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  bool matched_any = false;
  while (true) {
    SingleThreadedStageWithDeque<Sequenced<Lfm2AudioEmbeddingSegment>>*
        ready_stage = nullptr;
    for (auto* candidate : {&text_stage_, audio_stage_}) {
      if (candidate == nullptr || !candidate->HasOutput()) {
        continue;
      }
      auto peeked = candidate->PeekOutput();
      if (peeked.ok() && (*peeked)->seqnum == next_expected_seqnum_) {
        ready_stage = candidate;
        break;
      }
    }

    if (ready_stage == nullptr) {
      break;
    }

    matched_any = true;
    ABSL_ASSIGN_OR_RETURN(Sequenced<Lfm2AudioEmbeddingSegment> seq_seg,
                          ready_stage->GetOutput());
    ++next_expected_seqnum_;

    if (seq_seg.item.is_end_of_turn) {
      ABSL_RETURN_IF_ERROR(FinalizeTurn());
    } else if (seq_seg.item.num_tokens > 0) {
      ABSL_RETURN_IF_ERROR(AppendSegment(seq_seg.item));
    }
  }

  if (!matched_any) {
    return absl::NotFoundError(
        "Lfm2AudioFrontendStage has no ready sequenced items.");
  }
  return absl::OkStatus();
}

void Lfm2AudioFrontendStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  next_expected_seqnum_ = 0;
  is_first_turn_ = true;
  turn_segments_.clear();
}

absl::Status Lfm2AudioFrontendStage::AppendSegment(
    const Lfm2AudioEmbeddingSegment& seg) {
  if (seg.embeddings.size() !=
      static_cast<size_t>(seg.num_tokens) * hidden_dim_) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Lfm2AudioEmbeddingSegment embeddings size (", seg.embeddings.size(),
        ") does not match num_tokens * llm_hidden_size (",
        seg.num_tokens * hidden_dim_, ")."));
  }
  turn_segments_.push_back(seg);
  return absl::OkStatus();
}

absl::Status Lfm2AudioFrontendStage::FinalizeTurn() {
  Lfm2AudioFrontendOutput out;
  out.generate_audio = config_.generate_audio;
  out.generation_mode = config_.generate_audio
                            ? resolved_mode_
                            : Lfm2AudioGenerationMode::kSequential;

  auto append_to_output = [&out](const Lfm2AudioEmbeddingSegment& seg) {
    out.prompt_embeddings.insert(out.prompt_embeddings.end(),
                                 seg.embeddings.begin(), seg.embeddings.end());
    out.prompt_token_ids.insert(out.prompt_token_ids.end(),
                                seg.token_ids.begin(), seg.token_ids.end());
    out.prompt_len += seg.num_tokens;
  };

  if (is_first_turn_) {
    append_to_output(system_prefix_seg_);
    is_first_turn_ = false;
  }
  append_to_output(user_start_seg_);
  for (const auto& seg : turn_segments_) {
    append_to_output(seg);
  }
  turn_segments_.clear();
  append_to_output(user_end_seg_);

  PushOutput(std::move(out));
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
