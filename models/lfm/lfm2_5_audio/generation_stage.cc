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

#include "models/lfm/lfm2_5_audio/generation_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "models/lfm/lfm2_5_audio/io_types.h"
#include "models/lfm/lfm2_5_audio/lfm2_audio_model_config.h"
#include "models/lfm/lfm2_5_audio/stage_utils.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/any2any/any2any_sampler.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::any2any {
namespace {

absl::StatusOr<std::vector<float>> ReadTensorBufferAsFloatVector(
    TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                          tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          tensor_type.Layout().NumElements());
  if (tensor_type.ElementType() == ElementType::Float32) {
    std::vector<float> out(num_elements);
    LITERT_RETURN_IF_ERROR(tensor_buffer.Read<float>(absl::MakeSpan(out)));
    return out;
  }
  return absl::InvalidArgumentError(
      "Expected Float32 TensorBuffer from LiteRtLmRunner::Decode.");
}

absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithText(
    const TensorBuffer& text_tokens) {
  LITERT_ASSIGN_OR_RETURN(TensorBuffer text_tokens_dup,
                          text_tokens.Duplicate());
  return lm::ExecutorInputs(lm::ExecutorTextData(std::move(text_tokens_dup)),
                            /*vision_data=*/std::nullopt,
                            /*audio_data=*/std::nullopt);
}

absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithAudio(
    const TensorBuffer& audio_embeddings) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                          audio_embeddings.TensorType());
  const auto& dims = tensor_type.Layout().Dimensions();
  if (dims.size() != 3 || dims[0] != 1) {
    return absl::InvalidArgumentError(
        "Audio embeddings tensor must have shape [1, seq_len, hidden_dim].");
  }
  const int seq_len = dims[1];
  std::vector<int32_t> placeholder_tokens(seq_len,
                                          lm::ExecutorAudioData::kSpecialToken);
  LITERT_ASSIGN_OR_RETURN(TensorBuffer text_tokens,
                          support::CreateTensorBuffer<int32_t>({1, seq_len}));
  LITERT_RETURN_IF_ERROR(
      text_tokens.Write<int32_t>(absl::MakeConstSpan(placeholder_tokens)));
  LITERT_ASSIGN_OR_RETURN(TensorBuffer audio_dup, audio_embeddings.Duplicate());
  return lm::ExecutorInputs(
      lm::ExecutorTextData(std::move(text_tokens)),
      /*vision_data=*/std::nullopt,
      lm::ExecutorAudioData(std::move(audio_dup),
                            /*per_layer_embeddings=*/std::nullopt));
}

struct DecodeStepOutput {
  std::vector<float> logits;
  std::vector<float> hidden_state;
};

absl::StatusOr<DecodeStepOutput> ExtractDecodeStepOutput(
    TensorBuffer& decode_buf, int vocab_size, int hidden_dim,
    bool require_hidden_state) {
  ABSL_ASSIGN_OR_RETURN(std::vector<float> raw_out,
                        ReadTensorBufferAsFloatVector(decode_buf));
  if (raw_out.empty()) {
    return absl::InternalError("LLM Decode returned an empty TensorBuffer.");
  }
  DecodeStepOutput step_out;
  const int raw_len = static_cast<int>(raw_out.size());
  if (raw_len == vocab_size + hidden_dim) {
    step_out.hidden_state.assign(raw_out.begin() + vocab_size, raw_out.end());
    raw_out.resize(vocab_size);
  } else if (raw_len == vocab_size) {
    if (require_hidden_state) {
      return absl::FailedPreconditionError(
          "LFM2 Decode output contains only vocabulary logits without "
          "last_hidden_state required for Depthformer audio generation.");
    }
    step_out.hidden_state.assign(hidden_dim, 0.0f);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("LFM2 Decode output size (", raw_len,
                     ") does not match llm_vocab_size (", vocab_size,
                     ") or llm_vocab_size + llm_hidden_size (",
                     vocab_size + hidden_dim, ")."));
  }
  step_out.logits = std::move(raw_out);
  return step_out;
}

}  // namespace

Lfm2AudioGenerationStage::Lfm2AudioGenerationStage(
    Stage<Lfm2AudioFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::vector<TensorBuffer> proj_input_buffers,
    std::vector<TensorBuffer> proj_output_buffers,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner,
    std::vector<TensorBuffer> dec_input_buffers,
    std::vector<TensorBuffer> dec_output_buffers,
    std::unique_ptr<LiteRtRunner> absl_nullable audio_vocoder_runner,
    std::vector<TensorBuffer> voc_input_buffers,
    std::vector<TensorBuffer> voc_output_buffers, int voc_chunk_frames,
    int voc_codebooks_dim_index, const Lfm2AudioModelConfig& config)
    : frontend_(*frontend),
      config_(config),
      hidden_dim_(config_.llm_hidden_size),
      tokenizer_(std::move(tokenizer)),
      llm_runner_(std::move(llm_runner)),
      tts_projector_runner_(std::move(tts_projector_runner)),
      proj_input_buffers_(std::move(proj_input_buffers)),
      proj_output_buffers_(std::move(proj_output_buffers)),
      tts_decoder_runner_(std::move(tts_decoder_runner)),
      dec_input_buffers_(std::move(dec_input_buffers)),
      dec_output_buffers_(std::move(dec_output_buffers)),
      audio_vocoder_runner_(std::move(audio_vocoder_runner)),
      voc_input_buffers_(std::move(voc_input_buffers)),
      voc_output_buffers_(std::move(voc_output_buffers)),
      voc_chunk_frames_(voc_chunk_frames),
      voc_codebooks_dim_index_(voc_codebooks_dim_index),
      text_sampler_(Any2AnySampler::Options{
          .do_sample = config_.do_sample,
          .temperature = config_.temperature,
          .top_k = config_.top_k,
          .top_p = config_.top_p,
          .seed = config_.seed,
      }),
      tts_sampler_(Any2AnySampler::Options{
          .do_sample = config_.tts_temperature > 0.0f,
          .temperature = config_.tts_temperature,
          .top_k = config_.tts_top_k,
          .top_p = config_.tts_top_p,
          .seed = config_.seed,
      }) {}

Lfm2AudioGenerationStage::~Lfm2AudioGenerationStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioGenerationStage>>
Lfm2AudioGenerationStage::Create(
    Stage<Lfm2AudioFrontendOutput>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable audio_vocoder_runner,
    const Lfm2AudioModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0 ||
      config.tts_hidden_size <= 0 || config.tts_vocab_size <= 1 ||
      config.tts_num_codebooks <= 0) {
    return absl::InvalidArgumentError(
        "Lfm2AudioGenerationStage config dimensions must be positive (and "
        "tts_vocab_size >= 2).");
  }

  std::vector<TensorBuffer> proj_inputs;
  std::vector<TensorBuffer> proj_outputs;
  if (tts_projector_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(proj_inputs, tts_projector_runner->CreateInputBuffers(
                                           /*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        proj_outputs,
        tts_projector_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (proj_inputs.size() != 2 || proj_outputs.size() != 2) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage tts_projector_runner must have 2 input "
          "buffers (llm_hidden, audio_codes) and 2 output buffers "
          "(depthformer_in, audio_emb).");
    }
    if (static_cast<size_t>(config.tts_num_codebooks) >
        std::numeric_limits<size_t>::max() /
            static_cast<size_t>(config.tts_hidden_size)) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage depthformer_in size overflows size_t.");
    }
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        proj_inputs[0], ElementType::Float32,
        static_cast<size_t>(config.llm_hidden_size),
        "tts_projector input[0] (llm_hidden)"));
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        proj_inputs[1], ElementType::Int32,
        static_cast<size_t>(config.tts_num_codebooks),
        "tts_projector input[1] (audio_codes)"));
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        proj_outputs[0], ElementType::Float32,
        static_cast<size_t>(config.tts_num_codebooks) *
            static_cast<size_t>(config.tts_hidden_size),
        "tts_projector output[0] (depthformer_in)"));
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        proj_outputs[1], ElementType::Float32,
        static_cast<size_t>(config.llm_hidden_size),
        "tts_projector output[1] (audio_emb)"));
  }

  std::vector<TensorBuffer> dec_inputs;
  std::vector<TensorBuffer> dec_outputs;
  if (tts_decoder_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(dec_inputs, tts_decoder_runner->CreateInputBuffers(
                                          /*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(dec_outputs, tts_decoder_runner->CreateOutputBuffers(
                                           /*signature_name=*/""));
    if (dec_inputs.size() != 4 || dec_outputs.size() != 2) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage tts_decoder_runner must have 4 input "
          "buffers (cond_emb, prev_audio_code, step_idx, kv_cache) and 2 "
          "output buffers (logits, new_kv_cache).");
    }
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        dec_inputs[0], ElementType::Float32,
        static_cast<size_t>(config.tts_hidden_size),
        "tts_decoder input[0] (cond_emb)"));
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        dec_inputs[1], ElementType::Int32, /*expected_elements=*/1,
        "tts_decoder input[1] (prev_audio_code)"));
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        dec_inputs[2], ElementType::Int32, /*expected_elements=*/1,
        "tts_decoder input[2] (step_idx)"));
    ABSL_RETURN_IF_ERROR(lfm2_audio::ValidateTensorBuffer(
        dec_outputs[0], ElementType::Float32,
        static_cast<size_t>(config.tts_vocab_size),
        "tts_decoder output[0] (logits)"));
    LITERT_ASSIGN_OR_RETURN(RankedTensorType kv_in_type,
                            dec_inputs[3].TensorType());
    LITERT_ASSIGN_OR_RETURN(RankedTensorType kv_out_type,
                            dec_outputs[1].TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t kv_in_elements,
                            kv_in_type.Layout().NumElements());
    if (kv_in_type.ElementType() != ElementType::Float32 ||
        kv_in_elements == 0 || kv_in_type != kv_out_type) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage tts_decoder_runner kv_cache input and "
          "output must be non-empty Float32 tensors with identical shape.");
    }
    if (kv_in_type.Layout().Rank() == 4) {
      const auto& kv_dims = kv_in_type.Layout().Dimensions();
      if (kv_dims[0] != 1 || kv_dims[2] < config.tts_num_codebooks) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Lfm2AudioGenerationStage tts_decoder_runner 4D kv_cache shape [",
            kv_dims[0], ", ", kv_dims[1], ", ", kv_dims[2], ", ", kv_dims[3],
            "] must have batch=1 and cache_len >= tts_num_codebooks (",
            config.tts_num_codebooks, ")."));
      }
    }
  }

  std::vector<TensorBuffer> voc_inputs;
  std::vector<TensorBuffer> voc_outputs;
  int voc_chunk_frames = std::max(1, config.vocoder_chunk_frames);
  int voc_codebooks_dim_index = 1;
  if (audio_vocoder_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(voc_inputs, audio_vocoder_runner->CreateInputBuffers(
                                          /*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        voc_outputs,
        audio_vocoder_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (voc_inputs.empty() || voc_outputs.empty()) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage audio_vocoder_runner must have at least 1 "
          "input and 1 output buffer.");
    }
    LITERT_ASSIGN_OR_RETURN(auto voc_in_type, voc_inputs[0].TensorType());
    if (voc_in_type.ElementType() != ElementType::Int32) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage audio_vocoder_runner input[0] must be "
          "Int32.");
    }
    const auto& dims = voc_in_type.Layout().Dimensions();
    if (dims.size() == 3) {
      if (dims[0] != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "audio_vocoder_runner 3D input batch dimension must be 1, got ",
            dims[0], "."));
      }
      if (dims[1] == config.tts_num_codebooks) {
        voc_codebooks_dim_index = 1;
        voc_chunk_frames = dims[2];
      } else if (dims[2] == config.tts_num_codebooks) {
        voc_codebooks_dim_index = 2;
        voc_chunk_frames = dims[1];
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "audio_vocoder_runner 3D input shape [", dims[0], ", ", dims[1],
            ", ", dims[2], "] does not match tts_num_codebooks (",
            config.tts_num_codebooks, ")."));
      }
    } else if (dims.size() == 2) {
      voc_codebooks_dim_index = 0;
      voc_chunk_frames = dims[1];
    } else {
      return absl::InvalidArgumentError(
          "audio_vocoder_runner input must have rank 2 or 3.");
    }
    if (voc_chunk_frames <= 0) {
      return absl::InvalidArgumentError(
          "audio_vocoder_runner chunk_frames dimension must be positive.");
    }
    LITERT_ASSIGN_OR_RETURN(auto voc_out_type, voc_outputs[0].TensorType());
    if (voc_out_type.ElementType() != ElementType::Float32) {
      return absl::InvalidArgumentError(
          "Lfm2AudioGenerationStage audio_vocoder_runner output[0] must be "
          "Float32.");
    }
    LITERT_ASSIGN_OR_RETURN(size_t voc_out_elements,
                            voc_out_type.Layout().NumElements());
    if (voc_out_elements == 0 ||
        voc_out_elements % static_cast<size_t>(voc_chunk_frames) != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "audio_vocoder_runner output element count (", voc_out_elements,
          ") must be a positive multiple of voc_chunk_frames (",
          voc_chunk_frames, ")."));
    }
  }

  auto stage = absl::WrapUnique(new Lfm2AudioGenerationStage(
      frontend, std::move(tokenizer), std::move(llm_runner),
      std::move(tts_projector_runner), std::move(proj_inputs),
      std::move(proj_outputs), std::move(tts_decoder_runner),
      std::move(dec_inputs), std::move(dec_outputs),
      std::move(audio_vocoder_runner), std::move(voc_inputs),
      std::move(voc_outputs), voc_chunk_frames, voc_codebooks_dim_index,
      config));

  int im_end_id = config.im_end_token_id;
  if (auto tok_id = stage->tokenizer_->TokenToId(lfm2_audio::kImEndToken);
      tok_id.ok()) {
    im_end_id = *tok_id;
  }
  stage->turn_end_token_ids_.push_back(im_end_id);
  if (auto nl_ids = stage->tokenizer_->TextToTokenIds("\n");
      nl_ids.ok() && !nl_ids->empty()) {
    stage->turn_end_token_ids_.insert(stage->turn_end_token_ids_.end(),
                                      nl_ids->begin(), nl_ids->end());
  }

  return stage;
}

bool Lfm2AudioGenerationStage::NeedScheduleInternal() const {
  return frontend_.HasOutput();
}

absl::Status Lfm2AudioGenerationStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_ASSIGN_OR_RETURN(Lfm2AudioFrontendOutput prompt, frontend_.GetOutput());

  if (prompt.prompt_len <= 0 || prompt.prompt_embeddings.empty()) {
    return absl::InvalidArgumentError(
        "Lfm2AudioFrontendOutput has empty prompt_embeddings.");
  }
  if (prompt.prompt_embeddings.size() !=
      static_cast<size_t>(prompt.prompt_len) * hidden_dim_) {
    return absl::InvalidArgumentError(
        "Lfm2AudioFrontendOutput prompt_embeddings size mismatch.");
  }

  const int turn_end_len = static_cast<int>(turn_end_token_ids_.size());
  if (config_.max_kv_len > 0 &&
      kv_tokens_used_ + prompt.prompt_len + 1 + turn_end_len >
          config_.max_kv_len) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Prompt length (", prompt.prompt_len,
        ") plus existing session context (", kv_tokens_used_,
        ") and turn suffix (", turn_end_len,
        ") exceeds configured max_kv_len (", config_.max_kv_len, ")."));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto prefill_buf,
      support::CreateTensorBuffer<float>({1, prompt.prompt_len, hidden_dim_}));
  LITERT_RETURN_IF_ERROR(
      prefill_buf.Write<float>(absl::MakeConstSpan(prompt.prompt_embeddings)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs prefill_inputs,
                        CreateExecutorInputsWithAudio(prefill_buf));
  ABSL_RETURN_IF_ERROR(llm_runner_->Prefill(prefill_inputs));
  kv_tokens_used_ += prompt.prompt_len;

  enum class Modality { kText, kAudioOut };
  Modality modality = Modality::kText;
  const bool can_generate_audio = prompt.generate_audio &&
                                  tts_projector_runner_ != nullptr &&
                                  tts_decoder_runner_ != nullptr;
  Lfm2AudioGenerationMode effective_mode = prompt.generation_mode;
  if (config_.generation_mode != Lfm2AudioGenerationMode::kAuto) {
    effective_mode = config_.generation_mode;
  } else if (effective_mode == Lfm2AudioGenerationMode::kAuto) {
    effective_mode =
        absl::StrContainsIgnoreCase(config_.system_prompt, "interleaved")
            ? Lfm2AudioGenerationMode::kInterleaved
            : Lfm2AudioGenerationMode::kSequential;
  }
  resolved_session_mode_ = effective_mode;
  const bool interleaved_mode =
      can_generate_audio &&
      (effective_mode == Lfm2AudioGenerationMode::kInterleaved) &&
      config_.interleaved_n_text > 0 && config_.interleaved_n_audio > 0;
  int modality_left = std::max(1, config_.interleaved_n_text);
  bool text_done = false;
  bool audio_done = false;

  int32_t current_token = -1;
  bool has_pending_audio_emb = false;
  std::vector<float> pending_audio_emb(hidden_dim_, 0.0f);

  LITERT_ASSIGN_OR_RETURN(auto token_buf,
                          support::CreateTensorBuffer<int32_t>({1, 1}));
  LITERT_ASSIGN_OR_RETURN(
      auto audio_emb_buf,
      support::CreateTensorBuffer<float>({1, 1, hidden_dim_}));

  std::vector<int> generated_text_token_ids;
  std::vector<std::vector<int32_t>> generated_audio_frames;
  int max_steps = std::max(1, config_.max_new_tokens);
  if (config_.max_kv_len > 0) {
    const int available_decode_steps =
        std::max(1, config_.max_kv_len - kv_tokens_used_ - 1 - turn_end_len);
    if (available_decode_steps < max_steps) {
      ABSL_LOG(WARNING) << "Lfm2AudioGenerationStage capping max_new_tokens "
                           "from "
                        << max_steps << " to " << available_decode_steps
                        << " to stay within max_kv_len (" << config_.max_kv_len
                        << ").";
      max_steps = available_decode_steps;
    }
  }
  const int max_audio_frames = std::max(1, config_.max_tts_tokens);
  bool stopped_on_eos = false;

  for (int step = 0; step < max_steps; ++step) {
    --modality_left;
    lm::ExecutorInputs decode_inputs;
    if (has_pending_audio_emb) {
      LITERT_RETURN_IF_ERROR(
          audio_emb_buf.Write<float>(absl::MakeConstSpan(pending_audio_emb)));
      ABSL_ASSIGN_OR_RETURN(decode_inputs,
                            CreateExecutorInputsWithAudio(audio_emb_buf));
    } else {
      LITERT_RETURN_IF_ERROR(
          token_buf.Write<int32_t>(absl::MakeConstSpan(&current_token, 1)));
      ABSL_ASSIGN_OR_RETURN(decode_inputs,
                            CreateExecutorInputsWithText(token_buf));
    }

    ABSL_ASSIGN_OR_RETURN(TensorBuffer decode_buf,
                          llm_runner_->Decode(decode_inputs));
    ++kv_tokens_used_;
    ABSL_ASSIGN_OR_RETURN(
        DecodeStepOutput step_out,
        ExtractDecodeStepOutput(decode_buf, config_.llm_vocab_size, hidden_dim_,
                                /*require_hidden_state=*/can_generate_audio));

    if (modality == Modality::kText) {
      const int token_id = text_sampler_.Sample(step_out.logits);
      if (token_id == config_.im_end_token_id ||
          (!can_generate_audio && (token_id == config_.text_end_token_id ||
                                   token_id == config_.audio_start_token_id))) {
        stopped_on_eos = (token_id == config_.im_end_token_id);
        has_pending_audio_emb = false;
        current_token = token_id;
        break;
      }
      // Runtime guard (not in the Python reference loop): once both
      // `<|text_end|>` (`text_done`) and `EOAudio` (`audio_done`) have been
      // emitted, end the turn immediately if the next text step does not emit
      // `<|im_end|>`, preventing post-`EOAudio` modality re-entry or trailing
      // tokens.
      if (text_done && audio_done) {
        has_pending_audio_emb = false;
        current_token = token_id;
        break;
      }
      if (token_id == config_.text_end_token_id) {
        text_done = true;
      }
      if (can_generate_audio && !audio_done &&
          (token_id == config_.audio_start_token_id ||
           (interleaved_mode && (modality_left <= 0 || text_done)))) {
        if (!text_done && token_id != config_.text_end_token_id &&
            token_id != config_.pad_token_id &&
            token_id != config_.bos_token_id &&
            token_id != config_.im_start_token_id &&
            token_id != config_.audio_start_token_id) {
          generated_text_token_ids.push_back(token_id);
        }
        modality = Modality::kAudioOut;
        modality_left = std::max(1, config_.interleaved_n_audio);
        current_token = token_id;
        has_pending_audio_emb = false;
        continue;
      }
      if (!text_done && token_id != config_.text_end_token_id &&
          token_id != config_.pad_token_id &&
          token_id != config_.bos_token_id &&
          token_id != config_.im_start_token_id &&
          token_id != config_.audio_start_token_id) {
        generated_text_token_ids.push_back(token_id);
      }
      current_token = token_id;
      has_pending_audio_emb = false;
    } else {
      const bool force_eo_audio =
          static_cast<int>(generated_audio_frames.size()) >= max_audio_frames;
      if (force_eo_audio) {
        ABSL_LOG(WARNING) << "Lfm2AudioGenerationStage reached max_tts_tokens ("
                          << max_audio_frames
                          << " audio frames); forcing EOAudio.";
      }
      ABSL_ASSIGN_OR_RETURN(
          AudioFrameResult frame_res,
          GenerateAudioFrame(step_out.hidden_state, force_eo_audio));
      if (frame_res.is_eo_audio) {
        modality = Modality::kText;
        modality_left = std::max(1, config_.interleaved_n_text);
        if (text_done || !interleaved_mode) {
          audio_done = true;
        }
      } else {
        generated_audio_frames.push_back(std::move(frame_res.codes));
        if (interleaved_mode && modality_left <= 0 && !text_done) {
          modality = Modality::kText;
          modality_left = std::max(1, config_.interleaved_n_text);
        }
      }
      pending_audio_emb = std::move(frame_res.audio_embedding);
      has_pending_audio_emb = true;
    }
  }

  // Commit the final token/embedding and `<|im_end|>\n` into the LFM2 KV cache.
  if (!stopped_on_eos) {
    if (has_pending_audio_emb) {
      LITERT_RETURN_IF_ERROR(
          audio_emb_buf.Write<float>(absl::MakeConstSpan(pending_audio_emb)));
      ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs dec_in,
                            CreateExecutorInputsWithAudio(audio_emb_buf));
      ABSL_RETURN_IF_ERROR(llm_runner_->Decode(dec_in).status());
      ++kv_tokens_used_;
    } else if (current_token >= 0) {
      LITERT_RETURN_IF_ERROR(
          token_buf.Write<int32_t>(absl::MakeConstSpan(&current_token, 1)));
      ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs dec_in,
                            CreateExecutorInputsWithText(token_buf));
      ABSL_RETURN_IF_ERROR(llm_runner_->Decode(dec_in).status());
      ++kv_tokens_used_;
    }
  }
  for (int tok : turn_end_token_ids_) {
    const int32_t tok32 = static_cast<int32_t>(tok);
    LITERT_RETURN_IF_ERROR(
        token_buf.Write<int32_t>(absl::MakeConstSpan(&tok32, 1)));
    ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs dec_in,
                          CreateExecutorInputsWithText(token_buf));
    ABSL_RETURN_IF_ERROR(llm_runner_->Decode(dec_in).status());
    ++kv_tokens_used_;
  }

  if (!generated_text_token_ids.empty()) {
    ABSL_ASSIGN_OR_RETURN(std::string text,
                          tokenizer_->TokenIdsToText(generated_text_token_ids));
    if (!text.empty()) {
      PushOutput(Output(TextOutput{.confirmed_text = std::move(text)}));
    }
  }

  if (!generated_audio_frames.empty() && audio_vocoder_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(std::vector<float> waveform,
                          SynthesizeWaveform(generated_audio_frames));
    if (!waveform.empty()) {
      PushOutput(Output(AudioOutput{
          .pcm_samples = std::move(waveform),
          .sample_rate_hz = config_.audio_output_sample_rate_hz,
      }));
    }
  }

  PushOutput(Output(EndOfOutputTurn()));
  return absl::OkStatus();
}

void Lfm2AudioGenerationStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  resolved_session_mode_ = Lfm2AudioGenerationMode::kAuto;
  kv_tokens_used_ = 0;
  text_sampler_.Reset();
  tts_sampler_.Reset();
  (void)llm_runner_->Reset();
}

absl::StatusOr<Lfm2AudioGenerationStage::AudioFrameResult>
Lfm2AudioGenerationStage::GenerateAudioFrame(
    absl::Span<const float> llm_hidden_state, bool force_eo_audio) {
  const int codebooks = config_.tts_num_codebooks;
  const int df_dim = config_.tts_hidden_size;
  const int df_vocab = config_.tts_vocab_size;
  const int eo_audio_code = df_vocab - 1;

  // 1. Run `tts_projector_runner_` (`depth_linear`) on `llm_hidden_state` to
  // get `depthformer_in` of shape `[1, codebooks, df_dim]`.
  std::vector<int32_t> frame_codes(codebooks, 0);
  LITERT_RETURN_IF_ERROR(proj_input_buffers_[0].Write<float>(llm_hidden_state));
  LITERT_RETURN_IF_ERROR(
      proj_input_buffers_[1].Write<int32_t>(absl::MakeConstSpan(frame_codes)));
  ABSL_RETURN_IF_ERROR(tts_projector_runner_->Run(
      /*signature_name=*/"", proj_input_buffers_, proj_output_buffers_));

  std::vector<float> depthformer_in(static_cast<size_t>(codebooks) * df_dim,
                                    0.0f);
  LITERT_RETURN_IF_ERROR(
      proj_output_buffers_[0].Read<float>(absl::MakeSpan(depthformer_in)));

  // 2. Run 8-codebook RQ-Transformer Depthformer (`tts_decoder_runner_`).
  LITERT_ASSIGN_OR_RETURN(auto kv_type, dec_input_buffers_[3].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t kv_elements, kv_type.Layout().NumElements());
  std::vector<float> kv_cache(kv_elements, 0.0f);
  std::vector<float> logits(df_vocab, 0.0f);

  AudioFrameResult result;
  result.codes.assign(codebooks, 0);
  int32_t prev_code = 0;

  if (force_eo_audio) {
    result.codes.assign(codebooks, eo_audio_code);
    result.is_eo_audio = true;
  } else {
    for (int k = 0; k < codebooks; ++k) {
      absl::Span<const float> cond_k(
          depthformer_in.data() + static_cast<size_t>(k) * df_dim, df_dim);
      const int32_t step_idx = static_cast<int32_t>(k);

      LITERT_RETURN_IF_ERROR(dec_input_buffers_[0].Write<float>(cond_k));
      LITERT_RETURN_IF_ERROR(dec_input_buffers_[1].Write<int32_t>(
          absl::MakeConstSpan(&prev_code, 1)));
      LITERT_RETURN_IF_ERROR(dec_input_buffers_[2].Write<int32_t>(
          absl::MakeConstSpan(&step_idx, 1)));
      LITERT_RETURN_IF_ERROR(
          dec_input_buffers_[3].Write<float>(absl::MakeConstSpan(kv_cache)));

      ABSL_RETURN_IF_ERROR(tts_decoder_runner_->Run(
          /*signature_name=*/"", dec_input_buffers_, dec_output_buffers_));
      LITERT_RETURN_IF_ERROR(
          dec_output_buffers_[0].Read<float>(absl::MakeSpan(logits)));
      LITERT_RETURN_IF_ERROR(
          dec_output_buffers_[1].Read<float>(absl::MakeSpan(kv_cache)));

      if (k > 0 && eo_audio_code >= 0 &&
          static_cast<size_t>(eo_audio_code) < logits.size()) {
        // Runtime safeguard: only codebook 0 may emit `EOAudio` (`2048`),
        // because `LFM2AudioDetokenizer` only embeds codec tokens in `[0,
        // 2047]`. Mask `logits[eo_audio_code] = -inf` on codebooks 1..7 so a
        // non-zero codebook never samples `EOAudio`.
        logits[eo_audio_code] = -std::numeric_limits<float>::infinity();
      }

      const int sampled_code = tts_sampler_.Sample(logits);
      if (k == 0 && sampled_code == eo_audio_code) {
        result.codes.assign(codebooks, eo_audio_code);
        result.is_eo_audio = true;
        break;
      }
      result.codes[k] = static_cast<int32_t>(sampled_code);
      prev_code = static_cast<int32_t>(sampled_code);
    }
  }

  // 3. Embed `result.codes` via `tts_projector_runner_`
  // (`audio_embedding(codes + codebook_offsets).sum(0)`) to feed back into the
  // LFM2 backbone on the next step.
  LITERT_RETURN_IF_ERROR(proj_input_buffers_[0].Write<float>(llm_hidden_state));
  LITERT_RETURN_IF_ERROR(
      proj_input_buffers_[1].Write<int32_t>(absl::MakeConstSpan(result.codes)));
  ABSL_RETURN_IF_ERROR(tts_projector_runner_->Run(
      /*signature_name=*/"", proj_input_buffers_, proj_output_buffers_));

  result.audio_embedding.assign(hidden_dim_, 0.0f);
  LITERT_RETURN_IF_ERROR(proj_output_buffers_[1].Read<float>(
      absl::MakeSpan(result.audio_embedding)));
  return result;
}

absl::StatusOr<std::vector<float>> Lfm2AudioGenerationStage::SynthesizeWaveform(
    const std::vector<std::vector<int32_t>>& audio_frames) {
  if (audio_frames.empty() || audio_vocoder_runner_ == nullptr) {
    return std::vector<float>();
  }

  const int codebooks = config_.tts_num_codebooks;
  const int detok_vocab = config_.tts_vocab_size - 1;
  for (size_t f = 0; f < audio_frames.size(); ++f) {
    if (static_cast<int>(audio_frames[f].size()) != codebooks) {
      return absl::InvalidArgumentError(
          "Audio frame codebook count mismatch in SynthesizeWaveform.");
    }
    for (int k = 0; k < codebooks; ++k) {
      const int32_t code = audio_frames[f][k];
      if (code < 0 || code >= detok_vocab) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Audio code (", code, ") at frame ", f, ", codebook ", k,
            " is out of detokenizer vocabulary range [0, ", detok_vocab, ")."));
      }
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto voc_in_type, voc_input_buffers_[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t in_elements,
                          voc_in_type.Layout().NumElements());
  LITERT_ASSIGN_OR_RETURN(auto voc_out_type,
                          voc_output_buffers_[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t out_elements,
                          voc_out_type.Layout().NumElements());

  const int chunk_frames = std::max(1, voc_chunk_frames_);
  const size_t samples_per_frame =
      (out_elements % static_cast<size_t>(chunk_frames) == 0)
          ? (out_elements / static_cast<size_t>(chunk_frames))
          : static_cast<size_t>(config_.detokenizer_upsample_samples);

  std::vector<int32_t> chunk_in(in_elements, -1);
  std::vector<float> chunk_out(out_elements, 0.0f);
  std::vector<float> waveform;
  waveform.reserve(audio_frames.size() * samples_per_frame);

  auto run_vocoder_window = [&](int win_start, int valid) -> absl::Status {
    // Fill with `-1` so `Lfm2AudioDetokenizerModule` masks out any right-padded
    // slots `t >= valid` in `ISTFT`.
    absl::c_fill(chunk_in, -1);
    if (voc_codebooks_dim_index_ == 1) {
      // `[1, codebooks, chunk_frames]` (`LFM2AudioDetokenizer` layout).
      for (int t = 0; t < valid; ++t) {
        for (int k = 0; k < codebooks; ++k) {
          chunk_in[static_cast<size_t>(k) * chunk_frames + t] =
              audio_frames[win_start + t][k];
        }
      }
    } else if (voc_codebooks_dim_index_ == 2) {
      // `[1, chunk_frames, codebooks]`.
      for (int t = 0; t < valid; ++t) {
        for (int k = 0; k < codebooks; ++k) {
          chunk_in[static_cast<size_t>(t) * codebooks + k] =
              audio_frames[win_start + t][k];
        }
      }
    } else {
      // Rank-2 `[1, chunk_frames]` fallback.
      for (int t = 0; t < valid; ++t) {
        chunk_in[t] = audio_frames[win_start + t][0];
      }
    }

    LITERT_RETURN_IF_ERROR(
        voc_input_buffers_[0].Write<int32_t>(absl::MakeConstSpan(chunk_in)));
    ABSL_RETURN_IF_ERROR(audio_vocoder_runner_->Run(
        /*signature_name=*/"", voc_input_buffers_, voc_output_buffers_));
    LITERT_RETURN_IF_ERROR(
        voc_output_buffers_[0].Read<float>(absl::MakeSpan(chunk_out)));
    return absl::OkStatus();
  };

  const int total_frames = static_cast<int>(audio_frames.size());
  if (total_frames <= chunk_frames) {
    ABSL_RETURN_IF_ERROR(run_vocoder_window(/*win_start=*/0, total_frames));
    const size_t valid_samples = std::min(
        out_elements, static_cast<size_t>(total_frames) * samples_per_frame);
    waveform.insert(waveform.end(), chunk_out.begin(),
                    chunk_out.begin() + valid_samples);
    return waveform;
  }

  // Overlap-discard chunked detokenization:
  // `LFM2AudioDetokenizer` is causal with a receptive field of 17 code frames
  // in `DetokLfm2Model` plus 1 code frame on each side in `ISTFT` (`kLeftCtx =
  // 18`, `kRightCtx = 1`). When `chunk_frames >= 20`, we warm up each chunk
  // with >=18 left context frames and hold back 1 right context frame so
  // chunked output is identical to one-shot detokenization without boundary
  // splices.
  const int min_full_ctx_chunk_frames =
      lfm2_audio::kDetokenizerLeftContextFrames +
      lfm2_audio::kDetokenizerRightContextFrames + 1;
  const int left_ctx = (chunk_frames >= min_full_ctx_chunk_frames)
                           ? lfm2_audio::kDetokenizerLeftContextFrames
                           : ((chunk_frames >= 6) ? (chunk_frames / 2 - 1) : 0);
  const int right_ctx =
      (left_ctx > 0) ? lfm2_audio::kDetokenizerRightContextFrames : 0;
  if (chunk_frames < min_full_ctx_chunk_frames) {
    ABSL_LOG(WARNING) << "Lfm2AudioGenerationStage vocoder chunk_frames ("
                      << chunk_frames
                      << ") is smaller than the full causal receptive field ("
                      << min_full_ctx_chunk_frames
                      << " frames); falling back to left_ctx=" << left_ctx
                      << ", right_ctx=" << right_ctx
                      << ", which may introduce chunk-boundary artifacts.";
  }

  if (left_ctx == 0) {
    for (int start = 0; start < total_frames; start += chunk_frames) {
      const int valid = std::min(chunk_frames, total_frames - start);
      ABSL_RETURN_IF_ERROR(run_vocoder_window(start, valid));
      const size_t valid_samples = std::min(
          out_elements, static_cast<size_t>(valid) * samples_per_frame);
      waveform.insert(waveform.end(), chunk_out.begin(),
                      chunk_out.begin() + valid_samples);
    }
    return waveform;
  }

  // First chunk `[0 .. chunk_frames - 1]`: emit `[0 .. chunk_frames -
  // right_ctx)`.
  ABSL_RETURN_IF_ERROR(run_vocoder_window(/*win_start=*/0, chunk_frames));
  int emit_cursor = chunk_frames - right_ctx;
  const size_t first_samples = std::min(
      out_elements, static_cast<size_t>(emit_cursor) * samples_per_frame);
  waveform.insert(waveform.end(), chunk_out.begin(),
                  chunk_out.begin() + first_samples);

  while (emit_cursor < total_frames) {
    const int remaining = total_frames - emit_cursor;
    if (remaining <= chunk_frames - left_ctx) {
      // Final chunk: anchor window at `total_frames - chunk_frames` so it is
      // 100% full (`chunk_frames` valid frames, zero right-padding) with
      // `left_discard = chunk_frames - remaining >= left_ctx`.
      const int win_start = total_frames - chunk_frames;
      const int left_discard = emit_cursor - win_start;
      ABSL_RETURN_IF_ERROR(run_vocoder_window(win_start, chunk_frames));
      const size_t start_sample = std::min(
          out_elements, static_cast<size_t>(left_discard) * samples_per_frame);
      const size_t end_sample = std::min(
          out_elements, static_cast<size_t>(chunk_frames) * samples_per_frame);
      waveform.insert(waveform.end(), chunk_out.begin() + start_sample,
                      chunk_out.begin() + end_sample);
      emit_cursor = total_frames;
    } else {
      // Middle chunk: start at `emit_cursor - left_ctx`, discard `left_ctx`
      // left warmup frames and `right_ctx` right lookahead frame.
      const int win_start = emit_cursor - left_ctx;
      ABSL_RETURN_IF_ERROR(run_vocoder_window(win_start, chunk_frames));
      const size_t start_sample = std::min(
          out_elements, static_cast<size_t>(left_ctx) * samples_per_frame);
      const size_t end_sample =
          std::min(out_elements, static_cast<size_t>(chunk_frames - right_ctx) *
                                     samples_per_frame);
      waveform.insert(waveform.end(), chunk_out.begin() + start_sample,
                      chunk_out.begin() + end_sample);
      emit_cursor += chunk_frames - left_ctx - right_ctx;
    }
  }

  return waveform;
}

}  // namespace litert::omni::any2any
