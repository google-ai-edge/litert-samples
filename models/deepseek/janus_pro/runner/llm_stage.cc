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

#include "models/deepseek/janus_pro/runner/llm_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/runner/io_types.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/any2any/any2any_sampler.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/utils/tensor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "tflite/types/half.h"

namespace litert::omni::any2any {
namespace {

absl::StatusOr<std::vector<float>> ReadTensorBufferAsFloatVector(
    const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  const ElementType elem_type = tensor_type.ElementType();
  if (elem_type == ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(std::vector<float> out,
                            support::CopyFromTensorBuffer<float>(buffer));
    return out;
  }
  if (elem_type == ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(
        std::vector<tflite::half> fp16,
        support::CopyFromTensorBuffer<tflite::half>(buffer));
    return std::vector<float>(fp16.begin(), fp16.end());
  }
  return absl::InvalidArgumentError(
      "Unsupported TensorBuffer element type; expected Float32 or Float16.");
}

absl::Status PrefillEmbeddings(LiteRtLmRunner& llm_runner,
                               absl::Span<const float> embeddings,
                               int prompt_len, int hidden_dim) {
  if (prompt_len <= 0 || embeddings.empty()) {
    return absl::InvalidArgumentError(
        "JanusProFrontendChunk has empty prompt embeddings.");
  }
  if (embeddings.size() != static_cast<size_t>(prompt_len) * hidden_dim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "JanusProFrontendChunk embeddings size (", embeddings.size(),
        ") does not match prompt_len * llm_hidden_size (",
        static_cast<size_t>(prompt_len) * hidden_dim, ")."));
  }

  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer prefill_buf,
      support::CreateTensorBuffer<float>({1, prompt_len, hidden_dim}));
  LITERT_RETURN_IF_ERROR(prefill_buf.Write<float>(embeddings));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs prefill_inputs,
                        CreateExecutorInputsWithAudio(prefill_buf));
  return llm_runner.Prefill(prefill_inputs);
}

struct DecodeStepOutput {
  std::vector<float> logits;
  std::vector<float> hidden_state;
};

absl::StatusOr<DecodeStepOutput> SplitDecodeOutput(std::vector<float> raw_out,
                                                   int vocab_size,
                                                   int hidden_dim) {
  if (raw_out.empty()) {
    return absl::InternalError("LLM Decode returned an empty TensorBuffer.");
  }
  DecodeStepOutput step_out;
  const int raw_len = static_cast<int>(raw_out.size());
  if (raw_len == vocab_size + hidden_dim) {
    step_out.hidden_state.assign(raw_out.begin() + vocab_size, raw_out.end());
    raw_out.resize(vocab_size);
  } else if (raw_len != vocab_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("LLM Decode output size (", raw_len,
                     ") does not match llm_vocab_size (", vocab_size,
                     ") or llm_vocab_size + llm_hidden_size (",
                     vocab_size + hidden_dim, ")."));
  }
  step_out.logits = std::move(raw_out);
  return step_out;
}

absl::StatusOr<DecodeStepOutput> RunTextDecodeStep(LiteRtLmRunner& llm_runner,
                                                   TensorBuffer& token_buf,
                                                   int32_t input_token,
                                                   int vocab_size,
                                                   int hidden_dim) {
  LITERT_RETURN_IF_ERROR(
      token_buf.Write<int32_t>(absl::MakeConstSpan(&input_token, 1)));
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs decode_inputs,
                        CreateExecutorInputsWithText(token_buf));
  ABSL_ASSIGN_OR_RETURN(TensorBuffer decode_buf,
                        llm_runner.Decode(decode_inputs));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> raw_out,
                        ReadTensorBufferAsFloatVector(decode_buf));
  return SplitDecodeOutput(std::move(raw_out), vocab_size, hidden_dim);
}

absl::StatusOr<DecodeStepOutput> RunEmbeddingDecodeStep(
    LiteRtLmRunner& llm_runner, const TensorBuffer& emb_buf, int vocab_size,
    int hidden_dim) {
  ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs decode_inputs,
                        CreateExecutorInputsWithVision(emb_buf));
  ABSL_ASSIGN_OR_RETURN(TensorBuffer decode_buf,
                        llm_runner.Decode(decode_inputs));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> raw_out,
                        ReadTensorBufferAsFloatVector(decode_buf));
  return SplitDecodeOutput(std::move(raw_out), vocab_size, hidden_dim);
}

absl::StatusOr<std::vector<float>> RunGenHead(
    LiteRtRunner& gen_head_runner, absl::Span<TensorBuffer> input_buffers,
    absl::Span<TensorBuffer> output_buffers,
    absl::Span<const float> hidden_state, int codebook_size) {
  LITERT_RETURN_IF_ERROR(input_buffers[0].Write<float>(hidden_state));
  ABSL_RETURN_IF_ERROR(gen_head_runner.Run(/*signature_name=*/"", input_buffers,
                                           output_buffers));
  std::vector<float> logits(codebook_size);
  LITERT_RETURN_IF_ERROR(output_buffers[0].Read<float>(absl::MakeSpan(logits)));
  return logits;
}

}  // namespace

JanusProLlmStage::JanusProLlmStage(
    Stage<JanusProFrontendChunk>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    std::shared_ptr<LiteRtLmRunner> absl_nullable uncond_llm_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable vq_image_embedder_runner,
    std::vector<TensorBuffer> vq_emb_input_buffers,
    std::vector<TensorBuffer> vq_emb_output_buffers,
    std::unique_ptr<LiteRtRunner> absl_nullable gen_head_runner,
    std::vector<TensorBuffer> gen_head_input_buffers,
    std::vector<TensorBuffer> gen_head_output_buffers,
    JanusProModelConfig config)
    : frontend_(*frontend),
      config_(std::move(config)),
      tokenizer_(std::move(tokenizer)),
      llm_runner_(std::move(llm_runner)),
      uncond_llm_runner_(std::move(uncond_llm_runner)),
      vq_image_embedder_runner_(std::move(vq_image_embedder_runner)),
      vq_emb_input_buffers_(std::move(vq_emb_input_buffers)),
      vq_emb_output_buffers_(std::move(vq_emb_output_buffers)),
      gen_head_runner_(std::move(gen_head_runner)),
      gen_head_input_buffers_(std::move(gen_head_input_buffers)),
      gen_head_output_buffers_(std::move(gen_head_output_buffers)),
      hidden_dim_(config_.llm_hidden_size),
      text_sampler_(Any2AnySampler::Options{
          .do_sample = config_.do_sample,
          .temperature = config_.temperature,
          .top_k = config_.top_k,
          .top_p = config_.top_p,
          .seed = config_.seed,
      }),
      image_sampler_(Any2AnySampler::Options{
          .do_sample = config_.image_temperature > 0.0f,
          .temperature = config_.image_temperature,
          .top_k = 0,
          .top_p = 1.0f,
          .seed = config_.seed,
      }) {}

JanusProLlmStage::~JanusProLlmStage() = default;

absl::StatusOr<std::unique_ptr<JanusProLlmStage>> JanusProLlmStage::Create(
    Stage<JanusProFrontendChunk>* absl_nonnull frontend,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
    std::shared_ptr<LiteRtLmRunner> absl_nullable uncond_llm_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable vq_image_embedder_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable gen_head_runner,
    const JanusProModelConfig& config) {
  if (config.llm_hidden_size <= 0 || config.llm_vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "JanusProLlmStage config.llm_hidden_size and config.llm_vocab_size "
        "must be positive.");
  }

  std::vector<TensorBuffer> vq_emb_in;
  std::vector<TensorBuffer> vq_emb_out;
  if (vq_image_embedder_runner != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        vq_emb_in,
        vq_image_embedder_runner->CreateInputBuffers(/*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        vq_emb_out,
        vq_image_embedder_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (vq_emb_in.empty() || vq_emb_out.empty()) {
      return absl::InvalidArgumentError(
          "JanusProLlmStage vq_image_embedder_runner must provide non-empty "
          "input and output buffers.");
    }
    ABSL_RETURN_IF_ERROR(ValidateTensorBuffer(
        vq_emb_in[0], ElementType::Int32, /*expected_elements=*/1,
        "JanusProLlmStage vq_image_embedder input"));
    ABSL_RETURN_IF_ERROR(
        ValidateTensorBuffer(vq_emb_out[0], ElementType::Float32,
                             static_cast<size_t>(config.llm_hidden_size),
                             "JanusProLlmStage vq_image_embedder output"));
  }

  std::vector<TensorBuffer> gen_head_in;
  std::vector<TensorBuffer> gen_head_out;
  if (gen_head_runner != nullptr) {
    if (config.vq_codebook_size <= 0) {
      return absl::InvalidArgumentError(
          "JanusProLlmStage config.vq_codebook_size must be positive when "
          "gen_head_runner is provided.");
    }
    ABSL_ASSIGN_OR_RETURN(gen_head_in, gen_head_runner->CreateInputBuffers(
                                           /*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(gen_head_out, gen_head_runner->CreateOutputBuffers(
                                            /*signature_name=*/""));
    if (gen_head_in.empty() || gen_head_out.empty()) {
      return absl::InvalidArgumentError(
          "JanusProLlmStage gen_head_runner must provide non-empty input and "
          "output buffers.");
    }
    ABSL_RETURN_IF_ERROR(
        ValidateTensorBuffer(gen_head_in[0], ElementType::Float32,
                             static_cast<size_t>(config.llm_hidden_size),
                             "JanusProLlmStage gen_head input"));
    ABSL_RETURN_IF_ERROR(
        ValidateTensorBuffer(gen_head_out[0], ElementType::Float32,
                             static_cast<size_t>(config.vq_codebook_size),
                             "JanusProLlmStage gen_head output"));
  }

  auto stage = absl::WrapUnique(new JanusProLlmStage(
      frontend, std::move(tokenizer), std::move(llm_runner),
      std::move(uncond_llm_runner), std::move(vq_image_embedder_runner),
      std::move(vq_emb_in), std::move(vq_emb_out), std::move(gen_head_runner),
      std::move(gen_head_in), std::move(gen_head_out), config));

  absl::StatusOr<int> eos_tok =
      stage->tokenizer_->TokenToId(janus_pro::kEosToken);
  stage->stop_token_ids_.insert(eos_tok.ok() ? *eos_tok
                                             : janus_pro::kEosTokenId);
  for (absl::string_view stop_str : {"<|endoftext|>", "</s>"}) {
    absl::StatusOr<int> tok_id = stage->tokenizer_->TokenToId(stop_str);
    if (tok_id.ok()) {
      stage->stop_token_ids_.insert(*tok_id);
    }
  }

  return stage;
}

bool JanusProLlmStage::NeedScheduleInternal() const {
  return frontend_.HasOutput();
}

absl::Status JanusProLlmStage::RunUnderstandingTurn(
    const JanusProFrontendChunk& prompt) {
  if (last_turn_was_image_gen_) {
    ABSL_RETURN_IF_ERROR(llm_runner_->Reset());
    last_turn_was_image_gen_ = false;
  }
  ABSL_RETURN_IF_ERROR(PrefillEmbeddings(*llm_runner_, prompt.cond_embeddings,
                                         prompt.prompt_len, hidden_dim_));

  int32_t current_token = -1;
  LITERT_ASSIGN_OR_RETURN(TensorBuffer token_buf,
                          support::CreateTensorBuffer<int32_t>({1, 1}));
  const int max_steps = std::max(1, config_.max_new_tokens);
  bool stopped_on_eos = false;
  std::vector<int> generated_token_ids;

  for (int step = 0; step < max_steps; ++step) {
    ABSL_ASSIGN_OR_RETURN(
        DecodeStepOutput step_out,
        RunTextDecodeStep(*llm_runner_, token_buf, current_token,
                          config_.llm_vocab_size, hidden_dim_));
    const int token_id = text_sampler_.Sample(step_out.logits);
    if (stop_token_ids_.contains(token_id)) {
      stopped_on_eos = true;
      break;
    }
    generated_token_ids.push_back(token_id);
    current_token = token_id;
  }

  if (!stopped_on_eos && current_token >= 0) {
    LITERT_RETURN_IF_ERROR(
        token_buf.Write<int32_t>(absl::MakeConstSpan(&current_token, 1)));
    ABSL_ASSIGN_OR_RETURN(lm::ExecutorInputs decode_inputs,
                          CreateExecutorInputsWithText(token_buf));
    ABSL_RETURN_IF_ERROR(llm_runner_->Decode(decode_inputs).status());
  }

  std::string text;
  if (!generated_token_ids.empty()) {
    ABSL_ASSIGN_OR_RETURN(std::string raw_text,
                          tokenizer_->TokenIdsToText(generated_token_ids));
    text = std::string(absl::StripLeadingAsciiWhitespace(raw_text));
  }
  PushOutput(JanusProLlmOutput{.payload = std::move(text)});
  return absl::OkStatus();
}

absl::Status JanusProLlmStage::RunImageGenerationTurn(
    const JanusProFrontendChunk& prompt) {
  if (uncond_llm_runner_ == nullptr || vq_image_embedder_runner_ == nullptr ||
      gen_head_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "JanusProLlmStage requires uncond_llm_runner, "
        "vq_image_embedder_runner, and gen_head_runner when generate_image is "
        "enabled.");
  }
  if (config_.num_image_tokens <= 0 || config_.vq_codebook_size <= 0) {
    return absl::InvalidArgumentError(
        "JanusProLlmStage config.num_image_tokens and config.vq_codebook_size "
        "must be positive.");
  }

  ABSL_RETURN_IF_ERROR(llm_runner_->Reset());
  ABSL_RETURN_IF_ERROR(uncond_llm_runner_->Reset());
  last_turn_was_image_gen_ = true;

  ABSL_RETURN_IF_ERROR(PrefillEmbeddings(*llm_runner_, prompt.cond_embeddings,
                                         prompt.prompt_len, hidden_dim_));
  ABSL_RETURN_IF_ERROR(
      PrefillEmbeddings(*uncond_llm_runner_, prompt.uncond_embeddings,
                        prompt.uncond_prompt_len, hidden_dim_));

  LITERT_ASSIGN_OR_RETURN(TensorBuffer token_buf,
                          support::CreateTensorBuffer<int32_t>({1, 1}));
  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer step_emb_buf,
      support::CreateTensorBuffer<float>({1, 1, hidden_dim_}));

  JanusProVqCodeGrid grid;
  grid.vq_codes.reserve(config_.num_image_tokens);
  grid.image_width = config_.image_size;
  grid.image_height = config_.image_size;

  int32_t prev_vq_code = 0;
  std::vector<float> vq_emb(hidden_dim_);
  std::vector<float> cfg_logits(config_.vq_codebook_size);

  for (int step = 0; step < config_.num_image_tokens; ++step) {
    DecodeStepOutput cond_step;
    DecodeStepOutput uncond_step;
    if (step == 0) {
      ABSL_ASSIGN_OR_RETURN(
          cond_step, RunTextDecodeStep(*llm_runner_, token_buf, -1,
                                       config_.llm_vocab_size, hidden_dim_));
      ABSL_ASSIGN_OR_RETURN(
          uncond_step, RunTextDecodeStep(*uncond_llm_runner_, token_buf, -1,
                                         config_.llm_vocab_size, hidden_dim_));
    } else {
      LITERT_RETURN_IF_ERROR(vq_emb_input_buffers_[0].Write<int32_t>(
          absl::MakeConstSpan(&prev_vq_code, 1)));
      ABSL_RETURN_IF_ERROR(vq_image_embedder_runner_->Run(
          /*signature_name=*/"", vq_emb_input_buffers_,
          vq_emb_output_buffers_));
      LITERT_RETURN_IF_ERROR(
          vq_emb_output_buffers_[0].Read<float>(absl::MakeSpan(vq_emb)));
      LITERT_RETURN_IF_ERROR(
          step_emb_buf.Write<float>(absl::MakeConstSpan(vq_emb)));

      ABSL_ASSIGN_OR_RETURN(
          cond_step,
          RunEmbeddingDecodeStep(*llm_runner_, step_emb_buf,
                                 config_.llm_vocab_size, hidden_dim_));
      ABSL_ASSIGN_OR_RETURN(
          uncond_step,
          RunEmbeddingDecodeStep(*uncond_llm_runner_, step_emb_buf,
                                 config_.llm_vocab_size, hidden_dim_));
    }

    if (cond_step.hidden_state.empty() || uncond_step.hidden_state.empty()) {
      return absl::FailedPreconditionError(
          "JanusProLlmStage requires last-layer hidden states from llm_runner "
          "and uncond_llm_runner when generate_image is enabled.");
    }

    ABSL_ASSIGN_OR_RETURN(
        std::vector<float> logits_cond,
        RunGenHead(*gen_head_runner_, absl::MakeSpan(gen_head_input_buffers_),
                   absl::MakeSpan(gen_head_output_buffers_),
                   cond_step.hidden_state, config_.vq_codebook_size));
    ABSL_ASSIGN_OR_RETURN(
        std::vector<float> logits_uncond,
        RunGenHead(*gen_head_runner_, absl::MakeSpan(gen_head_input_buffers_),
                   absl::MakeSpan(gen_head_output_buffers_),
                   uncond_step.hidden_state, config_.vq_codebook_size));

    for (int i = 0; i < config_.vq_codebook_size; ++i) {
      cfg_logits[i] = logits_uncond[i] +
                      config_.cfg_weight * (logits_cond[i] - logits_uncond[i]);
    }

    const int32_t sampled_code =
        static_cast<int32_t>(image_sampler_.Sample(cfg_logits));
    grid.vq_codes.push_back(sampled_code);
    prev_vq_code = sampled_code;
  }

  PushOutput(JanusProLlmOutput{.payload = std::move(grid)});
  return absl::OkStatus();
}

absl::Status JanusProLlmStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_ASSIGN_OR_RETURN(JanusProFrontendChunk prompt, frontend_.GetOutput());
  if (prompt.generate_image) {
    return RunImageGenerationTurn(prompt);
  }
  return RunUnderstandingTurn(prompt);
}

void JanusProLlmStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  last_turn_was_image_gen_ = false;
  text_sampler_.Reset();
  image_sampler_.Reset();
  (void)llm_runner_->Reset();
  if (uncond_llm_runner_ != nullptr) {
    (void)uncond_llm_runner_->Reset();
  }
}

}  // namespace litert::omni::any2any
