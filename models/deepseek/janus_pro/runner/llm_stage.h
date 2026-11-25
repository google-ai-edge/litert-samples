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

#ifndef MODELS_DEEPSEEK_JANUS_PRO_RUNNER_LLM_STAGE_H_
#define MODELS_DEEPSEEK_JANUS_PRO_RUNNER_LLM_STAGE_H_

#include <memory>
#include <vector>

#include "models/deepseek/janus_pro/runner/io_types.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/any2any/any2any_sampler.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Stage 2: Autoregressive Multimodal LLM Backbone Stage for DeepSeek Janus-Pro.
// - In Multimodal Understanding mode (`generate_image == false`):
//   Prefills `llm_runner` with `cond_embeddings`, decodes text tokens up to
//   `max_new_tokens` or `<｜end▁of▁sentence｜>`, detokenizes the assistant
//   response, and emits `JanusProLlmOutput` holding `std::string`.
// - In Visual Generation mode (`generate_image == true`):
//   Prefills `llm_runner` (`cond_embeddings`) and `uncond_llm_runner`
//   (`uncond_embeddings`), runs `num_image_tokens` (`576`) autoregressive
//   decode steps feeding each step's sampled VQ code back through
//   `vq_image_embedder_runner` into both runners, projects `cond_hidden` and
//   `uncond_hidden` through `gen_head_runner`, combines logits via
//   Classifier-Free Guidance (`logits_uncond + cfg_weight * (logits_cond -
//   logits_uncond)`), and emits `JanusProLlmOutput` holding
//   `JanusProVqCodeGrid`.
class JanusProLlmStage
    : public SingleThreadedStageWithDeque<JanusProLlmOutput> {
 public:
  // Creates a `JanusProLlmStage` with the given upstream frontend stage,
  // tokenizer, conditional/unconditional LLM runners, and optional visual
  // generation submodel runners (`vq_image_embedder_runner` and
  // `gen_head_runner`).
  static absl::StatusOr<std::unique_ptr<JanusProLlmStage>> Create(
      Stage<JanusProFrontendChunk>* absl_nonnull frontend,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtLmRunner> absl_nonnull llm_runner,
      std::shared_ptr<LiteRtLmRunner> absl_nullable uncond_llm_runner = nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable vq_image_embedder_runner =
          nullptr,
      std::unique_ptr<LiteRtRunner> absl_nullable gen_head_runner = nullptr,
      const JanusProModelConfig& config = {});

  ~JanusProLlmStage() override;

 protected:
  void ResetInternal() override;
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  JanusProLlmStage(
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
      JanusProModelConfig config);

  absl::Status RunUnderstandingTurn(const JanusProFrontendChunk& prompt);
  absl::Status RunImageGenerationTurn(const JanusProFrontendChunk& prompt);

  Stage<JanusProFrontendChunk>& frontend_;
  JanusProModelConfig config_;
  std::shared_ptr<support::Tokenizer> tokenizer_;
  std::shared_ptr<LiteRtLmRunner> llm_runner_;
  std::shared_ptr<LiteRtLmRunner> uncond_llm_runner_;

  std::unique_ptr<LiteRtRunner> vq_image_embedder_runner_;
  std::vector<TensorBuffer> vq_emb_input_buffers_;
  std::vector<TensorBuffer> vq_emb_output_buffers_;

  std::unique_ptr<LiteRtRunner> gen_head_runner_;
  std::vector<TensorBuffer> gen_head_input_buffers_;
  std::vector<TensorBuffer> gen_head_output_buffers_;

  absl::flat_hash_set<int> stop_token_ids_;
  int hidden_dim_ = janus_pro::kLlmHiddenSize;
  bool last_turn_was_image_gen_ = false;
  Any2AnySampler text_sampler_;
  Any2AnySampler image_sampler_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_RUNNER_LLM_STAGE_H_
