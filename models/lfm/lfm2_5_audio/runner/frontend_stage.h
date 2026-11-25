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

#ifndef MODELS_LFM_LFM2_5_AUDIO_RUNNER_FRONTEND_STAGE_H_
#define MODELS_LFM_LFM2_5_AUDIO_RUNNER_FRONTEND_STAGE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "models/lfm/lfm2_5_audio/runner/io_types.h"
#include "models/lfm/lfm2_5_audio/runner/lfm2_audio_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Chat-template assembly stage for `LiquidAI/LFM2.5-Audio-1.5B` (`ChatState`
// format: `<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n` +
// `<|im_start|>user\n{interleaved text and AUDIO_IN embeddings}<|im_end|>\n` +
// `<|im_start|>assistant\n`).
//
// Static control-token segments (`system_prefix_seg_`, `user_start_seg_`,
// `user_end_seg_`) are precomputed synchronously during `Create()` so the
// `text_embedder` model is never invoked concurrently with `Lfm2AudioTextStage`
// at runtime.
class Lfm2AudioFrontendStage
    : public SingleThreadedStageWithDeque<Lfm2AudioFrontendOutput> {
 public:
  // Creates an `Lfm2AudioFrontendStage`.
  //
  // Args:
  //   text_stage: Non-null `Lfm2AudioTextStage`.
  //   audio_stage: Optional `Lfm2AudioStage`.
  //   tokenizer: Non-null tokenizer used during `Create()` to tokenize control
  //     templates.
  //   text_embedder_runner: Runner executing `tf_lite_text_embedder`, used
  //     only synchronously during `Create()` to precompute control-token
  //     embeddings (not retained for runtime execution).
  //   config: LFM2.5-Audio model configuration.
  //
  // Returns:
  //   Initialized `Lfm2AudioFrontendStage`, or an error status on invalid
  //   config or incompatible embedder buffer shapes.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioFrontendStage>> Create(
      SingleThreadedStageWithDeque<
          Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nonnull text_stage,
      SingleThreadedStageWithDeque<
          Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nullable audio_stage,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nullable text_embedder_runner =
          nullptr,
      const Lfm2AudioModelConfig& config = {});

  ~Lfm2AudioFrontendStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  Lfm2AudioFrontendStage(
      SingleThreadedStageWithDeque<
          Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nonnull text_stage,
      SingleThreadedStageWithDeque<
          Sequenced<Lfm2AudioEmbeddingSegment>>* absl_nullable audio_stage,
      Lfm2AudioEmbeddingSegment system_prefix_seg,
      Lfm2AudioEmbeddingSegment user_start_seg,
      Lfm2AudioEmbeddingSegment user_end_seg,
      Lfm2AudioGenerationMode resolved_mode,
      const Lfm2AudioModelConfig& config);

  absl::Status AppendSegment(const Lfm2AudioEmbeddingSegment& seg);
  absl::Status FinalizeTurn();

  SingleThreadedStageWithDeque<Sequenced<Lfm2AudioEmbeddingSegment>>&
      text_stage_;
  SingleThreadedStageWithDeque<Sequenced<Lfm2AudioEmbeddingSegment>>*
      audio_stage_;
  Lfm2AudioModelConfig config_;
  int hidden_dim_;
  Lfm2AudioEmbeddingSegment system_prefix_seg_;
  Lfm2AudioEmbeddingSegment user_start_seg_;
  Lfm2AudioEmbeddingSegment user_end_seg_;
  Lfm2AudioGenerationMode resolved_mode_;

  uint64_t next_expected_seqnum_ = 0;
  bool is_first_turn_ = true;
  std::vector<Lfm2AudioEmbeddingSegment> turn_segments_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_RUNNER_FRONTEND_STAGE_H_
