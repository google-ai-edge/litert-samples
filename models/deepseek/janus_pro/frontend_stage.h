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

#ifndef MODELS_DEEPSEEK_JANUS_PRO_FRONTEND_STAGE_H_
#define MODELS_DEEPSEEK_JANUS_PRO_FRONTEND_STAGE_H_

#include <memory>
#include <vector>

#include "models/deepseek/janus_pro/io_types.h"
#include "models/deepseek/janus_pro/janus_pro_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "omni/base/fan_in_stage.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Frontend stage for DeepSeek Janus-Pro:
// - Reorders `Sequenced<JanusProEmbeddingSegment>` items from parallel modality
//   stages (`JanusProTextStage` and optional `JanusProImageStage`) in
//   contiguous `seqnum` order via an internal `FanInStage`.
// - Precomputes all static DeepSeek SFT chat-template control embeddings and
//   unconditional CFG token embeddings synchronously in `Create()` so there are
//   zero runtime calls to the shared `text_embedder_runner`.
// - Formats DeepSeek SFT chat-template turn headers and suffixes:
//   - Understanding mode (`generate_image == false`):
//     `<｜begin▁of▁sentence｜>{system_prompt}\n\n<|User|>: ...`
//     `\n\n<|Assistant|>:`
//   - Visual Generation mode (`generate_image == true`):
//     Conditional prompt:
//       `<｜begin▁of▁sentence｜><|User|>: {prompt}`
//       `\n\n<|Assistant|>:<begin_of_image>`
//     Unconditional CFG prompt:
//       `[<｜begin▁of▁sentence｜>, <｜▁pad▁｜> * (N - 2), <begin_of_image>]`
// - Frames image segments with `<begin_of_image>...<end_of_image>\n` boundary
//   token embeddings.
// - Emits `JanusProFrontendChunk` at the end of each user turn.
class JanusProFrontendStage
    : public SingleThreadedStageWithDeque<JanusProFrontendChunk> {
 public:
  // Creates a `JanusProFrontendStage` that precomputes all static control-token
  // embeddings using `tokenizer` and `text_embedder_runner` (without retaining
  // `text_embedder_runner`), then reorders sequenced segments from `text_stage`
  // and optional `image_stage`.
  static absl::StatusOr<std::unique_ptr<JanusProFrontendStage>> Create(
      SingleThreadedStageWithDeque<
          Sequenced<JanusProEmbeddingSegment>>* absl_nonnull text_stage,
      SingleThreadedStageWithDeque<
          Sequenced<JanusProEmbeddingSegment>>* absl_nullable image_stage,
      std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::shared_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
      const JanusProModelConfig& config = {});

  ~JanusProFrontendStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  absl::Status FlushInternal() override;
  void ResetInternal() override;

 private:
  struct PrecomputedTemplates {
    JanusProEmbeddingSegment first_turn_prefix;
    JanusProEmbeddingSegment next_turn_prefix;
    JanusProEmbeddingSegment turn_end_suffix;
    int boi_token_id = 0;
    int eoi_token_id = 0;
    std::vector<float> boi_embedding;
    std::vector<float> eoi_embedding;
    std::vector<int> image_newline_token_ids;
    std::vector<float> image_newline_embeddings;
    int bos_token_id = 0;
    int pad_token_id = 0;
    std::vector<float> bos_embedding;
    std::vector<float> pad_embedding;
  };

  JanusProFrontendStage(
      SingleThreadedStageWithDeque<
          Sequenced<JanusProEmbeddingSegment>>* absl_nonnull text_stage,
      SingleThreadedStageWithDeque<
          Sequenced<JanusProEmbeddingSegment>>* absl_nullable image_stage,
      PrecomputedTemplates templates, const JanusProModelConfig& config);

  absl::Status ProcessSegment(const JanusProEmbeddingSegment& segment);
  absl::StatusOr<JanusProEmbeddingSegment> FrameImageSegment(
      const JanusProEmbeddingSegment& image) const;
  void AppendRawSegment(const JanusProEmbeddingSegment& segment);
  absl::Status FlushTurn();
  void ResetTurn();

  FanInStage<JanusProEmbeddingSegment> fan_in_;
  PrecomputedTemplates templates_;
  int hidden_dim_;
  std::vector<float> prompt_embeddings_;
  std::vector<int> token_ids_;
  bool generate_image_ = false;
  bool is_first_turn_ = true;
  bool in_turn_ = false;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_FRONTEND_STAGE_H_
