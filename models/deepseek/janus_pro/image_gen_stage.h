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

#ifndef MODELS_DEEPSEEK_JANUS_PRO_IMAGE_GEN_STAGE_H_
#define MODELS_DEEPSEEK_JANUS_PRO_IMAGE_GEN_STAGE_H_

#include <memory>
#include <vector>

#include "models/deepseek/janus_pro/io_types.h"
#include "models/deepseek/janus_pro/janus_pro_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Stage 3: Output & VQ-16 Image Decoder Stage for DeepSeek Janus-Pro.
// - In Multimodal Understanding mode: forwards the assistant text from
//   `JanusProLlmStage` as `TextOutput` followed by `EndOfOutputTurn`.
// - In Visual Generation mode: decodes the `576` (`24x24`) VQ-16 codebook
//   indices via `vq_image_decoder_runner` into `[1, 3, image_size, image_size]`
//   float32 NCHW pixels in `[-1, 1]`, converts to interleaved RGB888 HWC
//   `uint8_t` bytes (`[0, 255]`), and emits `ImageOutput` followed by
//   `EndOfOutputTurn`.
class JanusProImageGenStage : public SingleThreadedStageWithDeque<Output> {
 public:
  // Creates a `JanusProImageGenStage` consuming `JanusProLlmOutput` from
  // `llm_stage` and optionally decoding VQ-16 code grids via
  // `vq_image_decoder_runner`.
  static absl::StatusOr<std::unique_ptr<JanusProImageGenStage>> Create(
      Stage<JanusProLlmOutput>* absl_nonnull llm_stage,
      const JanusProModelConfig& config = {},
      std::unique_ptr<LiteRtRunner> absl_nullable vq_image_decoder_runner =
          nullptr);

  ~JanusProImageGenStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  JanusProImageGenStage(
      Stage<JanusProLlmOutput>* absl_nonnull llm_stage,
      const JanusProModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nullable vq_image_decoder_runner,
      std::vector<TensorBuffer> vq_dec_input_buffers,
      std::vector<TensorBuffer> vq_dec_output_buffers);

  absl::StatusOr<ImageOutput> DecodeVqCodeGrid(const JanusProVqCodeGrid& grid);

  Stage<JanusProLlmOutput>& llm_stage_;
  JanusProModelConfig config_;
  std::unique_ptr<LiteRtRunner> vq_image_decoder_runner_;
  std::vector<TensorBuffer> vq_dec_input_buffers_;
  std::vector<TensorBuffer> vq_dec_output_buffers_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_IMAGE_GEN_STAGE_H_
