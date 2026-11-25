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

#ifndef MODELS_DEEPSEEK_JANUS_PRO_RUNNER_IMAGE_STAGE_H_
#define MODELS_DEEPSEEK_JANUS_PRO_RUNNER_IMAGE_STAGE_H_

#include <memory>
#include <vector>

#include "models/deepseek/janus_pro/runner/io_types.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/preprocessor/image_preprocessor.h"

namespace litert::omni::any2any {

// Image input processing stage for DeepSeek Janus-Pro:
// - Consumes `Sequenced<ImageInput>` from `JanusProInputFanOutStage`.
// - Decodes compressed PNG/JPEG bytes or converts raw RGB/RGBA/grayscale pixels
//   into 8-bit RGB HWC layout.
// - Resizes the longest edge to `config.image_size` (`384`), pads to a square
//   with `(127, 127, 127)` (`expand2square` in `VLMImageProcessor`), and
//   normalizes into NCHW `[1, 3, image_size, image_size]` with `mean = 0.5` and
//   `std = 0.5`.
// - Runs the SigLIP + MlpProjector vision encoder (`vision_encoder_runner`) to
//   produce `[1, num_image_tokens, llm_hidden_size]` continuous embeddings.
// - Emits `Sequenced<JanusProEmbeddingSegment>` marked with
//   `kBeginOfImageTokenId`, `kImageEmbedPlaceholderTokenId` (`-1`), and
//   `kEndOfImageTokenId` for `JanusProFrontendStage`.
class JanusProImageStage
    : public SingleThreadedStageWithDeque<Sequenced<JanusProEmbeddingSegment>> {
 public:
  // Creates a `JanusProImageStage` that consumes sequenced image inputs,
  // preprocesses and encodes each image with `vision_encoder_runner`, and
  // outputs `Sequenced<JanusProEmbeddingSegment>`.
  static absl::StatusOr<std::unique_ptr<JanusProImageStage>> Create(
      FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
          image_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
      const JanusProModelConfig& config = {});

  ~JanusProImageStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;

 private:
  JanusProImageStage(
      FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
          image_queue,
      std::unique_ptr<lm::ImagePreprocessor> absl_nullable image_preprocessor,
      std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
      std::vector<TensorBuffer> vision_input_buffers,
      std::vector<TensorBuffer> vision_output_buffers,
      const JanusProModelConfig& config);

  // Decodes, resizes, pads, normalizes, and encodes `image` using
  // `vision_encoder_runner_`.
  absl::StatusOr<JanusProEmbeddingSegment> ProcessImage(
      const ImageInput& image);

  FanOutOutputQueue<OmniSession::Input, ImageInput>& image_queue_;
  JanusProModelConfig config_;
  int hidden_dim_;

  std::unique_ptr<lm::ImagePreprocessor> image_preprocessor_;
  std::unique_ptr<LiteRtRunner> vision_encoder_runner_;
  std::vector<TensorBuffer> vision_input_buffers_;
  std::vector<TensorBuffer> vision_output_buffers_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_RUNNER_IMAGE_STAGE_H_
