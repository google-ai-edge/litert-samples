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

#ifndef MODELS_TONGYI_Z_IMAGE_TURBO_RUNNER_DENOISER_STAGE_H_
#define MODELS_TONGYI_Z_IMAGE_TURBO_RUNNER_DENOISER_STAGE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "models/tongyi/z_image_turbo/runner/z_image_turbo_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/io_types.h"
#include "omni/text2image/text_encoder_stage.h"

namespace litert::omni::text2image {

// Sharded S3-DiT FlowMatch Euler denoiser stage for Z-Image-Turbo.
//
// Runs `cap_embedder` and `context_refiner` once per prompt before the Euler
// loop, then executes `x_embedder`, `noise_refiner`, `main_blocks`, and
// `final_layer` at each denoising step, updating the patchified latents via
// `latents += (sigma[i] - sigma[i + 1]) * model_out`. Emits denoised
// patchified latents `[1, img_tokens, packed_ch]` as `DenoisedLatents`.
class ZImageDenoiserStage
    : public SingleThreadedStageWithDeque<DenoisedLatents> {
 public:
  // Collection of sharded LiteRT runners for the Z-Image-Turbo S3-DiT denoiser.
  struct Runners {
    // Caption projection (`z_embc`): `[1, seq_len, prompt_dim] -> [1, seq_len,
    // dim]`.
    std::unique_ptr<LiteRtRunner> cap_embedder;
    // Unmodulated caption context refiner (`z_refc`):
    // `(cap, cap_cos, cap_sin) -> cap_refined [1, seq_len, dim]`.
    std::unique_ptr<LiteRtRunner> context_refiner;
    // Image patch + timestep embedder (`z_embx`):
    // `(x, t) -> (x_emb [1, img_tokens, dim], adaln [1, adaln_dim])`.
    std::unique_ptr<LiteRtRunner> x_embedder;
    // Modulated image noise refiner (`z_refx`):
    // `(x_emb, img_cos, img_sin, adaln) -> x_refined [1, img_tokens, dim]`.
    std::unique_ptr<LiteRtRunner> noise_refiner;
    // Sharded unified S3-DiT main transformer blocks (`zc_main0`..`zc_main5`):
    // `(unified, unified_cos, unified_sin, adaln) -> unified [1, total_tokens,
    // dim]`.
    std::vector<std::unique_ptr<LiteRtRunner>> main_blocks;
    // Final AdaLN + linear projection (`zc_final`):
    // `(unified, adaln) -> model_out [1, img_tokens, packed_ch]`.
    std::unique_ptr<LiteRtRunner> final_layer;
  };

  // Creates a `ZImageDenoiserStage`, validating all runners and precomputing
  // the initial 3D axial RoPE tables for caption, image, and unified sequences.
  // Returns `InvalidArgumentError` if any runner or configuration is invalid.
  static absl::StatusOr<std::unique_ptr<ZImageDenoiserStage>> Create(
      Stage<TextEncoderOutput>* absl_nonnull text_encoder,
      const ZImageTurboModelConfig& config, Runners runners);

  ~ZImageDenoiserStage() override;

 protected:
  // Returns true when `text_encoder` has output ready to consume.
  bool NeedScheduleInternal() const override;

  // Pulls encoded text features and runs the S3-DiT Euler denoising loop.
  absl::Status ScheduleInternal() override;

 private:
  struct Buffers {
    std::vector<TensorBuffer> embc_in;
    std::vector<TensorBuffer> embc_out;
    std::vector<TensorBuffer> refc_in;
    std::vector<TensorBuffer> refc_out;
    std::vector<TensorBuffer> embx_in;
    std::vector<TensorBuffer> embx_out;
    std::vector<TensorBuffer> refx_in;
    std::vector<TensorBuffer> refx_out;
    std::vector<std::vector<TensorBuffer>> main_in;
    std::vector<std::vector<TensorBuffer>> main_out;
    std::vector<TensorBuffer> final_in;
    std::vector<TensorBuffer> final_out;
  };

  struct StepScratch {
    std::vector<float> x_emb;
    std::vector<float> adaln;
    std::vector<float> x_refined;
    std::vector<float> unified;
    std::vector<float> model_out;
  };

  ZImageDenoiserStage(Stage<TextEncoderOutput>* absl_nonnull text_encoder,
                      ZImageTurboModelConfig config, Runners runners,
                      Buffers buffers, int last_active_cap_len);

  absl::Status UpdateRotaryBuffers(int active_cap_len);
  absl::StatusOr<std::vector<float>> EncodeAndRefineCaption(
      absl::Span<const float> prompt_embeds);
  absl::StatusOr<std::vector<float>> RunEulerDenoisingLoop(
      absl::Span<const float> cap_refined, int steps, uint64_t seed);
  absl::Status RunDenoisingStep(float sigma, float next_sigma,
                                absl::Span<const float> cap_refined,
                                absl::Span<float> latents,
                                StepScratch& scratch);

  Stage<TextEncoderOutput>& text_encoder_;
  const ZImageTurboModelConfig config_;
  Runners runners_;
  Buffers buffers_;
  int last_active_cap_len_ = 0;
};

}  // namespace litert::omni::text2image

#endif  // MODELS_TONGYI_Z_IMAGE_TURBO_RUNNER_DENOISER_STAGE_H_
