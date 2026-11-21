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

#ifndef MODELS_TONGYI_Z_IMAGE_TURBO_VAE_DECODER_STAGE_H_
#define MODELS_TONGYI_Z_IMAGE_TURBO_VAE_DECODER_STAGE_H_

#include <memory>
#include <vector>

#include "models/tongyi/z_image_turbo/io_types.h"
#include "models/tongyi/z_image_turbo/z_image_turbo_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/image_decoder.h"

namespace litert::omni::text2image {

// VAE decoder stage for Z-Image-Turbo (`AutoencoderKL`).
//
// Unpatchifies the denoised S3-DiT tokens `[1, (H/2)*(W/2), packed_ch]` into
// NCHW `[1, in_channels, H, W]`, applies inverse latent scaling and shift
// (`z = z / vae_scaling_factor + vae_shift_factor`), runs the VAE decoder
// subgraph, and converts the NCHW `[-1, 1]` float output into RGB888 bytes.
class ZImageVaeDecoderStage : public ImageDecoder {
 public:
  // Creates a `ZImageVaeDecoderStage`. Returns `InvalidArgumentError` if
  // `denoiser`, `runner`, or `config` is invalid.
  static absl::StatusOr<std::unique_ptr<ZImageVaeDecoderStage>> Create(
      Stage<ZImageDenoiserOutput>* absl_nonnull denoiser,
      const ZImageTurboModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner);

  ~ZImageVaeDecoderStage() override;

  // Flushes any buffered output.
  absl::Status Flush() override;

 protected:
  // Returns true when `denoiser` has output ready to consume.
  bool NeedScheduleInternal() const override;

  // Pulls denoised latents, unpatchifies + rescales, runs VAE, and emits RGB.
  absl::Status ScheduleInternal() override;

 private:
  ZImageVaeDecoderStage(Stage<ZImageDenoiserOutput>* absl_nonnull denoiser,
                        ZImageTurboModelConfig config,
                        std::unique_ptr<LiteRtRunner> absl_nonnull runner,
                        std::vector<TensorBuffer> input_buffers,
                        std::vector<TensorBuffer> output_buffers);

  Stage<ZImageDenoiserOutput>& denoiser_;
  const ZImageTurboModelConfig config_;
  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
};

}  // namespace litert::omni::text2image

#endif  // MODELS_TONGYI_Z_IMAGE_TURBO_VAE_DECODER_STAGE_H_
