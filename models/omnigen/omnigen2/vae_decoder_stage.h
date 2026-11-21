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

#ifndef MODELS_OMNIGEN_OMNIGEN2_VAE_DECODER_STAGE_H_
#define MODELS_OMNIGEN_OMNIGEN2_VAE_DECODER_STAGE_H_

#include <memory>
#include <vector>

#include "models/omnigen/omnigen2/io_types.h"
#include "models/omnigen/omnigen2/omnigen2_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/image_decoder.h"

namespace litert::omni::text2image::omnigen2 {

// Stage 3 for `OmniGen2/OmniGen2`:
// Unpacks 2x2-patchified latents `[1, grid_h * grid_w, 64]` into NCHW
// `[1, 16, H/8, W/8]`, applies inverse VAE scaling and shift
// (`z = z_unpacked / vae_scaling_factor + vae_shift_factor`), runs the
// 16-channel `AutoencoderKL` decoder, and converts planar NCHW float output in
// `[-1, 1]` to interleaved `RGB888`.
class OmniGen2VaeDecoderStage : public ImageDecoder {
 public:
  static absl::StatusOr<std::unique_ptr<OmniGen2VaeDecoderStage>> Create(
      Stage<OmniGen2DenoiserOutput>* absl_nonnull denoiser,
      const OmniGen2ModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner);

  ~OmniGen2VaeDecoderStage() override = default;

  absl::Status Flush() override { return absl::OkStatus(); }

 protected:
  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  OmniGen2VaeDecoderStage(Stage<OmniGen2DenoiserOutput>* absl_nonnull denoiser,
                          OmniGen2ModelConfig config,
                          std::unique_ptr<LiteRtRunner> absl_nonnull runner,
                          std::vector<TensorBuffer> input_buffers,
                          std::vector<TensorBuffer> output_buffers);

  Stage<OmniGen2DenoiserOutput>& denoiser_;
  const OmniGen2ModelConfig config_;
  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
};

}  // namespace litert::omni::text2image::omnigen2

#endif  // MODELS_OMNIGEN_OMNIGEN2_VAE_DECODER_STAGE_H_
