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

#include "models/tongyi/z_image_turbo/vae_decoder_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "models/tongyi/z_image_turbo/io_types.h"
#include "models/tongyi/z_image_turbo/z_image_turbo_model_config.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_math.h"

namespace litert::omni::text2image {

absl::StatusOr<std::unique_ptr<ZImageVaeDecoderStage>>
ZImageVaeDecoderStage::Create(
    Stage<ZImageDenoiserOutput>* absl_nonnull denoiser,
    const ZImageTurboModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner) {
  if (denoiser == nullptr) {
    return absl::InvalidArgumentError(
        "ZImageVaeDecoderStage denoiser must not be null.");
  }
  if (runner == nullptr) {
    return absl::InvalidArgumentError(
        "ZImageVaeDecoderStage runner must not be null.");
  }
  if (config.vae_scaling_factor == 0.0f || config.img_size <= 0 ||
      config.in_channels <= 0 || config.patch_size <= 0 ||
      config.GridSize() <= 0 ||
      config.img_size % z_image_turbo::kVaeSpatialDownsampleFactor != 0 ||
      config.LatentSize() % config.patch_size != 0) {
    return absl::InvalidArgumentError(
        "ZImageVaeDecoderStage received invalid ZImageTurboModelConfig.");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> input_buffers,
                        runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> output_buffers,
                        runner->CreateOutputBuffers(""));
  if (input_buffers.size() != 1 || output_buffers.size() != 1) {
    return absl::InvalidArgumentError(
        "ZImageVaeDecoderStage expects exactly 1 input buffer and 1 output "
        "buffer.");
  }

  return absl::WrapUnique(new ZImageVaeDecoderStage(
      denoiser, config, std::move(runner), std::move(input_buffers),
      std::move(output_buffers)));
}

ZImageVaeDecoderStage::ZImageVaeDecoderStage(
    Stage<ZImageDenoiserOutput>* absl_nonnull denoiser,
    ZImageTurboModelConfig config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    std::vector<TensorBuffer> input_buffers,
    std::vector<TensorBuffer> output_buffers)
    : denoiser_(*denoiser),
      config_(std::move(config)),
      runner_(std::move(runner)),
      input_buffers_(std::move(input_buffers)),
      output_buffers_(std::move(output_buffers)) {}

ZImageVaeDecoderStage::~ZImageVaeDecoderStage() = default;

absl::Status ZImageVaeDecoderStage::Flush() { return absl::OkStatus(); }

bool ZImageVaeDecoderStage::NeedScheduleInternal() const {
  return denoiser_.HasOutput();
}

absl::Status ZImageVaeDecoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  absl::StatusOr<ZImageDenoiserOutput> denoised = denoiser_.GetOutput();
  if (absl::IsNotFound(denoised.status())) {
    return absl::OkStatus();
  }
  if (!denoised.ok()) {
    return denoised.status();
  }

  const int latent_size = config_.LatentSize();
  const size_t total_latent_elems = static_cast<size_t>(config_.in_channels) *
                                    static_cast<size_t>(latent_size) *
                                    static_cast<size_t>(latent_size);

  std::vector<float> nchw_latents(total_latent_elems);
  ABSL_RETURN_IF_ERROR(UnpackZImageLatents(
      absl::MakeConstSpan(denoised->packed_latents),
      absl::MakeSpan(nchw_latents), latent_size, latent_size,
      config_.patch_size, config_.in_channels));

  const float inv_scale = 1.0f / config_.vae_scaling_factor;
  const float shift = config_.vae_shift_factor;
  for (size_t i = 0; i < total_latent_elems; ++i) {
    nchw_latents[i] = nchw_latents[i] * inv_scale + shift;
  }

  LITERT_RETURN_IF_ERROR(
      input_buffers_[0].Write<float>(absl::MakeConstSpan(nchw_latents)));
  ABSL_RETURN_IF_ERROR(runner_->Run("", input_buffers_, output_buffers_));

  const int width = config_.img_size;
  const int height = config_.img_size;
  const size_t pixels =
      static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<float> nchw_rgb(3 * pixels, 0.0f);
  LITERT_RETURN_IF_ERROR(
      output_buffers_[0].Read<float>(absl::MakeSpan(nchw_rgb)));

  ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> rgb888,
                        ConvertNchwToRgb888(nchw_rgb, height, width));

  ImageOutput img_out;
  img_out.width = width;
  img_out.height = height;
  img_out.channels = 3;
  img_out.rgb_data = std::move(rgb888);

  PushOutput(std::move(img_out));
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image
