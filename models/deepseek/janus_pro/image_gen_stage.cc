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

#include "models/deepseek/janus_pro/image_gen_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/deepseek/janus_pro/io_types.h"
#include "models/deepseek/janus_pro/janus_pro_model_config.h"
#include "models/deepseek/janus_pro/stage_utils.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

JanusProImageGenStage::JanusProImageGenStage(
    Stage<JanusProLlmOutput>* absl_nonnull llm_stage,
    const JanusProModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nullable vq_image_decoder_runner,
    std::vector<TensorBuffer> vq_dec_input_buffers,
    std::vector<TensorBuffer> vq_dec_output_buffers)
    : llm_stage_(*llm_stage),
      config_(config),
      vq_image_decoder_runner_(std::move(vq_image_decoder_runner)),
      vq_dec_input_buffers_(std::move(vq_dec_input_buffers)),
      vq_dec_output_buffers_(std::move(vq_dec_output_buffers)) {}

JanusProImageGenStage::~JanusProImageGenStage() = default;

absl::StatusOr<std::unique_ptr<JanusProImageGenStage>>
JanusProImageGenStage::Create(
    Stage<JanusProLlmOutput>* absl_nonnull llm_stage,
    const JanusProModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nullable vq_image_decoder_runner) {
  std::vector<TensorBuffer> vq_dec_in;
  std::vector<TensorBuffer> vq_dec_out;
  if (vq_image_decoder_runner != nullptr) {
    if (config.num_image_tokens <= 0 || config.image_size <= 0) {
      return absl::InvalidArgumentError(
          "JanusProImageGenStage config.num_image_tokens and "
          "config.image_size must be positive when vq_image_decoder_runner is "
          "provided.");
    }
    ABSL_ASSIGN_OR_RETURN(
        vq_dec_in,
        vq_image_decoder_runner->CreateInputBuffers(/*signature_name=*/""));
    ABSL_ASSIGN_OR_RETURN(
        vq_dec_out,
        vq_image_decoder_runner->CreateOutputBuffers(/*signature_name=*/""));
    if (vq_dec_in.empty() || vq_dec_out.empty()) {
      return absl::InvalidArgumentError(
          "JanusProImageGenStage vq_image_decoder_runner must provide "
          "non-empty input and output buffers.");
    }
    ABSL_RETURN_IF_ERROR(janus_pro::ValidateTensorBuffer(
        vq_dec_in[0], ElementType::Int32,
        static_cast<size_t>(config.num_image_tokens),
        "JanusProImageGenStage vq_image_decoder input"));
    ABSL_RETURN_IF_ERROR(janus_pro::ValidateTensorBuffer(
        vq_dec_out[0], ElementType::Float32,
        static_cast<size_t>(3) * config.image_size * config.image_size,
        "JanusProImageGenStage vq_image_decoder output"));
  }
  return absl::WrapUnique(new JanusProImageGenStage(
      llm_stage, config, std::move(vq_image_decoder_runner),
      std::move(vq_dec_in), std::move(vq_dec_out)));
}

absl::StatusOr<ImageOutput> JanusProImageGenStage::DecodeVqCodeGrid(
    const JanusProVqCodeGrid& grid) {
  if (vq_image_decoder_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "JanusProImageGenStage requires vq_image_decoder_runner when decoding "
        "a VQ-16 code grid.");
  }
  if (grid.vq_codes.size() != static_cast<size_t>(config_.num_image_tokens)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "JanusProVqCodeGrid size (", grid.vq_codes.size(),
        ") does not match num_image_tokens (", config_.num_image_tokens, ")."));
  }
  for (int32_t code : grid.vq_codes) {
    if (code < 0 || code >= config_.vq_codebook_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "JanusProVqCodeGrid contains out-of-range VQ code (", code,
          "); expected [0, ", config_.vq_codebook_size, ")."));
    }
  }

  const int width =
      grid.image_width > 0 ? grid.image_width : config_.image_size;
  const int height =
      grid.image_height > 0 ? grid.image_height : config_.image_size;
  if (width != config_.image_size || height != config_.image_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("JanusProVqCodeGrid dimensions (", width, "x", height,
                     ") do not match configured image_size (",
                     config_.image_size, "x", config_.image_size, ")."));
  }

  LITERT_RETURN_IF_ERROR(vq_dec_input_buffers_[0].Write<int32_t>(
      absl::MakeConstSpan(grid.vq_codes)));
  ABSL_RETURN_IF_ERROR(vq_image_decoder_runner_->Run(
      /*signature_name=*/"", vq_dec_input_buffers_, vq_dec_output_buffers_));

  const size_t plane_size = static_cast<size_t>(width) * height;
  std::vector<float> decoded_nchw(3 * plane_size);
  LITERT_RETURN_IF_ERROR(
      vq_dec_output_buffers_[0].Read<float>(absl::MakeSpan(decoded_nchw)));

  std::vector<uint8_t> rgb_bytes(3 * plane_size);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t pixel_idx = static_cast<size_t>(y) * width + x;
      const size_t hwc_idx = pixel_idx * 3;
      for (int c = 0; c < 3; ++c) {
        const float val =
            (decoded_nchw[c * plane_size + pixel_idx] + 1.0f) * 0.5f * 255.0f;
        rgb_bytes[hwc_idx + c] =
            static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
      }
    }
  }

  return ImageOutput{
      .rgb_data = std::move(rgb_bytes),
      .width = width,
      .height = height,
      .channels = 3,
  };
}

bool JanusProImageGenStage::NeedScheduleInternal() const {
  return llm_stage_.HasOutput();
}

absl::Status JanusProImageGenStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  while (llm_stage_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(JanusProLlmOutput out, llm_stage_.GetOutput());
    if (const auto* text = std::get_if<std::string>(&out.payload)) {
      PushOutput(TextOutput{.confirmed_text = *text});
      PushOutput(EndOfOutputTurn{});
    } else if (const auto* grid =
                   std::get_if<JanusProVqCodeGrid>(&out.payload)) {
      ABSL_ASSIGN_OR_RETURN(ImageOutput img_out, DecodeVqCodeGrid(*grid));
      PushOutput(std::move(img_out));
      PushOutput(EndOfOutputTurn{});
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
