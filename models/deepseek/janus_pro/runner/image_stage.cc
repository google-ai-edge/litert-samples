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

#include "models/deepseek/janus_pro/runner/image_stage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/runner/io_types.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/utils/tensor_utils.h"
#include "runtime/components/preprocessor/image_preprocessor.h"

namespace litert::omni::any2any {
namespace {

// Converts raw RGB (3-channel), RGBA (4-channel), or grayscale (1-channel)
// `image.image_bytes` into interleaved RGB888 pixels, or decodes compressed
// image bytes using `preprocessor`.
absl::StatusOr<lm::DecodedImage> DecodeOrConvertImageToRgb(
    const ImageInput& image, const lm::ImagePreprocessor* preprocessor) {
  if (image.width < 0 || image.height < 0 ||
      (image.width > 0) != (image.height > 0)) {
    return absl::InvalidArgumentError(
        "ImageInput width and height must both be positive for raw pixels or "
        "both be zero for encoded images.");
  }

  if (image.width > 0 && image.height > 0) {
    const size_t num_pixels = static_cast<size_t>(image.width) * image.height;
    if (image.channels == 3 && image.image_bytes.size() == num_pixels * 3) {
      return lm::DecodedImage{
          .pixels = image.image_bytes,
          .width = image.width,
          .height = image.height,
      };
    }
    if (image.channels == 4 && image.image_bytes.size() == num_pixels * 4) {
      std::vector<uint8_t> rgb_pixels;
      rgb_pixels.reserve(num_pixels * 3);
      for (size_t i = 0; i < image.image_bytes.size(); i += 4) {
        rgb_pixels.push_back(image.image_bytes[i]);
        rgb_pixels.push_back(image.image_bytes[i + 1]);
        rgb_pixels.push_back(image.image_bytes[i + 2]);
      }
      return lm::DecodedImage{
          .pixels = std::move(rgb_pixels),
          .width = image.width,
          .height = image.height,
      };
    }
    if (image.channels == 1 && image.image_bytes.size() == num_pixels) {
      std::vector<uint8_t> rgb_pixels;
      rgb_pixels.reserve(num_pixels * 3);
      for (uint8_t gray : image.image_bytes) {
        rgb_pixels.push_back(gray);
        rgb_pixels.push_back(gray);
        rgb_pixels.push_back(gray);
      }
      return lm::DecodedImage{
          .pixels = std::move(rgb_pixels),
          .width = image.width,
          .height = image.height,
      };
    }
    return absl::InvalidArgumentError(
        "ImageInput dimensions/channels do not match image_bytes size.");
  }

  if (preprocessor == nullptr) {
    return absl::FailedPreconditionError(
        "JanusProImageStage requires an ImagePreprocessor to decode "
        "compressed image bytes.");
  }
  absl::string_view encoded_view(
      reinterpret_cast<const char*>(image.image_bytes.data()),
      image.image_bytes.size());
  return preprocessor->Decode(encoded_view);
}

// Resizes `decoded` preserving aspect ratio, pads to `image_size x image_size`
// with `(127, 127, 127)` (`VLMImageProcessor.resize` + `expand2square`), and
// normalizes into NCHW `[1, 3, image_size, image_size]` with `mean = 0.5` and
// `std = 0.5`.
std::vector<float> PreprocessRgbToNchw(const lm::DecodedImage& decoded,
                                       int image_size) {
  const int src_w = decoded.width;
  const int src_h = decoded.height;
  const int max_dim = std::max(src_w, src_h);
  const int min_size = std::min(janus_pro::kMinImageSize, image_size);

  const int dst_h = std::clamp(
      static_cast<int>(static_cast<double>(src_h) / max_dim * image_size),
      min_size, image_size);
  const int dst_w = std::clamp(
      static_cast<int>(static_cast<double>(src_w) / max_dim * image_size),
      min_size, image_size);

  // Bilinear resize from `(src_w, src_h)` to `(dst_w, dst_h)`.
  std::vector<uint8_t> resized(static_cast<size_t>(dst_h) * dst_w * 3);
  for (int y = 0; y < dst_h; ++y) {
    const float src_y = dst_h > 1
                            ? (static_cast<float>(y) + 0.5f) *
                                      (static_cast<float>(src_h) / dst_h) -
                                  0.5f
                            : 0.0f;
    const float clamped_y =
        std::clamp(src_y, 0.0f, static_cast<float>(src_h - 1));
    const int y0 = static_cast<int>(std::floor(clamped_y));
    const int y1 = std::min(y0 + 1, src_h - 1);
    const float wy = clamped_y - y0;

    for (int x = 0; x < dst_w; ++x) {
      const float src_x = dst_w > 1
                              ? (static_cast<float>(x) + 0.5f) *
                                        (static_cast<float>(src_w) / dst_w) -
                                    0.5f
                              : 0.0f;
      const float clamped_x =
          std::clamp(src_x, 0.0f, static_cast<float>(src_w - 1));
      const int x0 = static_cast<int>(std::floor(clamped_x));
      const int x1 = std::min(x0 + 1, src_w - 1);
      const float wx = clamped_x - x0;

      const size_t row0 = static_cast<size_t>(y0) * src_w;
      const size_t row1 = static_cast<size_t>(y1) * src_w;
      const size_t dst_pixel = (static_cast<size_t>(y) * dst_w + x) * 3;
      for (int c = 0; c < 3; ++c) {
        const float p00 = decoded.pixels[(row0 + x0) * 3 + c];
        const float p01 = decoded.pixels[(row0 + x1) * 3 + c];
        const float p10 = decoded.pixels[(row1 + x0) * 3 + c];
        const float p11 = decoded.pixels[(row1 + x1) * 3 + c];
        const float top = p00 * (1.0f - wx) + p01 * wx;
        const float bot = p10 * (1.0f - wx) + p11 * wx;
        const float val = top * (1.0f - wy) + bot * wy;
        resized[dst_pixel + c] =
            static_cast<uint8_t>(std::clamp(std::round(val), 0.0f, 255.0f));
      }
    }
  }

  // `expand2square` onto an `image_size x image_size` canvas filled with 127.
  std::vector<uint8_t> square(static_cast<size_t>(image_size) * image_size * 3,
                              janus_pro::kPadBackgroundColor);
  const int y_offset = (image_size - dst_h) / 2;
  const int x_offset = (image_size - dst_w) / 2;
  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      const size_t src_idx = (static_cast<size_t>(y) * dst_w + x) * 3;
      const size_t dst_idx =
          (static_cast<size_t>(y + y_offset) * image_size + (x + x_offset)) * 3;
      square[dst_idx] = resized[src_idx];
      square[dst_idx + 1] = resized[src_idx + 1];
      square[dst_idx + 2] = resized[src_idx + 2];
    }
  }

  // Convert HWC [0, 255] to NCHW [-1, 1] (`(x / 255.0 - 0.5) / 0.5`).
  const size_t plane_size = static_cast<size_t>(image_size) * image_size;
  std::vector<float> nchw(3 * plane_size);
  for (int y = 0; y < image_size; ++y) {
    for (int x = 0; x < image_size; ++x) {
      const size_t pixel_offset = static_cast<size_t>(y) * image_size + x;
      const size_t hwc_idx = pixel_offset * 3;
      for (int c = 0; c < 3; ++c) {
        const float normalized =
            (static_cast<float>(square[hwc_idx + c]) / 255.0f - 0.5f) / 0.5f;
        nchw[c * plane_size + pixel_offset] = normalized;
      }
    }
  }
  return nchw;
}

}  // namespace

JanusProImageStage::JanusProImageStage(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<lm::ImagePreprocessor> absl_nullable image_preprocessor,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    std::vector<TensorBuffer> vision_input_buffers,
    std::vector<TensorBuffer> vision_output_buffers,
    const JanusProModelConfig& config)
    : image_queue_(*image_queue),
      config_(config),
      hidden_dim_(config.llm_hidden_size),
      image_preprocessor_(std::move(image_preprocessor)),
      vision_encoder_runner_(std::move(vision_encoder_runner)),
      vision_input_buffers_(std::move(vision_input_buffers)),
      vision_output_buffers_(std::move(vision_output_buffers)) {}

JanusProImageStage::~JanusProImageStage() = default;

absl::StatusOr<std::unique_ptr<JanusProImageStage>> JanusProImageStage::Create(
    FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull vision_encoder_runner,
    const JanusProModelConfig& config) {
  if (config.llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(
        "JanusProImageStage config.llm_hidden_size must be positive.");
  }
  if (config.image_size <= 0 || config.num_image_tokens <= 0) {
    return absl::InvalidArgumentError(
        "JanusProImageStage config.image_size and config.num_image_tokens "
        "must be positive.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::vector<TensorBuffer> vision_input_buffers,
      vision_encoder_runner->CreateInputBuffers(/*signature_name=*/""));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<TensorBuffer> vision_output_buffers,
      vision_encoder_runner->CreateOutputBuffers(/*signature_name=*/""));
  if (vision_input_buffers.empty() || vision_output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "JanusProImageStage vision_encoder_runner must provide non-empty "
        "input and output buffers.");
  }

  ABSL_RETURN_IF_ERROR(ValidateTensorBuffer(
      vision_input_buffers[0], ElementType::Float32,
      static_cast<size_t>(3) * config.image_size * config.image_size,
      "JanusProImageStage vision input"));
  ABSL_RETURN_IF_ERROR(ValidateTensorBuffer(
      vision_output_buffers[0], ElementType::Float32,
      static_cast<size_t>(config.num_image_tokens) * config.llm_hidden_size,
      "JanusProImageStage vision output"));

  return absl::WrapUnique(new JanusProImageStage(
      image_queue, lm::ImagePreprocessor::Create(),
      std::move(vision_encoder_runner), std::move(vision_input_buffers),
      std::move(vision_output_buffers), config));
}

absl::StatusOr<JanusProEmbeddingSegment> JanusProImageStage::ProcessImage(
    const ImageInput& image) {
  if (image.image_bytes.empty()) {
    return JanusProEmbeddingSegment{};
  }

  ABSL_ASSIGN_OR_RETURN(
      lm::DecodedImage decoded,
      DecodeOrConvertImageToRgb(image, image_preprocessor_.get()));
  if (decoded.width <= 0 || decoded.height <= 0 ||
      decoded.pixels.size() !=
          static_cast<size_t>(decoded.width) * decoded.height * 3) {
    return absl::InvalidArgumentError(
        "Decoded image dimensions do not match RGB pixel buffer size.");
  }

  std::vector<float> pixel_values =
      PreprocessRgbToNchw(decoded, config_.image_size);
  LITERT_RETURN_IF_ERROR(
      vision_input_buffers_[0].Write<float>(absl::MakeConstSpan(pixel_values)));
  ABSL_RETURN_IF_ERROR(vision_encoder_runner_->Run(
      /*signature_name=*/"", vision_input_buffers_, vision_output_buffers_));

  std::vector<float> vision_emb(static_cast<size_t>(config_.num_image_tokens) *
                                hidden_dim_);
  LITERT_RETURN_IF_ERROR(
      vision_output_buffers_[0].Read<float>(absl::MakeSpan(vision_emb)));

  JanusProEmbeddingSegment segment;
  segment.token_ids.reserve(static_cast<size_t>(config_.num_image_tokens) + 2);
  segment.token_ids.push_back(janus_pro::kBeginOfImageTokenId);
  segment.token_ids.insert(segment.token_ids.end(), config_.num_image_tokens,
                           janus_pro::kImageEmbedPlaceholderTokenId);
  segment.token_ids.push_back(janus_pro::kEndOfImageTokenId);
  segment.embeddings = std::move(vision_emb);
  segment.num_tokens = config_.num_image_tokens;
  return segment;
}

bool JanusProImageStage::NeedScheduleInternal() const {
  return !image_queue_.Empty();
}

absl::Status JanusProImageStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (image_queue_.Empty()) {
    return absl::NotFoundError("JanusProImageStage queue has no output.");
  }

  while (!image_queue_.Empty()) {
    ABSL_ASSIGN_OR_RETURN(const Sequenced<ImageInput>* sequenced_input,
                          image_queue_.PeekFront());
    ABSL_ASSIGN_OR_RETURN(JanusProEmbeddingSegment image_emb,
                          ProcessImage(sequenced_input->item));
    const uint64_t seqnum = sequenced_input->seqnum;
    ABSL_RETURN_IF_ERROR(image_queue_.PopFront().status());
    this->PushOutput(Sequenced<JanusProEmbeddingSegment>{
        .seqnum = seqnum,
        .item = std::move(image_emb),
    });
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
