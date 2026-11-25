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

#include "models/deepseek/janus_pro/runner/fan_out_stage.h"

#include <memory>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {

JanusProInputFanOutStage::JanusProInputFanOutStage(
    Stage<OmniSession::Input>* absl_nonnull upstream, bool has_vision_encoder)
    : FanOutStage<OmniSession::Input>(upstream),
      text_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, TextInput>>())),
      image_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, ImageInput>>())),
      end_of_turn_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, EndOfInputTurn>>())),
      end_of_input_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, EndOfInput>>())),
      has_vision_encoder_(has_vision_encoder) {}

JanusProInputFanOutStage::~JanusProInputFanOutStage() = default;

absl::StatusOr<std::unique_ptr<JanusProInputFanOutStage>>
JanusProInputFanOutStage::Create(
    Stage<OmniSession::Input>* absl_nonnull upstream, bool has_vision_encoder) {
  return absl::WrapUnique(
      new JanusProInputFanOutStage(upstream, has_vision_encoder));
}

absl::Status JanusProInputFanOutStage::RouteInput(OmniSession::Input input) {
  if (std::holds_alternative<AudioInput>(input) ||
      std::holds_alternative<AudioInputMetadata>(input)) {
    return absl::UnimplementedError(
        "Audio input is not supported in JanusProInputFanOutStage.");
  }
  if (std::holds_alternative<ImageGenInputMetadata>(input)) {
    return absl::UnimplementedError(
        "ImageGenInputMetadata is not yet supported in "
        "JanusProInputFanOutStage.");
  }
  if (!has_vision_encoder_ && std::holds_alternative<ImageInput>(input)) {
    return absl::FailedPreconditionError(
        "Image input is not supported because vision_encoder is not loaded.");
  }
  return FanOutStage<OmniSession::Input>::RouteInput(std::move(input));
}

}  // namespace litert::omni::any2any
