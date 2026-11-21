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

#include "models/lfm/lfm2_5_audio/input_fan_out_stage.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {

Lfm2AudioInputFanOutStage::Lfm2AudioInputFanOutStage(
    Stage<OmniSession::Input>* absl_nonnull upstream, bool has_audio_encoder)
    : FanOutStage<OmniSession::Input>(upstream),
      text_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, TextInput>>())),
      audio_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, AudioInput>>())),
      audio_input_metadata_queue_(AddOutputQueue(
          std::make_unique<SequencedVariantQueue<OmniSession::Input,
                                                 AudioInputMetadata>>())),
      end_of_turn_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, EndOfInputTurn>>())),
      has_audio_encoder_(has_audio_encoder) {}

Lfm2AudioInputFanOutStage::~Lfm2AudioInputFanOutStage() = default;

absl::StatusOr<std::unique_ptr<Lfm2AudioInputFanOutStage>>
Lfm2AudioInputFanOutStage::Create(
    Stage<OmniSession::Input>* absl_nonnull upstream, bool has_audio_encoder) {
  return absl::WrapUnique(
      new Lfm2AudioInputFanOutStage(upstream, has_audio_encoder));
}

absl::Status Lfm2AudioInputFanOutStage::FlushPendingAudio() {
  if (!has_pending_audio_) {
    return absl::OkStatus();
  }
  has_pending_audio_ = false;
  AudioInput coalesced;
  coalesced.pcm_samples = std::move(pending_audio_pcm_);
  pending_audio_pcm_.clear();
  const uint64_t seqnum = GetNextSeqnum();
  return audio_queue_->PushBack(seqnum, std::move(coalesced));
}

absl::Status Lfm2AudioInputFanOutStage::RouteInput(OmniSession::Input input) {
  if (std::holds_alternative<EndOfInput>(input)) {
    ABSL_RETURN_IF_ERROR(FlushPendingAudio());
    if (has_open_turn_) {
      has_open_turn_ = false;
      const uint64_t seqnum = GetNextSeqnum();
      return end_of_turn_queue_->PushBack(seqnum, EndOfInputTurn{});
    }
    return absl::OkStatus();
  }
  if (std::holds_alternative<ImageInput>(input) ||
      std::holds_alternative<ImageGenInputMetadata>(input)) {
    return absl::InvalidArgumentError(
        "LiquidAI/LFM2.5-Audio-1.5B does not support visual (image) inputs.");
  }
  if (std::holds_alternative<AudioInputMetadata>(input)) {
    if (!has_audio_encoder_) {
      return absl::FailedPreconditionError(
          "AudioInputMetadata is not supported because the model bundle has no "
          "audio_encoder section.");
    }
    ABSL_RETURN_IF_ERROR(FlushPendingAudio());
    const uint64_t seqnum = GetNextSeqnum();
    return audio_input_metadata_queue_->PushBack(seqnum, std::move(input));
  }
  if (std::holds_alternative<AudioInput>(input)) {
    if (!has_audio_encoder_) {
      return absl::FailedPreconditionError(
          "AudioInput is not supported because the model bundle has no "
          "audio_encoder section.");
    }
    has_open_turn_ = true;
    has_pending_audio_ = true;
    const auto& pcm = std::get<AudioInput>(input).pcm_samples;
    pending_audio_pcm_.insert(pending_audio_pcm_.end(), pcm.begin(), pcm.end());
    return absl::OkStatus();
  }
  if (std::holds_alternative<TextInput>(input)) {
    ABSL_RETURN_IF_ERROR(FlushPendingAudio());
    has_open_turn_ = true;
    const uint64_t seqnum = GetNextSeqnum();
    return text_queue_->PushBack(seqnum, std::move(input));
  }
  if (std::holds_alternative<EndOfInputTurn>(input)) {
    ABSL_RETURN_IF_ERROR(FlushPendingAudio());
    has_open_turn_ = false;
    const uint64_t seqnum = GetNextSeqnum();
    return end_of_turn_queue_->PushBack(seqnum, std::move(input));
  }
  return absl::InvalidArgumentError(
      "Unsupported input variant in Lfm2AudioInputFanOutStage.");
}

absl::Status Lfm2AudioInputFanOutStage::FlushInternal() {
  return FlushPendingAudio();
}

void Lfm2AudioInputFanOutStage::ResetInternal() {
  FanOutStage<OmniSession::Input>::ResetInternal();
  has_open_turn_ = false;
  has_pending_audio_ = false;
  pending_audio_pcm_.clear();
}

}  // namespace litert::omni::any2any
