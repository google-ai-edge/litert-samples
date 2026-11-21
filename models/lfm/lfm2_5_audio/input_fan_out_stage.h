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

#ifndef MODELS_LFM_LFM2_5_AUDIO_INPUT_FAN_OUT_STAGE_H_
#define MODELS_LFM_LFM2_5_AUDIO_INPUT_FAN_OUT_STAGE_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {

// Fan-out stage for `LiquidAI/LFM2.5-Audio-1.5B` that routes `TextInput`,
// `AudioInput`, `AudioInputMetadata`, and `EndOfInputTurn` into modality
// queues, coalesces consecutive `AudioInput` chunks into a single utterance
// segment before flushing on modality/turn/metadata boundaries, and rejects
// unsupported visual inputs (`ImageInput`, `ImageGenInputMetadata`).
class Lfm2AudioInputFanOutStage : public FanOutStage<OmniSession::Input> {
 public:
  // Creates an `Lfm2AudioInputFanOutStage` connected to `upstream`.
  //
  // Args:
  //   upstream: Non-null upstream `Stage<OmniSession::Input>`.
  //   has_audio_encoder: Whether the model bundle includes `audio_encoder`.
  //
  // Returns:
  //   Initialized `Lfm2AudioInputFanOutStage`.
  static absl::StatusOr<std::unique_ptr<Lfm2AudioInputFanOutStage>> Create(
      Stage<OmniSession::Input>* absl_nonnull upstream,
      bool has_audio_encoder = true);

  ~Lfm2AudioInputFanOutStage() override;

  FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue() {
    return text_queue_;
  }
  FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
  audio_queue() {
    return audio_queue_;
  }
  FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
  audio_input_metadata_queue() {
    return audio_input_metadata_queue_;
  }
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
  end_of_turn_queue() {
    return end_of_turn_queue_;
  }

 protected:
  absl::Status RouteInput(OmniSession::Input input) override;
  absl::Status FlushInternal() override;
  void ResetInternal() override;

 private:
  Lfm2AudioInputFanOutStage(Stage<OmniSession::Input>* absl_nonnull upstream,
                            bool has_audio_encoder);

  absl::Status FlushPendingAudio();

  FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue_;
  FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue_;
  FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
      audio_input_metadata_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
      end_of_turn_queue_;
  bool has_audio_encoder_ = true;
  bool has_open_turn_ = false;
  bool has_pending_audio_ = false;
  std::vector<float> pending_audio_pcm_;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_INPUT_FAN_OUT_STAGE_H_
