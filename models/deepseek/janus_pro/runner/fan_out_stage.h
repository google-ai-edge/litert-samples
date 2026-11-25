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

#ifndef MODELS_DEEPSEEK_JANUS_PRO_RUNNER_FAN_OUT_STAGE_H_
#define MODELS_DEEPSEEK_JANUS_PRO_RUNNER_FAN_OUT_STAGE_H_

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {

// Fan-out stage that routes `OmniSession::Input` items to Janus-Pro modality
// queues (`TextInput`, `ImageInput`, `EndOfInputTurn`, and `EndOfInput`).
class JanusProInputFanOutStage : public FanOutStage<OmniSession::Input> {
 public:
  // Creates a `JanusProInputFanOutStage` pulling inputs from `upstream` and
  // registering output queues for `TextInput`, `ImageInput`, `EndOfInputTurn`,
  // and `EndOfInput`. When `has_vision_encoder` is false, `ImageInput` is
  // rejected with `FailedPreconditionError` before assigning a sequence number.
  static absl::StatusOr<std::unique_ptr<JanusProInputFanOutStage>> Create(
      Stage<OmniSession::Input>* absl_nonnull upstream,
      bool has_vision_encoder = true);

  ~JanusProInputFanOutStage() override;

  // Returns the output queue for `TextInput` items.
  FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue() {
    return text_queue_;
  }

  // Returns the output queue for `ImageInput` items.
  FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull
  image_queue() {
    return image_queue_;
  }

  // Returns the output queue for `EndOfInputTurn` control items.
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
  end_of_turn_queue() {
    return end_of_turn_queue_;
  }

  // Returns the output queue for `EndOfInput` end-of-stream control items.
  FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
  end_of_input_queue() {
    return end_of_input_queue_;
  }

 protected:
  absl::Status RouteInput(OmniSession::Input input) override;

 private:
  JanusProInputFanOutStage(Stage<OmniSession::Input>* absl_nonnull upstream,
                           bool has_vision_encoder);

  FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue_;
  FanOutOutputQueue<OmniSession::Input, ImageInput>* absl_nonnull image_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
      end_of_turn_queue_;
  FanOutOutputQueue<OmniSession::Input, EndOfInput>* absl_nonnull
      end_of_input_queue_;
  bool has_vision_encoder_ = true;
};

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_RUNNER_FAN_OUT_STAGE_H_
