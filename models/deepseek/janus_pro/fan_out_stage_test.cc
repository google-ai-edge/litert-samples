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

#include "models/deepseek/janus_pro/fan_out_stage.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "omni/omni_session.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;

TEST(JanusProFanOutStageTest, RoutesSupportedInputsInAscendingSeqnumOrder) {
  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<JanusProInputFanOutStage> fan_out,
      JanusProInputFanOutStage::Create(&input_source,
                                       /*has_vision_encoder=*/true));

  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(input_source.PushInput(OmniSession::ImageInput{
      .image_bytes = {10, 20, 30},
      .width = 1,
      .height = 1,
      .channels = 3,
  }));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  input_source.Finish();

  ASSERT_OK(fan_out->Schedule());
  EXPECT_FALSE(fan_out->text_queue()->Empty());
  EXPECT_FALSE(fan_out->image_queue()->Empty());
  EXPECT_FALSE(fan_out->end_of_turn_queue()->Empty());
  EXPECT_FALSE(fan_out->end_of_input_queue()->Empty());

  ASSERT_OK_AND_ASSIGN(auto text_item, fan_out->text_queue()->PopFront());
  ASSERT_OK_AND_ASSIGN(auto image_item, fan_out->image_queue()->PopFront());
  ASSERT_OK_AND_ASSIGN(auto eot_item, fan_out->end_of_turn_queue()->PopFront());
  ASSERT_OK_AND_ASSIGN(auto eoi_item,
                       fan_out->end_of_input_queue()->PopFront());
  EXPECT_EQ(text_item.seqnum, 0u);
  EXPECT_EQ(image_item.seqnum, 1u);
  EXPECT_EQ(eot_item.seqnum, 2u);
  EXPECT_EQ(eoi_item.seqnum, 3u);
}

TEST(JanusProFanOutStageTest, RejectsUnsupportedInputModalities) {
  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<JanusProInputFanOutStage> fan_out,
      JanusProInputFanOutStage::Create(&input_source,
                                       /*has_vision_encoder=*/true));

  ASSERT_OK(input_source.PushInput(
      OmniSession::AudioInputMetadata{.sample_rate_hz = 16000}));
  EXPECT_THAT(fan_out->Schedule(), StatusIs(absl::StatusCode::kUnimplemented));

  ASSERT_OK(
      input_source.PushInput(OmniSession::AudioInput{.pcm_samples = {0.1f}}));
  EXPECT_THAT(fan_out->Schedule(), StatusIs(absl::StatusCode::kUnimplemented));

  ASSERT_OK(
      input_source.PushInput(OmniSession::ImageGenInputMetadata{.width = 384}));
  EXPECT_THAT(fan_out->Schedule(), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(JanusProFanOutStageTest, RejectsImageInputWithoutVisionEncoder) {
  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<JanusProInputFanOutStage> fan_out,
      JanusProInputFanOutStage::Create(&input_source,
                                       /*has_vision_encoder=*/false));

  ASSERT_OK(input_source.PushInput(OmniSession::ImageInput{
      .image_bytes = {10, 20, 30},
      .width = 1,
      .height = 1,
      .channels = 3,
  }));
  EXPECT_THAT(fan_out->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_TRUE(fan_out->image_queue()->Empty());
}

}  // namespace
}  // namespace litert::omni::any2any
