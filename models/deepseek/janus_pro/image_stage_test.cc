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

#include "models/deepseek/janus_pro/image_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/fan_out_stage.h"
#include "models/deepseek/janus_pro/janus_pro_model_config.h"
#include "models/deepseek/janus_pro/janus_pro_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/omni_session.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::NiceMock;

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockVisionEncoderRunner(
    int image_size, int num_image_tokens, int hidden_size,
    std::vector<float>* observed_input = nullptr) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(_))
      .WillByDefault([image_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto in_buf,
            support::CreateTensorBuffer<float>({1, 3, image_size, image_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(in_buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(_))
      .WillByDefault([num_image_tokens, hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(auto out_buf,
                                support::CreateTensorBuffer<float>(
                                    {1, num_image_tokens, hidden_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  ON_CALL(*runner, Run(_, _, _))
      .WillByDefault(
          [image_size, num_image_tokens, hidden_size, observed_input](
              absl::string_view, absl::Span<const TensorBuffer> input_buffers,
              absl::Span<const TensorBuffer> output_buffers) -> absl::Status {
            std::vector<float> in_vec(3 * image_size * image_size, 0.0f);
            auto& in_buf = const_cast<TensorBuffer&>(input_buffers[0]);
            LITERT_RETURN_IF_ERROR(in_buf.Read<float>(absl::MakeSpan(in_vec)));
            if (observed_input != nullptr) {
              *observed_input = in_vec;
            }
            const float first_px = in_vec.empty() ? 0.0f : in_vec[0];
            std::vector<float> out_vals(num_image_tokens * hidden_size);
            for (size_t i = 0; i < out_vals.size(); ++i) {
              out_vals[i] = 500.0f + first_px + static_cast<float>(i);
            }
            auto& out_buf = const_cast<TensorBuffer&>(output_buffers[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(out_vals)));
            return absl::OkStatus();
          });
  return runner;
}

TEST(JanusProImageStageTest, ValidatesConfigAndBufferShapes) {
  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       JanusProInputFanOutStage::Create(&input_source, true));

  JanusProModelConfig bad_cfg;
  bad_cfg.llm_hidden_size = 0;
  EXPECT_THAT(
      JanusProImageStage::Create(
          fan_out->image_queue(),
          MakeMockVisionEncoderRunner(/*image_size=*/16, /*num_image_tokens=*/4,
                                      /*hidden_size=*/4),
          bad_cfg),
      StatusIs(absl::StatusCode::kInvalidArgument));

  JanusProModelConfig wrong_size_cfg;
  wrong_size_cfg.llm_hidden_size = 4;
  wrong_size_cfg.image_size = 32;  // Runner provides 16x16.
  wrong_size_cfg.num_image_tokens = 4;
  EXPECT_THAT(
      JanusProImageStage::Create(
          fan_out->image_queue(),
          MakeMockVisionEncoderRunner(/*image_size=*/16, /*num_image_tokens=*/4,
                                      /*hidden_size=*/4),
          wrong_size_cfg),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JanusProImageStageTest, PreprocessesRgbRgbaGrayPngAndExpand2Square) {
  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       JanusProInputFanOutStage::Create(&input_source, true));

  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.image_size = 16;
  cfg.vision_patch_size = 8;
  cfg.num_image_tokens = 4;

  std::vector<float> observed_nchw;
  ASSERT_OK_AND_ASSIGN(
      auto image_stage,
      JanusProImageStage::Create(
          fan_out->image_queue(),
          MakeMockVisionEncoderRunner(/*image_size=*/16, /*num_image_tokens=*/4,
                                      /*hidden_size=*/4, &observed_nchw),
          cfg));

  // 1. 2x4 RGB image triggers aspect-ratio resize + expand2square padding
  //    with 127 (which normalizes to `(127/255 - 0.5)/0.5 ~= -0.00392`).
  std::vector<uint8_t> rgb(2 * 4 * 3, 255);
  ASSERT_OK(input_source.PushInput(OmniSession::ImageInput{
      .image_bytes = rgb,
      .width = 2,
      .height = 4,
      .channels = 3,
  }));
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(image_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto out_rgb, image_stage->GetOutput());
  EXPECT_THAT(out_rgb.item.token_ids,
              ElementsAre(janus_pro::kBeginOfImageTokenId,
                          janus_pro::kImageEmbedPlaceholderTokenId,
                          janus_pro::kImageEmbedPlaceholderTokenId,
                          janus_pro::kImageEmbedPlaceholderTokenId,
                          janus_pro::kImageEmbedPlaceholderTokenId,
                          janus_pro::kEndOfImageTokenId));
  EXPECT_EQ(out_rgb.item.embeddings.size(), 16u);
  ASSERT_EQ(observed_nchw.size(), 3u * 16u * 16u);
  // Top-left corner (0, 0) is pad background 127; center pixel (8, 8) is 255.
  EXPECT_NEAR(observed_nchw[0], (127.0f / 255.0f - 0.5f) / 0.5f, 1e-4f);
  EXPECT_NEAR(observed_nchw[8 * 16 + 8], 1.0f, 1e-4f);

  // 2. Grayscale raw image and compressed PNG bytes.
  ASSERT_OK(input_source.PushInput(OmniSession::ImageInput{
      .image_bytes = std::vector<uint8_t>(4 * 4, 0),
      .width = 4,
      .height = 4,
      .channels = 1,
  }));
  ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> png_bytes,
      EncodePngBytes(std::vector<uint8_t>(4 * 4 * 4, 128), /*width=*/4,
                     /*height=*/4, /*channels=*/4));
  ASSERT_OK(input_source.PushInput(OmniSession::ImageInput{
      .image_bytes = std::move(png_bytes),
      .width = 0,
      .height = 0,
      .channels = 0,
  }));
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(image_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto out_gray, image_stage->GetOutput());
  EXPECT_EQ(out_gray.item.embeddings.size(), 16u);
  ASSERT_OK_AND_ASSIGN(auto out_png, image_stage->GetOutput());
  EXPECT_EQ(out_png.item.embeddings.size(), 16u);

  // 3. Invalid channel count returns InvalidArgumentError.
  ASSERT_OK(input_source.PushInput(OmniSession::ImageInput{
      .image_bytes = {1, 2},
      .width = 1,
      .height = 1,
      .channels = 2,
  }));
  ASSERT_OK(fan_out->Schedule());
  EXPECT_THAT(image_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JanusProImageStageTest, RunsOnRealCompiledLiteRtVisionModel) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  Environment env = std::move(*env_or);

  std::vector<float> expected_emb(16, 2.25f);
  const std::string tflite_buf = BuildMinimalTfliteModelBuffer(
      {{"pixel_values", {1, 3, 16, 16}, tflite::TensorType_FLOAT32}},
      {{"vision_embeds", {1, 4, 4}, tflite::TensorType_FLOAT32, expected_emb}});
  ModelOptions opts{
      .cache_dir = ":nocache", .backend = lm::Backend::CPU, .num_threads = 1};
  ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled,
      CreateCompiledModelFromBuffer(env, opts, tflite_buf, "vision_encoder"));

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       JanusProInputFanOutStage::Create(&input_source, true));
  JanusProModelConfig cfg;
  cfg.llm_hidden_size = 4;
  cfg.image_size = 16;
  cfg.vision_patch_size = 8;
  cfg.num_image_tokens = 4;
  ASSERT_OK_AND_ASSIGN(auto image_stage,
                       JanusProImageStage::Create(
                           fan_out->image_queue(),
                           std::make_unique<LiteRtRunnerImpl>(&compiled), cfg));

  ASSERT_OK(input_source.PushInput(OmniSession::ImageInput{
      .image_bytes = std::vector<uint8_t>(4 * 4 * 3, 128),
      .width = 4,
      .height = 4,
      .channels = 3,
  }));
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(image_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto out, image_stage->GetOutput());
  EXPECT_EQ(out.item.embeddings, expected_emb);
}

}  // namespace
}  // namespace litert::omni::any2any
