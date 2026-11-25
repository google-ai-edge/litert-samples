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

#include "models/deepseek/janus_pro/runner/image_gen_stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/deepseek/janus_pro/runner/io_types.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/any2any/any2any_test_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
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

class FakeLlmOutputStage
    : public SingleThreadedStageWithDeque<JanusProLlmOutput> {
 public:
  void Push(JanusProLlmOutput item) { PushOutput(std::move(item)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockVqDecoderRunner(
    int num_image_tokens, int image_size, std::vector<float> nchw_output) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(_))
      .WillByDefault([num_image_tokens](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto in_buf,
            support::CreateTensorBuffer<int32_t>({1, num_image_tokens}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(in_buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(_))
      .WillByDefault([image_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto out_buf,
            support::CreateTensorBuffer<float>({1, 3, image_size, image_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  ON_CALL(*runner, Run(_, _, _))
      .WillByDefault(
          [nchw_output = std::move(nchw_output)](
              absl::string_view, absl::Span<const TensorBuffer>,
              absl::Span<const TensorBuffer> output_buffers) -> absl::Status {
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(output_buffers[0])
                    .Write<float>(absl::MakeConstSpan(nchw_output)));
            return absl::OkStatus();
          });
  return runner;
}

TEST(JanusProImageGenStageTest, ForwardsTextAndRejectsGridWithoutDecoder) {
  FakeLlmOutputStage llm_stage;
  JanusProModelConfig cfg;
  cfg.image_size = 2;
  cfg.num_image_tokens = 4;
  cfg.vq_codebook_size = 8;

  ASSERT_OK_AND_ASSIGN(auto stage,
                       JanusProImageGenStage::Create(&llm_stage, cfg, nullptr));

  llm_stage.Push(JanusProLlmOutput{.payload = std::string("Caption text")});
  ASSERT_OK(stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Output out1, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text, "Caption text");

  ASSERT_OK_AND_ASSIGN(Output out2, stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out2));

  // Without a decoder, a VQ code grid returns kFailedPrecondition.
  llm_stage.Push(JanusProLlmOutput{
      .payload = JanusProVqCodeGrid{
          .vq_codes = {0, 1, 2, 3}, .image_width = 2, .image_height = 2}});
  EXPECT_THAT(stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(JanusProImageGenStageTest, CreateValidatesConfigAndDecoderBuffers) {
  FakeLlmOutputStage llm_stage;
  JanusProModelConfig bad_cfg;
  bad_cfg.num_image_tokens = 0;
  bad_cfg.image_size = 2;
  EXPECT_THAT(JanusProImageGenStage::Create(
                  &llm_stage, bad_cfg,
                  MakeMockVqDecoderRunner(/*num_image_tokens=*/4,
                                          /*image_size=*/2,
                                          std::vector<float>(12, 0.0f))),
              StatusIs(absl::StatusCode::kInvalidArgument));

  JanusProModelConfig cfg;
  cfg.num_image_tokens = 4;
  cfg.image_size = 2;
  // Wrong input size (2 != 4).
  EXPECT_THAT(JanusProImageGenStage::Create(
                  &llm_stage, cfg,
                  MakeMockVqDecoderRunner(/*num_image_tokens=*/2,
                                          /*image_size=*/2,
                                          std::vector<float>(12, 0.0f))),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Wrong output size (3*4*4 != 3*2*2).
  EXPECT_THAT(JanusProImageGenStage::Create(
                  &llm_stage, cfg,
                  MakeMockVqDecoderRunner(/*num_image_tokens=*/4,
                                          /*image_size=*/4,
                                          std::vector<float>(48, 0.0f))),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JanusProImageGenStageTest, DecodesVqCodeGridNchwToClampedHwcRgb888) {
  FakeLlmOutputStage llm_stage;
  JanusProModelConfig cfg;
  cfg.image_size = 2;
  cfg.num_image_tokens = 4;
  cfg.vq_codebook_size = 8;

  // 2x2 image in NCHW planar layout:
  //   R plane: [ 1.5, -2.0, 0.0,  1.0] -> clamped [255,   0, 127, 255]
  //   G plane: [ 0.0,  1.0, -1.0, 0.0] -> clamped [127, 255,   0, 127]
  //   B plane: [-1.0,  0.0,  1.0, -1.0] -> clamped [  0, 127, 255,   0]
  const std::vector<float> nchw = {
      1.5f,  -2.0f, 0.0f,  1.0f,   // R
      0.0f,  1.0f,  -1.0f, 0.0f,   // G
      -1.0f, 0.0f,  1.0f,  -1.0f,  // B
  };
  ASSERT_OK_AND_ASSIGN(auto stage,
                       JanusProImageGenStage::Create(
                           &llm_stage, cfg,
                           MakeMockVqDecoderRunner(/*num_image_tokens=*/4,
                                                   /*image_size=*/2, nchw)));

  llm_stage.Push(JanusProLlmOutput{
      .payload = JanusProVqCodeGrid{
          .vq_codes = {0, 1, 2, 7}, .image_width = 2, .image_height = 2}});
  ASSERT_OK(stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Output out1, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(out1));
  const auto& img = std::get<ImageOutput>(out1);
  EXPECT_EQ(img.width, 2);
  EXPECT_EQ(img.height, 2);
  EXPECT_EQ(img.channels, 3);
  EXPECT_THAT(img.rgb_data, ElementsAre(255, 127, 0,    // pixel (0, 0)
                                        0, 255, 127,    // pixel (1, 0)
                                        127, 0, 255,    // pixel (0, 1)
                                        255, 127, 0));  // pixel (1, 1)

  ASSERT_OK_AND_ASSIGN(Output out2, stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out2));
}

TEST(JanusProImageGenStageTest, RejectsInvalidCodeCountRangeAndDimensions) {
  FakeLlmOutputStage llm_stage;
  JanusProModelConfig cfg;
  cfg.image_size = 2;
  cfg.num_image_tokens = 4;
  cfg.vq_codebook_size = 8;

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      JanusProImageGenStage::Create(
          &llm_stage, cfg,
          MakeMockVqDecoderRunner(/*num_image_tokens=*/4, /*image_size=*/2,
                                  std::vector<float>(12, 0.0f))));

  // Wrong code count.
  llm_stage.Push(JanusProLlmOutput{
      .payload = JanusProVqCodeGrid{
          .vq_codes = {0, 1}, .image_width = 2, .image_height = 2}});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Negative code.
  llm_stage.Push(JanusProLlmOutput{
      .payload = JanusProVqCodeGrid{
          .vq_codes = {0, -1, 2, 3}, .image_width = 2, .image_height = 2}});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Out-of-range code (>= vq_codebook_size).
  llm_stage.Push(JanusProLlmOutput{
      .payload = JanusProVqCodeGrid{
          .vq_codes = {0, 1, 8, 3}, .image_width = 2, .image_height = 2}});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Dimension mismatch.
  llm_stage.Push(JanusProLlmOutput{
      .payload = JanusProVqCodeGrid{
          .vq_codes = {0, 1, 2, 3}, .image_width = 4, .image_height = 2}});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JanusProImageGenStageTest, DecodesVqCodeGridWithRealCompiledTfliteModel) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  const ModelOptions cpu_opts{
      .cache_dir = ":nocache",
      .backend = lm::Backend::CPU,
      .num_threads = 1,
  };

  // 4x4 image with constant R=-1.0 (0), G=1.0 (255), B=0.0 (127).
  std::vector<float> nchw(3 * 16, 0.0f);
  for (int i = 0; i < 16; ++i) {
    nchw[i] = -1.0f;
    nchw[16 + i] = 1.0f;
    nchw[32 + i] = 0.0f;
  }
  std::string tflite_buf = BuildMinimalTfliteModelBuffer(
      {{"vq_codes", {1, 4}, tflite::TensorType_INT32}},
      {{"rgb_image", {1, 3, 4, 4}, tflite::TensorType_FLOAT32, nchw}});
  ASSERT_OK_AND_ASSIGN(CompiledModel model,
                       CreateCompiledModelFromBuffer(*env, cpu_opts, tflite_buf,
                                                     "vq_image_decoder"));

  FakeLlmOutputStage llm_stage;
  JanusProModelConfig cfg;
  cfg.image_size = 4;
  cfg.num_image_tokens = 4;
  cfg.vq_codebook_size = 8;

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      JanusProImageGenStage::Create(
          &llm_stage, cfg, std::make_unique<LiteRtRunnerImpl>(&model)));

  llm_stage.Push(JanusProLlmOutput{
      .payload = JanusProVqCodeGrid{
          .vq_codes = {1, 2, 3, 4}, .image_width = 4, .image_height = 4}});
  ASSERT_OK(stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Output out1, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(out1));
  const auto& img = std::get<ImageOutput>(out1);
  ASSERT_EQ(img.rgb_data.size(), 4u * 4u * 3u);
  EXPECT_EQ(img.rgb_data[0], 0);
  EXPECT_EQ(img.rgb_data[1], 255);
  EXPECT_EQ(img.rgb_data[2], 127);

  ASSERT_OK_AND_ASSIGN(Output out2, stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out2));
}

}  // namespace
}  // namespace litert::omni::any2any
