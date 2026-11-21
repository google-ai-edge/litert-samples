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

#include "models/omnigen/omnigen2/vae_decoder_stage.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/omnigen/omnigen2/io_types.h"
#include "models/omnigen/omnigen2/omnigen2_model_config.h"
#include "models/omnigen/omnigen2/omnigen2_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image::omnigen2 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::FloatNear;

class StubDenoiserStage
    : public SingleThreadedStageWithDeque<OmniGen2DenoiserOutput> {
 public:
  void Push(OmniGen2DenoiserOutput output) { PushOutput(std::move(output)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

OmniGen2ModelConfig MakeSmallConfig() {
  OmniGen2ModelConfig config;
  config.seq_len = 8;
  config.img_size = 32;  // latent 4x4, grid 2x2
  config.in_channels = 16;
  config.patch_size = 2;
  config.packed_ch = 64;
  config.text_hidden_size = 8;
  config.dit_hidden_size = 48;
  config.dit_num_heads = 3;
  config.dit_num_kv_heads = 1;
  config.dit_head_dim = 16;
  config.axes_dims_rope = {4, 6, 6};
  config.steps = 2;
  config.vae_scaling_factor = 0.5f;
  config.vae_shift_factor = 0.25f;
  return config;
}

class OmniGen2VaeDecoderStageTest : public ::testing::Test {
 protected:
  struct BufferSpec {
    ElementType element_type;
    std::vector<int> dims;
    size_t elem_size;
  };

  void SetUp() override {
    auto env = Environment::Create({});
    ASSERT_TRUE(env.HasValue());
    env_ = std::make_unique<Environment>(std::move(*env));
  }

  absl::StatusOr<TensorBuffer> CreateBuffer(ElementType element_type,
                                            const std::vector<int>& dims,
                                            size_t elem_size) {
    RankedTensorType type(element_type,
                          Layout(Dimensions(dims.begin(), dims.end())));
    size_t num_elements = 1;
    for (int d : dims) num_elements *= d;
    LITERT_ASSIGN_OR_RETURN(
        auto buf,
        TensorBuffer::CreateManaged(*env_, TensorBufferType::kHostMemory,
                                    std::move(type), num_elements * elem_size));
    return buf;
  }

  void ExpectBuffers(MockLiteRtRunner& runner,
                     std::vector<BufferSpec> input_specs,
                     std::vector<BufferSpec> output_specs) {
    EXPECT_CALL(runner, CreateInputBuffers(absl::string_view("")))
        .WillOnce(
            [this, input_specs = std::move(input_specs)](absl::string_view)
                -> absl::StatusOr<std::vector<TensorBuffer>> {
              std::vector<TensorBuffer> bufs;
              bufs.reserve(input_specs.size());
              for (const auto& spec : input_specs) {
                ABSL_ASSIGN_OR_RETURN(
                    TensorBuffer buf,
                    CreateBuffer(spec.element_type, spec.dims, spec.elem_size));
                bufs.push_back(std::move(buf));
              }
              return bufs;
            });
    EXPECT_CALL(runner, CreateOutputBuffers(absl::string_view("")))
        .WillOnce(
            [this, output_specs = std::move(output_specs)](absl::string_view)
                -> absl::StatusOr<std::vector<TensorBuffer>> {
              std::vector<TensorBuffer> bufs;
              bufs.reserve(output_specs.size());
              for (const auto& spec : output_specs) {
                ABSL_ASSIGN_OR_RETURN(
                    TensorBuffer buf,
                    CreateBuffer(spec.element_type, spec.dims, spec.elem_size));
                bufs.push_back(std::move(buf));
              }
              return bufs;
            });
  }

  std::unique_ptr<Environment> env_;
};

TEST_F(OmniGen2VaeDecoderStageTest,
       UnpacksScalesLatentsAndConvertsNchwToClampedRgb888) {
  const OmniGen2ModelConfig config = MakeSmallConfig();
  constexpr int kLatentDim = 4;
  constexpr int kImgPixels = 32 * 32;

  StubDenoiserStage denoiser;
  auto runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *runner,
      {{ElementType::Float32,
        {1, config.in_channels, kLatentDim, kLatentDim},
        4}},
      {{ElementType::Float32, {1, 3, config.img_size, config.img_size}, 4}});

  std::vector<float> nchw_expected(config.in_channels * kLatentDim *
                                   kLatentDim);
  for (size_t i = 0; i < nchw_expected.size(); ++i) {
    nchw_expected[i] = static_cast<float>(i) * 0.05f;
  }
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> packed,
      PackOmniGen2Latents(nchw_expected, config.in_channels, kLatentDim,
                          kLatentDim, config.patch_size));

  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillOnce([&](absl::string_view, absl::Span<const TensorBuffer> inputs,
                    absl::Span<const TensorBuffer> outputs) {
        std::vector<float> z_in(nchw_expected.size(), 0.0f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(inputs[0])
                        .Read<float>(absl::MakeSpan(z_in))
                        .HasValue());
        for (size_t i = 0; i < nchw_expected.size(); ++i) {
          const float expected = nchw_expected[i] / config.vae_scaling_factor +
                                 config.vae_shift_factor;
          EXPECT_THAT(z_in[i], FloatNear(expected, 1e-5f));
        }

        // Write planar NCHW with R=-1.5 (clamps to 0), G=0.0 (128), B=1.5
        // (clamps to 255).
        std::vector<float> img_nchw(3 * kImgPixels, 0.0f);
        for (int p = 0; p < kImgPixels; ++p) {
          img_nchw[p] = -1.5f;
          img_nchw[kImgPixels + p] = 0.0f;
          img_nchw[2 * kImgPixels + p] = 1.5f;
        }
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(img_nchw)
                        .HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(auto stage, OmniGen2VaeDecoderStage::Create(
                                       &denoiser, config, std::move(runner)));

  EXPECT_FALSE(stage->NeedSchedule());
  OmniGen2DenoiserOutput denoised;
  denoised.packed_latents = std::move(packed);
  denoiser.Push(std::move(denoised));

  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->Flush());
  ASSERT_TRUE(stage->HasOutput());

  ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
  const auto& img = std::get<ImageOutput>(out);
  EXPECT_EQ(img.width, 32);
  EXPECT_EQ(img.height, 32);
  EXPECT_EQ(img.channels, 3);
  ASSERT_EQ(img.rgb_data.size(), static_cast<size_t>(kImgPixels * 3));
  EXPECT_EQ(img.rgb_data[0], 0);
  EXPECT_EQ(img.rgb_data[1], 128);
  EXPECT_EQ(img.rgb_data[2], 255);
}

TEST_F(OmniGen2VaeDecoderStageTest, RunsWithRealCompiledModel) {
  const OmniGen2ModelConfig config = MakeSmallConfig();
  constexpr int kLatentDim = 4;
  std::vector<float> decoded_vals(3 * config.img_size * config.img_size, 1.0f);
  const std::string tflite_buf = BuildMinimalTfliteModelBuffer(
      {{"args_0",
        {1, config.in_channels, kLatentDim, kLatentDim},
        tflite::TensorType_FLOAT32}},
      {{"image",
        {1, 3, config.img_size, config.img_size},
        tflite::TensorType_FLOAT32,
        decoded_vals}});

  ModelOptions options;
  options.backend = lm::Backend::CPU;
  options.num_threads = 1;
  options.cache_dir = ":nocache";
  ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled,
      CreateCompiledModelFromBuffer(*env_, options, tflite_buf, "test_vae"));

  StubDenoiserStage denoiser;
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      OmniGen2VaeDecoderStage::Create(
          &denoiser, config, std::make_unique<LiteRtRunnerImpl>(&compiled)));

  OmniGen2DenoiserOutput denoised;
  denoised.packed_latents.assign(4 * config.packed_ch, 0.0f);
  denoiser.Push(std::move(denoised));

  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
  const auto& img = std::get<ImageOutput>(out);
  EXPECT_EQ(img.width, 32);
  EXPECT_EQ(img.height, 32);
  EXPECT_EQ(img.rgb_data[0], 255);
}

TEST_F(OmniGen2VaeDecoderStageTest, RejectsInvalidRunnerConfigAndLatents) {
  const OmniGen2ModelConfig config = MakeSmallConfig();
  StubDenoiserStage denoiser;

  // Wrong output buffer size.
  auto bad_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*bad_runner,
                {{ElementType::Float32, {1, config.in_channels, 4, 4}, 4}},
                {{ElementType::Float32, {1, 3, 16, 16}, 4}});
  EXPECT_THAT(
      OmniGen2VaeDecoderStage::Create(&denoiser, config, std::move(bad_runner)),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong packed_latents size at Schedule() time.
  auto ok_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*ok_runner,
                {{ElementType::Float32, {1, config.in_channels, 4, 4}, 4}},
                {{ElementType::Float32, {1, 3, 32, 32}, 4}});
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      OmniGen2VaeDecoderStage::Create(&denoiser, config, std::move(ok_runner)));

  OmniGen2DenoiserOutput bad_denoised;
  bad_denoised.packed_latents = {1.0f, 2.0f};
  denoiser.Push(std::move(bad_denoised));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::text2image::omnigen2
