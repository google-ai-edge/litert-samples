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

#include "models/omnigen/omnigen2/runner/denoiser_stage.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/runner/io_types.h"
#include "models/omnigen/omnigen2/runner/omnigen2_model_config.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
#include "omni/any2any/any2any_test_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/text2image/io_types.h"
#include "omni/text2image/text2image_math.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image::omnigen2 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::FloatNear;

class StubTextEncoderStage
    : public SingleThreadedStageWithDeque<OmniGen2TextEncoderOutput> {
 public:
  void Push(OmniGen2TextEncoderOutput output) { PushOutput(std::move(output)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

OmniGen2ModelConfig MakeSmallConfig() {
  OmniGen2ModelConfig config;
  config.seq_len = 8;
  config.img_size = 32;  // latent 4x4, grid 2x2, img_tokens = 4
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
  config.text_guidance_scale = 2.0f;
  return config;
}

class OmniGen2DenoiserStageTest : public ::testing::Test {
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

TEST_F(OmniGen2DenoiserStageTest,
       MultiStepEulerWithCfgAppliesAsymmetricBranchRopeAndGuidanceScale) {
  const OmniGen2ModelConfig config = MakeSmallConfig();
  constexpr int kCondActiveLen = 5;
  constexpr int kUncondActiveLen = 3;
  constexpr int kImgTokens = 4;
  constexpr int kJointTokens = 8 + kImgTokens;
  constexpr int kTembDim = 16;

  StubTextEncoderStage text_encoder;
  OmniGen2DenoiserStage::Runners runners;

  auto cap_embedder = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *cap_embedder,
      {{ElementType::Float32, {1, config.seq_len, config.text_hidden_size}, 4}},
      {{ElementType::Float32, {1, config.seq_len, config.dit_hidden_size}, 4}});
  EXPECT_CALL(*cap_embedder, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly([&config](absl::string_view,
                                absl::Span<const TensorBuffer> /*inputs*/,
                                absl::Span<const TensorBuffer> outputs) {
        std::vector<float> cap_emb(config.seq_len * config.dit_hidden_size,
                                   0.25f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(cap_emb)
                        .HasValue());
        return absl::OkStatus();
      });
  runners.cap_embedder = std::move(cap_embedder);

  int ctx_call = 0;
  auto context_refiner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *context_refiner,
      {{ElementType::Float32, {1, config.seq_len, config.dit_hidden_size}, 4},
       {ElementType::Float32, {1, config.seq_len, 1, config.dit_head_dim}, 4},
       {ElementType::Float32, {1, config.seq_len, 1, config.dit_head_dim}, 4},
       {ElementType::Float32, {1, 1, 1, config.seq_len}, 4}},
      {{ElementType::Float32, {1, config.seq_len, config.dit_hidden_size}, 4}});
  EXPECT_CALL(*context_refiner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs) {
        const int expected_active =
            (ctx_call++ == 0) ? kCondActiveLen : kUncondActiveLen;
        std::vector<float> cap_mask(config.seq_len, 1.0f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(inputs[3])
                        .Read<float>(absl::MakeSpan(cap_mask))
                        .HasValue());
        for (int s = 0; s < config.seq_len; ++s) {
          EXPECT_FLOAT_EQ(cap_mask[s], s < expected_active ? 0.0f : -1e9f);
        }
        std::vector<float> cap_ref(config.seq_len * config.dit_hidden_size,
                                   0.5f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(cap_ref)
                        .HasValue());
        return absl::OkStatus();
      });
  runners.context_refiner = std::move(context_refiner);

  auto x_embedder = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *x_embedder,
      {{ElementType::Float32, {1, kImgTokens, config.packed_ch}, 4},
       {ElementType::Float32, {1}, 4}},
      {{ElementType::Float32, {1, kImgTokens, config.dit_hidden_size}, 4},
       {ElementType::Float32, {1, kTembDim}, 4}});
  EXPECT_CALL(*x_embedder, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly([&config](absl::string_view,
                                absl::Span<const TensorBuffer> /*inputs*/,
                                absl::Span<const TensorBuffer> outputs) {
        std::vector<float> x_emb(kImgTokens * config.dit_hidden_size, 0.1f);
        std::vector<float> temb(kTembDim, 0.2f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(x_emb)
                        .HasValue());
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[1])
                        .Write<float>(temb)
                        .HasValue());
        return absl::OkStatus();
      });
  runners.x_embedder = std::move(x_embedder);

  int noise_call = 0;
  auto noise_refiner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *noise_refiner,
      {{ElementType::Float32, {1, kImgTokens, config.dit_hidden_size}, 4},
       {ElementType::Float32, {1, kImgTokens, 1, config.dit_head_dim}, 4},
       {ElementType::Float32, {1, kImgTokens, 1, config.dit_head_dim}, 4},
       {ElementType::Float32, {1, kTembDim}, 4}},
      {{ElementType::Float32, {1, kImgTokens, config.dit_hidden_size}, 4}});
  EXPECT_CALL(*noise_refiner, Run(absl::string_view(""), _, _))
      .Times(4)
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs) {
        // Even calls are cond (id_offset=5), odd calls are uncond
        // (id_offset=3).
        const float expected_offset = (noise_call++ % 2 == 0) ? 5.0f : 3.0f;
        std::vector<float> img_cos(kImgTokens * config.dit_head_dim, 0.0f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(inputs[1])
                        .Read<float>(absl::MakeSpan(img_cos))
                        .HasValue());
        EXPECT_THAT(img_cos[0], FloatNear(std::cos(expected_offset), 1e-5f));
        std::vector<float> x_ref(kImgTokens * config.dit_hidden_size, 0.3f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(x_ref)
                        .HasValue());
        return absl::OkStatus();
      });
  runners.noise_refiner = std::move(noise_refiner);

  for (size_t i = 0; i < kNumMainBlockShards; ++i) {
    auto main_block = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *main_block,
        {{ElementType::Float32, {1, kJointTokens, config.dit_hidden_size}, 4},
         {ElementType::Float32, {1, kJointTokens, 1, config.dit_head_dim}, 4},
         {ElementType::Float32, {1, kJointTokens, 1, config.dit_head_dim}, 4},
         {ElementType::Float32, {1, 1, 1, kJointTokens}, 4},
         {ElementType::Float32, {1, kTembDim}, 4}},
        {{ElementType::Float32, {1, kJointTokens, config.dit_hidden_size}, 4}});
    EXPECT_CALL(*main_block, Run(absl::string_view(""), _, _))
        .Times(4)
        .WillRepeatedly([&config](absl::string_view,
                                  absl::Span<const TensorBuffer> inputs,
                                  absl::Span<const TensorBuffer> outputs) {
          std::vector<float> joint(kJointTokens * config.dit_hidden_size, 0.0f);
          EXPECT_TRUE(const_cast<TensorBuffer&>(inputs[0])
                          .Read<float>(absl::MakeSpan(joint))
                          .HasValue());
          EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                          .Write<float>(joint)
                          .HasValue());
          return absl::OkStatus();
        });
    runners.main_blocks[i] = std::move(main_block);
  }

  int final_call = 0;
  auto final_stage = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *final_stage,
      {{ElementType::Float32, {1, kJointTokens, config.dit_hidden_size}, 4},
       {ElementType::Float32, {1, kTembDim}, 4}},
      {{ElementType::Float32, {1, kImgTokens, config.packed_ch}, 4}});
  EXPECT_CALL(*final_stage, Run(absl::string_view(""), _, _))
      .Times(4)
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> /*inputs*/,
                          absl::Span<const TensorBuffer> outputs) {
        // Cond = 1.5f, Uncond = 0.5f; CFG scale 2.0 -> 0.5 + 2*(1.5-0.5) = 2.5.
        const float v = (final_call++ % 2 == 0) ? 1.5f : 0.5f;
        std::vector<float> vel(kImgTokens * config.packed_ch, v);
        EXPECT_TRUE(
            const_cast<TensorBuffer&>(outputs[0]).Write<float>(vel).HasValue());
        return absl::OkStatus();
      });
  runners.final_stage = std::move(final_stage);

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      OmniGen2DenoiserStage::Create(&text_encoder, config, std::move(runners)));

  OmniGen2TextEncoderOutput cond;
  cond.metadata = ImageGenInputMetadata{
      .width = 32, .height = 32, .num_inference_steps = 2, .seed = 42};
  cond.prompt_embeds.assign(config.seq_len * config.text_hidden_size, 0.5f);
  cond.active_seq_len = kCondActiveLen;
  cond.negative_prompt_embeds.assign(config.seq_len * config.text_hidden_size,
                                     0.25f);
  cond.negative_active_seq_len = kUncondActiveLen;
  text_encoder.Push(std::move(cond));

  ASSERT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(DenoisedLatents out, stage->GetOutput());

  std::vector<float> nchw_init =
      SampleGaussianLatents(kImgTokens * config.packed_ch, 42);
  ASSERT_OK_AND_ASSIGN(std::vector<float> packed_init,
                       PackOmniGen2Latents(nchw_init, config.in_channels, 4, 4,
                                           config.patch_size));
  ASSERT_EQ(out.packed_latents.size(), packed_init.size());
  for (size_t i = 0; i < packed_init.size(); ++i) {
    EXPECT_THAT(out.packed_latents[i], FloatNear(packed_init[i] + 2.5f, 1e-5f));
  }
}

TEST_F(OmniGen2DenoiserStageTest,
       RunsWithoutCfgUsingRealCompiledModelsAndInferredActiveCaptionLength) {
  OmniGen2ModelConfig config = MakeSmallConfig();
  config.text_guidance_scale = 1.0f;  // CFG disabled.
  constexpr int kImgTokens = 4;
  constexpr int kJointTokens = 8 + kImgTokens;
  constexpr int kTembDim = 16;

  ModelOptions options;
  options.backend = lm::Backend::CPU;
  options.num_threads = 1;
  options.cache_dir = ":nocache";

  // Keep flatbuffer payloads alive for the full lifetime of the compiled models
  // below, since `CreateCompiledModelFromBuffer` borrows `absl::string_view`.
  std::deque<std::string> model_buffers;
  auto compile = [&](std::string tflite_bytes,
                     absl::string_view name) -> absl::StatusOr<CompiledModel> {
    model_buffers.push_back(std::move(tflite_bytes));
    return CreateCompiledModelFromBuffer(*env_, options, model_buffers.back(),
                                         name);
  };

  ASSERT_OK_AND_ASSIGN(
      CompiledModel cap_emb_m,
      compile(any2any::BuildMinimalTfliteModelBuffer(
                  {{"args_0",
                    {1, config.seq_len, config.text_hidden_size},
                    tflite::TensorType_FLOAT32}},
                  {{"out",
                    {1, config.seq_len, config.dit_hidden_size},
                    tflite::TensorType_FLOAT32}}),
              "cap_emb"));
  ASSERT_OK_AND_ASSIGN(
      CompiledModel ctx_ref_m,
      compile(any2any::BuildMinimalTfliteModelBuffer(
                  {{"args_0",
                    {1, config.seq_len, config.dit_hidden_size},
                    tflite::TensorType_FLOAT32},
                   {"args_1",
                    {1, config.seq_len, 1, config.dit_head_dim},
                    tflite::TensorType_FLOAT32},
                   {"args_2",
                    {1, config.seq_len, 1, config.dit_head_dim},
                    tflite::TensorType_FLOAT32},
                   {"args_3",
                    {1, 1, 1, config.seq_len},
                    tflite::TensorType_FLOAT32}},
                  {{"out",
                    {1, config.seq_len, config.dit_hidden_size},
                    tflite::TensorType_FLOAT32}}),
              "ctx_ref"));
  ASSERT_OK_AND_ASSIGN(
      CompiledModel x_emb_m,
      compile(any2any::BuildMinimalTfliteModelBuffer(
                  {{"args_0",
                    {1, kImgTokens, config.packed_ch},
                    tflite::TensorType_FLOAT32},
                   {"args_1", {1}, tflite::TensorType_FLOAT32}},
                  {{"x_emb",
                    {1, kImgTokens, config.dit_hidden_size},
                    tflite::TensorType_FLOAT32},
                   {"temb", {1, kTembDim}, tflite::TensorType_FLOAT32}}),
              "x_emb"));
  ASSERT_OK_AND_ASSIGN(
      CompiledModel noise_ref_m,
      compile(any2any::BuildMinimalTfliteModelBuffer(
                  {{"args_0",
                    {1, kImgTokens, config.dit_hidden_size},
                    tflite::TensorType_FLOAT32},
                   {"args_1",
                    {1, kImgTokens, 1, config.dit_head_dim},
                    tflite::TensorType_FLOAT32},
                   {"args_2",
                    {1, kImgTokens, 1, config.dit_head_dim},
                    tflite::TensorType_FLOAT32},
                   {"args_3", {1, kTembDim}, tflite::TensorType_FLOAT32}},
                  {{"out",
                    {1, kImgTokens, config.dit_hidden_size},
                    tflite::TensorType_FLOAT32}}),
              "noise_ref"));

  const std::string main_buf = any2any::BuildMinimalTfliteModelBuffer(
      {{"args_0",
        {1, kJointTokens, config.dit_hidden_size},
        tflite::TensorType_FLOAT32},
       {"args_1",
        {1, kJointTokens, 1, config.dit_head_dim},
        tflite::TensorType_FLOAT32},
       {"args_2",
        {1, kJointTokens, 1, config.dit_head_dim},
        tflite::TensorType_FLOAT32},
       {"args_3", {1, 1, 1, kJointTokens}, tflite::TensorType_FLOAT32},
       {"args_4", {1, kTembDim}, tflite::TensorType_FLOAT32}},
      {{"out",
        {1, kJointTokens, config.dit_hidden_size},
        tflite::TensorType_FLOAT32}});
  std::vector<CompiledModel> main_models;
  main_models.reserve(kNumMainBlockShards);
  for (size_t i = 0; i < kNumMainBlockShards; ++i) {
    ASSERT_OK_AND_ASSIGN(CompiledModel m,
                         compile(main_buf, absl::StrCat("main_", i)));
    main_models.push_back(std::move(m));
  }

  std::vector<float> vel_const(kImgTokens * config.packed_ch, 0.75f);
  ASSERT_OK_AND_ASSIGN(
      CompiledModel final_m,
      compile(any2any::BuildMinimalTfliteModelBuffer(
                  {{"args_0",
                    {1, kJointTokens, config.dit_hidden_size},
                    tflite::TensorType_FLOAT32},
                   {"args_1", {1, kTembDim}, tflite::TensorType_FLOAT32}},
                  {{"out",
                    {1, kImgTokens, config.packed_ch},
                    tflite::TensorType_FLOAT32,
                    vel_const}}),
              "final"));

  OmniGen2DenoiserStage::Runners runners;
  runners.cap_embedder = std::make_unique<LiteRtRunnerImpl>(&cap_emb_m);
  runners.context_refiner = std::make_unique<LiteRtRunnerImpl>(&ctx_ref_m);
  runners.x_embedder = std::make_unique<LiteRtRunnerImpl>(&x_emb_m);
  runners.noise_refiner = std::make_unique<LiteRtRunnerImpl>(&noise_ref_m);
  for (size_t i = 0; i < kNumMainBlockShards; ++i) {
    runners.main_blocks[i] =
        std::make_unique<LiteRtRunnerImpl>(&main_models[i]);
  }
  runners.final_stage = std::make_unique<LiteRtRunnerImpl>(&final_m);

  StubTextEncoderStage text_encoder;
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      OmniGen2DenoiserStage::Create(&text_encoder, config, std::move(runners)));

  // Leave active_seq_len = 0 to exercise InferActiveCaptionLength fallback.
  OmniGen2TextEncoderOutput cond;
  cond.metadata.seed = 7;
  cond.prompt_embeds.assign(config.seq_len * config.text_hidden_size, 0.0f);
  cond.prompt_embeds[2 * config.text_hidden_size] = 1.0f;
  text_encoder.Push(std::move(cond));

  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(DenoisedLatents out, stage->GetOutput());

  std::vector<float> nchw_init =
      SampleGaussianLatents(kImgTokens * config.packed_ch, 7);
  ASSERT_OK_AND_ASSIGN(std::vector<float> packed_init,
                       PackOmniGen2Latents(nchw_init, config.in_channels, 4, 4,
                                           config.patch_size));
  ASSERT_EQ(out.packed_latents.size(), packed_init.size());
  EXPECT_THAT(out.packed_latents[0], FloatNear(packed_init[0] + 0.75f, 1e-5f));
}

TEST_F(OmniGen2DenoiserStageTest, RejectsNullRunnersAndInvalidInputs) {
  const OmniGen2ModelConfig config = MakeSmallConfig();
  StubTextEncoderStage text_encoder;

  OmniGen2DenoiserStage::Runners null_runners;
  EXPECT_THAT(OmniGen2DenoiserStage::Create(&text_encoder, config,
                                            std::move(null_runners)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_cfg = config;
  bad_cfg.img_size = 30;
  OmniGen2DenoiserStage::Runners bad_cfg_runners;
  EXPECT_THAT(OmniGen2DenoiserStage::Create(&text_encoder, bad_cfg,
                                            std::move(bad_cfg_runners)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::text2image::omnigen2
