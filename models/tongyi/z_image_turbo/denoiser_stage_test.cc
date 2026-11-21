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

#include "models/tongyi/z_image_turbo/denoiser_stage.h"

#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/tongyi/z_image_turbo/io_types.h"
#include "models/tongyi/z_image_turbo/z_image_turbo_model_config.h"
#include "models/tongyi/z_image_turbo/z_image_turbo_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_cache.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/text2image/text_encoder_stage.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;
using ::testing::FloatNear;
using ::testing::Pointwise;

class FakeTextEncoderStage
    : public SingleThreadedStageWithDeque<TextEncoderOutput> {
 public:
  void Push(TextEncoderOutput out) { PushOutput(std::move(out)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

absl::Status ReadFloatBuffer(const TensorBuffer& buffer,
                             absl::Span<float> dst) {
  LITERT_ASSIGN_OR_RETURN(auto lock_and_ptr,
                          TensorBufferScopedLock::Create<const float>(
                              buffer, TensorBuffer::LockMode::kRead));
  absl::c_copy(absl::MakeConstSpan(lock_and_ptr.second, dst.size()),
               dst.begin());
  return absl::OkStatus();
}

absl::Status WriteFloatBuffer(const TensorBuffer& buffer,
                              absl::Span<const float> src) {
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_ptr,
      TensorBufferScopedLock::Create<float>(const_cast<TensorBuffer&>(buffer),
                                            TensorBuffer::LockMode::kWrite));
  absl::c_copy(src, lock_and_ptr.second);
  return absl::OkStatus();
}

// Lightweight fake `LiteRtRunner` backed by host `TensorBuffer`s so stage tests
// can inspect exact tensors (RoPE tables, timestep values, Euler latents)
// written by `ZImageDenoiserStage`.
class RecordingLiteRtRunner : public LiteRtRunner {
 public:
  using RunCallback = std::function<absl::Status(
      absl::Span<const TensorBuffer>, absl::Span<const TensorBuffer>)>;

  RecordingLiteRtRunner(Environment& env,
                        std::vector<std::vector<int>> input_shapes,
                        std::vector<std::vector<int>> output_shapes,
                        RunCallback on_run = nullptr)
      : env_(env),
        input_shapes_(std::move(input_shapes)),
        output_shapes_(std::move(output_shapes)),
        on_run_(std::move(on_run)) {}

  absl::StatusOr<std::vector<TensorBuffer>> CreateInputBuffers(
      absl::string_view /*signature_name*/) override {
    std::vector<TensorBuffer> bufs;
    bufs.reserve(input_shapes_.size());
    for (const auto& shape : input_shapes_) {
      ABSL_ASSIGN_OR_RETURN(TensorBuffer buf,
                            MakeFloatTensorBuffer(env_, shape));
      bufs.push_back(std::move(buf));
    }
    return bufs;
  }

  absl::StatusOr<std::vector<TensorBuffer>> CreateOutputBuffers(
      absl::string_view /*signature_name*/) override {
    std::vector<TensorBuffer> bufs;
    bufs.reserve(output_shapes_.size());
    for (const auto& shape : output_shapes_) {
      ABSL_ASSIGN_OR_RETURN(TensorBuffer buf,
                            MakeFloatTensorBuffer(env_, shape));
      bufs.push_back(std::move(buf));
    }
    return bufs;
  }

  absl::Status Run(absl::string_view /*signature_name*/,
                   absl::Span<const TensorBuffer> inputs,
                   absl::Span<const TensorBuffer> outputs) override {
    if (on_run_) {
      return on_run_(inputs, outputs);
    }
    return absl::OkStatus();
  }

 private:
  Environment& env_;
  std::vector<std::vector<int>> input_shapes_;
  std::vector<std::vector<int>> output_shapes_;
  RunCallback on_run_;
};

absl::StatusOr<ZImageDenoiserStage::Runners> MakeRecordingRunners(
    Environment& env, const ZImageTurboModelConfig& cfg,
    RecordingLiteRtRunner::RunCallback refc_cb = nullptr,
    RecordingLiteRtRunner::RunCallback embx_cb = nullptr,
    RecordingLiteRtRunner::RunCallback refx_cb = nullptr,
    RecordingLiteRtRunner::RunCallback main_cb = nullptr,
    RecordingLiteRtRunner::RunCallback final_cb = nullptr) {
  const int seq = cfg.seq_len;
  const int img = cfg.ImageTokens();
  const int total = cfg.TotalTokens();
  const int dim = cfg.dim;
  const int pdim = cfg.prompt_dim;
  const int pch = cfg.PackedChannels();
  const int hdim = cfg.HeadDim();
  const int adaln = cfg.AdalnDim();

  ZImageDenoiserStage::Runners runners;
  runners.cap_embedder = std::make_unique<RecordingLiteRtRunner>(
      env, std::vector<std::vector<int>>{{1, seq, pdim}},
      std::vector<std::vector<int>>{{1, seq, dim}});
  runners.context_refiner = std::make_unique<RecordingLiteRtRunner>(
      env,
      std::vector<std::vector<int>>{
          {1, seq, dim}, {1, seq, 1, hdim}, {1, seq, 1, hdim}},
      std::vector<std::vector<int>>{{1, seq, dim}}, std::move(refc_cb));
  runners.x_embedder = std::make_unique<RecordingLiteRtRunner>(
      env, std::vector<std::vector<int>>{{1, img, pch}, {1}},
      std::vector<std::vector<int>>{{1, img, dim}, {1, adaln}},
      std::move(embx_cb));
  runners.noise_refiner = std::make_unique<RecordingLiteRtRunner>(
      env,
      std::vector<std::vector<int>>{
          {1, img, dim}, {1, img, 1, hdim}, {1, img, 1, hdim}, {1, adaln}},
      std::vector<std::vector<int>>{{1, img, dim}}, std::move(refx_cb));
  runners.main_blocks.push_back(std::make_unique<RecordingLiteRtRunner>(
      env,
      std::vector<std::vector<int>>{{1, total, dim},
                                    {1, total, 1, hdim},
                                    {1, total, 1, hdim},
                                    {1, adaln}},
      std::vector<std::vector<int>>{{1, total, dim}}, std::move(main_cb)));
  runners.final_layer = std::make_unique<RecordingLiteRtRunner>(
      env, std::vector<std::vector<int>>{{1, total, dim}, {1, adaln}},
      std::vector<std::vector<int>>{{1, img, pch}}, std::move(final_cb));
  return runners;
}

absl::StatusOr<Environment> CreateTestEnvironment() {
  LITERT_ASSIGN_OR_RETURN(Environment env,
                          Environment::Create(EnvironmentOptions(
                              absl::Span<const EnvironmentOptions::Option>{})));
  return env;
}

TEST(ZImageDenoiserStageTest, CreateValidatesInputsAndConfig) {
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  FakeTextEncoderStage text_encoder;
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();

  ZImageTurboModelConfig bad_cfg = cfg;
  bad_cfg.img_size = 12;  // Not divisible by 8 * patch_size.
  ASSERT_OK_AND_ASSIGN(ZImageDenoiserStage::Runners runners,
                       MakeRecordingRunners(env, cfg));
  EXPECT_THAT(
      ZImageDenoiserStage::Create(&text_encoder, bad_cfg, std::move(runners)),
      StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK_AND_ASSIGN(runners, MakeRecordingRunners(env, cfg));
  runners.main_blocks.clear();
  EXPECT_THAT(
      ZImageDenoiserStage::Create(&text_encoder, cfg, std::move(runners)),
      StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK_AND_ASSIGN(runners, MakeRecordingRunners(env, cfg));
  runners.final_layer.reset();
  EXPECT_THAT(
      ZImageDenoiserStage::Create(&text_encoder, cfg, std::move(runners)),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong number of inputs on `cap_embedder`.
  ASSERT_OK_AND_ASSIGN(runners, MakeRecordingRunners(env, cfg));
  runners.cap_embedder = std::make_unique<RecordingLiteRtRunner>(
      env, std::vector<std::vector<int>>{{1, cfg.seq_len, cfg.prompt_dim}, {1}},
      std::vector<std::vector<int>>{{1, cfg.seq_len, cfg.dim}});
  EXPECT_THAT(
      ZImageDenoiserStage::Create(&text_encoder, cfg, std::move(runners)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ZImageDenoiserStageTest, UpdatesDynamicCaptionRopeAndRunsEulerLoop) {
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  cfg.seq_len = 64;  // Two 32-token blocks so masking across blocks is visible.

  const size_t head_dim = static_cast<size_t>(cfg.HeadDim());
  const size_t img_tokens = static_cast<size_t>(cfg.ImageTokens());
  const size_t total_tokens = static_cast<size_t>(cfg.TotalTokens());
  const size_t packed_ch = static_cast<size_t>(cfg.PackedChannels());

  std::vector<float> seen_cap_cos;
  std::vector<float> seen_cap_sin;
  std::vector<float> seen_img_cos;
  std::vector<float> seen_unified_cos;
  std::vector<float> seen_t_vals;
  constexpr float kConstVelocity = 0.25f;

  ASSERT_OK_AND_ASSIGN(
      ZImageDenoiserStage::Runners runners,
      MakeRecordingRunners(
          env, cfg,
          /*refc_cb=*/
          [&](absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> /*outputs*/) -> absl::Status {
            seen_cap_cos.resize(static_cast<size_t>(cfg.seq_len) * head_dim);
            seen_cap_sin.resize(static_cast<size_t>(cfg.seq_len) * head_dim);
            ABSL_RETURN_IF_ERROR(
                ReadFloatBuffer(inputs[1], absl::MakeSpan(seen_cap_cos)));
            ABSL_RETURN_IF_ERROR(
                ReadFloatBuffer(inputs[2], absl::MakeSpan(seen_cap_sin)));
            return absl::OkStatus();
          },
          /*embx_cb=*/
          [&](absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> /*outputs*/) -> absl::Status {
            float t = 0.0f;
            ABSL_RETURN_IF_ERROR(
                ReadFloatBuffer(inputs[1], absl::MakeSpan(&t, 1)));
            seen_t_vals.push_back(t);
            return absl::OkStatus();
          },
          /*refx_cb=*/
          [&](absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> /*outputs*/) -> absl::Status {
            seen_img_cos.resize(img_tokens * head_dim);
            ABSL_RETURN_IF_ERROR(
                ReadFloatBuffer(inputs[1], absl::MakeSpan(seen_img_cos)));
            return absl::OkStatus();
          },
          /*main_cb=*/
          [&](absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> /*outputs*/) -> absl::Status {
            seen_unified_cos.resize(total_tokens * head_dim);
            ABSL_RETURN_IF_ERROR(
                ReadFloatBuffer(inputs[1], absl::MakeSpan(seen_unified_cos)));
            return absl::OkStatus();
          },
          /*final_cb=*/
          [&](absl::Span<const TensorBuffer> /*inputs*/,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> velocity(img_tokens * packed_ch, kConstVelocity);
            ABSL_RETURN_IF_ERROR(
                WriteFloatBuffer(outputs[0], absl::MakeConstSpan(velocity)));
            return absl::OkStatus();
          }));

  FakeTextEncoderStage text_encoder;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ZImageDenoiserStage> stage,
      ZImageDenoiserStage::Create(&text_encoder, cfg, std::move(runners)));

  // Prompt 1: 9 active tokens -> active_cap_len = 32 (second 32-token block
  // masked to zero, image temporal coordinate t = 33).
  TextEncoderOutput short_prompt;
  short_prompt.metadata.seed = 123;
  short_prompt.metadata.num_inference_steps = 4;
  short_prompt.prompt_embeds.assign(
      static_cast<size_t>(cfg.seq_len) * cfg.prompt_dim, 0.0f);
  for (int i = 0; i < 9; ++i) {
    short_prompt.prompt_embeds[static_cast<size_t>(i) * cfg.prompt_dim] = 1.0f;
  }
  text_encoder.Push(std::move(short_prompt));
  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  ASSERT_TRUE(stage->HasOutput());

  // Verify caption RoPE block [32..63] is zeroed and block [0..31] is active.
  for (size_t d = 0; d < head_dim; ++d) {
    EXPECT_NE(seen_cap_cos[d], 0.0f);
  }
  for (size_t idx = 32 * head_dim; idx < seen_cap_cos.size(); ++idx) {
    EXPECT_FLOAT_EQ(seen_cap_cos[idx], 0.0f);
    EXPECT_FLOAT_EQ(seen_cap_sin[idx], 0.0f);
  }

  // Verify image RoPE uses t = active_cap_len + 1 = 33.
  std::vector<float> expected_img_ids;
  BuildZImagePatchIds(cfg.GridSize(), cfg.GridSize(), /*seq_len=*/32,
                      expected_img_ids);
  ASSERT_OK_AND_ASSIGN(
      auto expected_img_rope,
      BuildFlux2RotaryPosEmbed(expected_img_ids, cfg.ImageTokens(),
                               cfg.axes_dims, cfg.rope_theta));
  EXPECT_THAT(seen_img_cos,
              Pointwise(FloatNear(1e-5f), expected_img_rope.first));
  // Unified cos concatenates img_cos followed by cap_cos.
  EXPECT_THAT(
      absl::MakeConstSpan(seen_unified_cos).subspan(0, img_tokens * head_dim),
      Pointwise(FloatNear(1e-5f), seen_img_cos));
  EXPECT_THAT(
      absl::MakeConstSpan(seen_unified_cos).subspan(img_tokens * head_dim),
      Pointwise(FloatNear(1e-5f), seen_cap_cos));

  // Verify Euler timesteps `t = 1 - sigma[i]` and exact latent sum.
  const std::vector<float> sigmas =
      ComputeZImageSigmas(4, cfg.ImageTokens(), cfg.shift, false);
  ASSERT_EQ(seen_t_vals.size(), 4u);
  for (int s = 0; s < 4; ++s) {
    EXPECT_NEAR(seen_t_vals[s], 1.0f - sigmas[s], 1e-6f);
  }

  ASSERT_OK_AND_ASSIGN(ZImageDenoiserOutput out, stage->GetOutput());
  EXPECT_EQ(out.metadata.width, cfg.img_size);
  EXPECT_EQ(out.metadata.height, cfg.img_size);
  EXPECT_EQ(out.metadata.num_inference_steps, 4);
  EXPECT_EQ(out.metadata.seed, 123u);
  const std::vector<float> init_latents =
      SampleGaussianLatents(img_tokens * packed_ch, /*seed=*/123);
  ASSERT_EQ(out.packed_latents.size(), init_latents.size());
  // Sum of (sigma[i] - sigma[i+1]) from 1.0 to 0.0 is exactly 1.0.
  for (size_t j = 0; j < init_latents.size(); ++j) {
    EXPECT_NEAR(out.packed_latents[j], init_latents[j] + kConstVelocity, 1e-5f);
  }

  // Prompt 2: 40 active tokens -> active_cap_len = 64; RoPE updates cleanly.
  TextEncoderOutput long_prompt;
  long_prompt.metadata.seed = 456;
  long_prompt.metadata.num_inference_steps = 2;
  long_prompt.prompt_embeds.assign(
      static_cast<size_t>(cfg.seq_len) * cfg.prompt_dim, 0.0f);
  for (int i = 0; i < 40; ++i) {
    long_prompt.prompt_embeds[static_cast<size_t>(i) * cfg.prompt_dim] = 1.0f;
  }
  text_encoder.Push(std::move(long_prompt));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(ZImageDenoiserOutput out2, stage->GetOutput());
  EXPECT_EQ(out2.metadata.num_inference_steps, 2);
  bool any_second_block_nonzero = false;
  for (size_t idx = 32 * head_dim; idx < seen_cap_cos.size(); ++idx) {
    if (std::abs(seen_cap_cos[idx]) > 1e-6f) {
      any_second_block_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(any_second_block_nonzero);
}

TEST(ZImageDenoiserStageTest, RejectsMismatchedPromptEmbedSize) {
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  ASSERT_OK_AND_ASSIGN(ZImageDenoiserStage::Runners runners,
                       MakeRecordingRunners(env, cfg));

  FakeTextEncoderStage text_encoder;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ZImageDenoiserStage> stage,
      ZImageDenoiserStage::Create(&text_encoder, cfg, std::move(runners)));

  TextEncoderOutput bad_prompt;
  bad_prompt.prompt_embeds = {1.0f, 2.0f, 3.0f};
  text_encoder.Push(std::move(bad_prompt));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ZImageDenoiserStageTest, ExecutesWithRealLiteRtCompiledModels) {
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();

  ModelOptions options;
  options.cache_dir = std::string(kNoCacheDir);
  options.backend = lm::Backend::CPU;
  options.num_threads = 1;

  const int seq = cfg.seq_len;
  const int img = cfg.ImageTokens();
  const int total = cfg.TotalTokens();
  const int dim = cfg.dim;
  const int pdim = cfg.prompt_dim;
  const int pch = cfg.PackedChannels();
  const int hdim = cfg.HeadDim();
  const int adaln = cfg.AdalnDim();

  // `CreateCompiledModelFromBuffer` borrows the FlatBuffer memory without
  // copying it; store buffers in a `std::deque` (stable element addresses) so
  // they outlive the compiled models.
  std::deque<std::string> model_bufs;
  auto compile = [&](absl::Span<const TensorSpec> ins,
                     absl::Span<const TensorSpec> outs,
                     absl::string_view name) -> absl::StatusOr<CompiledModel> {
    model_bufs.push_back(BuildMinimalTfliteModelBuffer(ins, outs));
    return CreateCompiledModelFromBuffer(env, options, model_bufs.back(), name);
  };

  ASSERT_OK_AND_ASSIGN(
      CompiledModel embc,
      compile({{"cap_feats", {1, seq, pdim}, tflite::TensorType_FLOAT32}},
              {{"cap", {1, seq, dim}, tflite::TensorType_FLOAT32}}, "embc"));
  ASSERT_OK_AND_ASSIGN(
      CompiledModel refc,
      compile({{"cap", {1, seq, dim}, tflite::TensorType_FLOAT32},
               {"cos", {1, seq, 1, hdim}, tflite::TensorType_FLOAT32},
               {"sin", {1, seq, 1, hdim}, tflite::TensorType_FLOAT32}},
              {{"cap_refined", {1, seq, dim}, tflite::TensorType_FLOAT32}},
              "refc"));
  ASSERT_OK_AND_ASSIGN(
      CompiledModel embx,
      compile({{"x", {1, img, pch}, tflite::TensorType_FLOAT32},
               {"t", {1}, tflite::TensorType_FLOAT32}},
              {{"x_emb", {1, img, dim}, tflite::TensorType_FLOAT32},
               {"adaln", {1, adaln}, tflite::TensorType_FLOAT32}},
              "embx"));
  ASSERT_OK_AND_ASSIGN(
      CompiledModel refx,
      compile({{"x_emb", {1, img, dim}, tflite::TensorType_FLOAT32},
               {"cos", {1, img, 1, hdim}, tflite::TensorType_FLOAT32},
               {"sin", {1, img, 1, hdim}, tflite::TensorType_FLOAT32},
               {"adaln", {1, adaln}, tflite::TensorType_FLOAT32}},
              {{"x_refined", {1, img, dim}, tflite::TensorType_FLOAT32}},
              "refx"));
  ASSERT_OK_AND_ASSIGN(
      CompiledModel main0,
      compile({{"unified", {1, total, dim}, tflite::TensorType_FLOAT32},
               {"cos", {1, total, 1, hdim}, tflite::TensorType_FLOAT32},
               {"sin", {1, total, 1, hdim}, tflite::TensorType_FLOAT32},
               {"adaln", {1, adaln}, tflite::TensorType_FLOAT32}},
              {{"unified_out", {1, total, dim}, tflite::TensorType_FLOAT32}},
              "main0"));
  std::vector<float> vel(static_cast<size_t>(img) * pch, 0.5f);
  ASSERT_OK_AND_ASSIGN(
      CompiledModel final_layer,
      compile({{"unified", {1, total, dim}, tflite::TensorType_FLOAT32},
               {"adaln", {1, adaln}, tflite::TensorType_FLOAT32}},
              {{"model_out", {1, img, pch}, tflite::TensorType_FLOAT32, vel}},
              "final"));

  ZImageDenoiserStage::Runners runners;
  runners.cap_embedder = std::make_unique<LiteRtRunnerImpl>(&embc);
  runners.context_refiner = std::make_unique<LiteRtRunnerImpl>(&refc);
  runners.x_embedder = std::make_unique<LiteRtRunnerImpl>(&embx);
  runners.noise_refiner = std::make_unique<LiteRtRunnerImpl>(&refx);
  runners.main_blocks.push_back(std::make_unique<LiteRtRunnerImpl>(&main0));
  runners.final_layer = std::make_unique<LiteRtRunnerImpl>(&final_layer);

  FakeTextEncoderStage text_encoder;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ZImageDenoiserStage> stage,
      ZImageDenoiserStage::Create(&text_encoder, cfg, std::move(runners)));

  TextEncoderOutput cond;
  cond.metadata.seed = 7;
  cond.metadata.num_inference_steps = 2;
  cond.prompt_embeds.assign(static_cast<size_t>(seq) * pdim, 0.1f);
  text_encoder.Push(std::move(cond));

  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(ZImageDenoiserOutput out, stage->GetOutput());

  const std::vector<float> expected_init =
      SampleGaussianLatents(static_cast<size_t>(img) * pch, /*seed=*/7);
  ASSERT_EQ(out.packed_latents.size(), expected_init.size());
  for (size_t i = 0; i < expected_init.size(); ++i) {
    EXPECT_NEAR(out.packed_latents[i], expected_init[i] + 0.5f, 1e-5f);
  }
}

}  // namespace
}  // namespace litert::omni::text2image
