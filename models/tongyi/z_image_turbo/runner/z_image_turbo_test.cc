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

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/tongyi/z_image_turbo/runner/z_image_turbo_factory.h"
#include "models/tongyi/z_image_turbo/runner/z_image_turbo_model_config.h"
#include "models/tongyi/z_image_turbo/runner/z_image_turbo_test_utils.h"
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
#include "omni/any2any/any2any_test_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_cache.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/io_types.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/vae_decoder_stage.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;
using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::Pointwise;

absl::StatusOr<Environment> CreateTestEnvironment() {
  LITERT_ASSIGN_OR_RETURN(Environment env,
                          Environment::Create(EnvironmentOptions(
                              absl::Span<const EnvironmentOptions::Option>{})));
  return env;
}

class FakeDenoiserStage : public SingleThreadedStageWithDeque<DenoisedLatents> {
 public:
  void Push(DenoisedLatents out) { PushOutput(std::move(out)); }

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

class RecordingVaeRunner : public LiteRtRunner {
 public:
  using RunCallback = std::function<absl::Status(
      absl::Span<const TensorBuffer>, absl::Span<const TensorBuffer>)>;

  RecordingVaeRunner(Environment& env, std::vector<std::vector<int>> in_shapes,
                     std::vector<std::vector<int>> out_shapes,
                     RunCallback on_run = nullptr)
      : env_(env),
        in_shapes_(std::move(in_shapes)),
        out_shapes_(std::move(out_shapes)),
        on_run_(std::move(on_run)) {}

  absl::StatusOr<std::vector<TensorBuffer>> CreateInputBuffers(
      absl::string_view /*signature_name*/) override {
    std::vector<TensorBuffer> bufs;
    for (const auto& shape : in_shapes_) {
      ABSL_ASSIGN_OR_RETURN(TensorBuffer buf,
                            MakeFloatTensorBuffer(env_, shape));
      bufs.push_back(std::move(buf));
    }
    return bufs;
  }

  absl::StatusOr<std::vector<TensorBuffer>> CreateOutputBuffers(
      absl::string_view /*signature_name*/) override {
    std::vector<TensorBuffer> bufs;
    for (const auto& shape : out_shapes_) {
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
  std::vector<std::vector<int>> in_shapes_;
  std::vector<std::vector<int>> out_shapes_;
  RunCallback on_run_;
};

TEST(ZImageTurboFactoryTest, CreateResourcesValidatesArguments) {
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  auto shared_env = std::make_shared<Environment>(std::move(env));
  EXPECT_THAT(CreateZImageTurboResources(cfg, /*model_folder=*/"", kNoCacheDir,
                                         lm::Backend::CPU, /*num_threads=*/1,
                                         shared_env, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ZImageTurboTestBundleOptions no_shards_opts;
  no_shards_opts.config = cfg;
  no_shards_opts.num_main_shards = 0;
  ASSERT_OK_AND_ASSIGN(
      std::string no_shards_path,
      WriteZImageTurboBundle("z_image_no_shards.litertlm", no_shards_opts));
  EXPECT_THAT(CreateZImageTurboResources(cfg, no_shards_path, kNoCacheDir,
                                         lm::Backend::CPU, /*num_threads=*/1,
                                         shared_env, nullptr),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ZImageTurboFactoryTest, CreateComponentsFailsWithoutTokenizerOrModels) {
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  auto empty_resources = std::make_shared<ModelResources>(
      std::make_shared<Environment>(std::move(env)));
  std::vector<std::unique_ptr<internal::StageBase>> stages;

  EXPECT_THAT(CreateZImageTurboComponents(cfg, /*model_folder=*/"",
                                          std::make_unique<PushPromptSource>(),
                                          empty_resources, stages),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ZImageVaeDecoderStageTest, CreateValidatesInputsAndConfig) {
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  FakeDenoiserStage denoiser;
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  const int lat = cfg.LatentSize();

  auto make_runner = [&] {
    return std::make_unique<RecordingVaeRunner>(
        env, std::vector<std::vector<int>>{{1, cfg.in_channels, lat, lat}},
        std::vector<std::vector<int>>{{1, 3, cfg.img_size, cfg.img_size}});
  };

  ZImageTurboModelConfig zero_scale = cfg;
  zero_scale.vae_scaling_factor = 0.0f;
  EXPECT_THAT(CreateZImageVaeDecoderStage(&denoiser, zero_scale, make_runner()),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ZImageTurboModelConfig bad_size = cfg;
  bad_size.img_size = 12;
  EXPECT_THAT(CreateZImageVaeDecoderStage(&denoiser, bad_size, make_runner()),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_runner = std::make_unique<RecordingVaeRunner>(
      env, std::vector<std::vector<int>>{{1, cfg.in_channels, lat, lat}, {1}},
      std::vector<std::vector<int>>{{1, 3, cfg.img_size, cfg.img_size}});
  EXPECT_THAT(
      CreateZImageVaeDecoderStage(&denoiser, cfg, std::move(bad_runner)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ZImageVaeDecoderStageTest, UnpatchifiesRescalesAndConvertsRgb888) {
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  cfg.vae_scaling_factor = 0.5f;
  cfg.vae_shift_factor = 0.25f;
  const int lat = cfg.LatentSize();
  const size_t total_latents = static_cast<size_t>(cfg.in_channels) * lat * lat;
  const size_t pixels =
      static_cast<size_t>(cfg.img_size) * static_cast<size_t>(cfg.img_size);

  std::vector<float> expected_nchw(total_latents);
  for (size_t i = 0; i < total_latents; ++i) {
    expected_nchw[i] = static_cast<float>(i) * 0.1f;
  }
  std::vector<float> packed(total_latents, 0.0f);
  ASSERT_OK(PackZImageLatents(expected_nchw, absl::MakeSpan(packed), lat, lat,
                              cfg.patch_size, cfg.in_channels));

  std::vector<float> seen_vae_input;
  auto runner = std::make_unique<RecordingVaeRunner>(
      env, std::vector<std::vector<int>>{{1, cfg.in_channels, lat, lat}},
      std::vector<std::vector<int>>{{1, 3, cfg.img_size, cfg.img_size}},
      [&](absl::Span<const TensorBuffer> inputs,
          absl::Span<const TensorBuffer> outputs) -> absl::Status {
        seen_vae_input.resize(total_latents);
        ABSL_RETURN_IF_ERROR(
            ReadFloatBuffer(inputs[0], absl::MakeSpan(seen_vae_input)));
        // Emit distinct per-channel values: R = -1.0 (0), G = 0.0 (128),
        // B = 1.5 (clamped to 255).
        std::vector<float> rgb(3 * pixels);
        for (size_t p = 0; p < pixels; ++p) {
          rgb[p] = -1.0f;
          rgb[pixels + p] = 0.0f;
          rgb[2 * pixels + p] = 1.5f;
        }
        ABSL_RETURN_IF_ERROR(
            WriteFloatBuffer(outputs[0], absl::MakeConstSpan(rgb)));
        return absl::OkStatus();
      });

  FakeDenoiserStage denoiser;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VaeDecoderStage> stage,
      CreateZImageVaeDecoderStage(&denoiser, cfg, std::move(runner)));
  EXPECT_FALSE(stage->NeedSchedule());
  ASSERT_OK(stage->Flush());

  DenoisedLatents denoised;
  denoised.packed_latents = packed;
  denoiser.Push(std::move(denoised));
  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());

  std::vector<float> expected_rescaled(total_latents);
  for (size_t i = 0; i < total_latents; ++i) {
    expected_rescaled[i] = expected_nchw[i] / 0.5f + 0.25f;
  }
  EXPECT_THAT(seen_vae_input, Pointwise(FloatNear(1e-5f), expected_rescaled));

  ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
  const auto* img = std::get_if<ImageOutput>(&out);
  ASSERT_NE(img, nullptr);
  EXPECT_EQ(img->width, cfg.img_size);
  EXPECT_EQ(img->height, cfg.img_size);
  EXPECT_EQ(img->channels, 3);
  ASSERT_EQ(img->rgb_data.size(), 3 * pixels);
  for (size_t p = 0; p < pixels; ++p) {
    EXPECT_EQ(img->rgb_data[3 * p + 0], 0);
    EXPECT_NEAR(img->rgb_data[3 * p + 1], 128, 1);
    EXPECT_EQ(img->rgb_data[3 * p + 2], 255);
  }
}

TEST(ZImageVaeDecoderStageTest, RejectsMismatchedPackedLatentsSize) {
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  const int lat = cfg.LatentSize();

  auto runner = std::make_unique<RecordingVaeRunner>(
      env, std::vector<std::vector<int>>{{1, cfg.in_channels, lat, lat}},
      std::vector<std::vector<int>>{{1, 3, cfg.img_size, cfg.img_size}});
  FakeDenoiserStage denoiser;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VaeDecoderStage> stage,
      CreateZImageVaeDecoderStage(&denoiser, cfg, std::move(runner)));

  DenoisedLatents bad;
  bad.packed_latents = {1.0f, 2.0f};
  denoiser.Push(std::move(bad));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ZImageVaeDecoderStageTest, ExecutesWithRealLiteRtCompiledModel) {
  ASSERT_OK_AND_ASSIGN(Environment env, CreateTestEnvironment());
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  const int lat = cfg.LatentSize();
  const size_t pixels =
      static_cast<size_t>(cfg.img_size) * static_cast<size_t>(cfg.img_size);

  std::vector<float> vae_out(3 * pixels, 0.5f);
  std::string model_buf = any2any::BuildMinimalTfliteModelBuffer(
      {{"latents", {1, cfg.in_channels, lat, lat}, tflite::TensorType_FLOAT32}},
      {{"rgb",
        {1, 3, cfg.img_size, cfg.img_size},
        tflite::TensorType_FLOAT32,
        std::move(vae_out)}});

  ModelOptions options;
  options.cache_dir = std::string(kNoCacheDir);
  options.backend = lm::Backend::CPU;
  options.num_threads = 1;
  ASSERT_OK_AND_ASSIGN(
      CompiledModel vae_model,
      CreateCompiledModelFromBuffer(env, options, model_buf, "zvae"));

  FakeDenoiserStage denoiser;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VaeDecoderStage> stage,
      CreateZImageVaeDecoderStage(
          &denoiser, cfg, std::make_unique<LiteRtRunnerImpl>(&vae_model)));

  DenoisedLatents denoised;
  denoised.packed_latents.assign(
      static_cast<size_t>(cfg.ImageTokens()) * cfg.PackedChannels(), 0.0f);
  denoiser.Push(std::move(denoised));

  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
  const auto* img = std::get_if<ImageOutput>(&out);
  ASSERT_NE(img, nullptr);
  EXPECT_EQ(img->width, cfg.img_size);
  EXPECT_EQ(img->height, cfg.img_size);
  ASSERT_EQ(img->rgb_data.size(), 3 * pixels);
  // 0.5f -> (0.5 * 0.5 + 0.5) * 255 = 191.25 -> 191.
  EXPECT_NEAR(img->rgb_data[0], 191, 1);
}

TEST(ZImageTurboRegistryTest, RegistersAllAliasesAndMatchesFilenames) {
  for (const char* alias : {"z-image-turbo", "z_image_turbo",
                            "Tongyi-MAI/Z-Image-Turbo", "z-image", "z_image"}) {
    EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered(alias)) << alias;
  }

  ASSERT_OK_AND_ASSIGN(OmniRegistry::RegisteredModel reg,
                       OmniRegistry::Get().GetRegisteredModel("z-image-turbo"));
  EXPECT_EQ(reg.canonical_model_name, "z-image-turbo");
  EXPECT_EQ(reg.session_config_proto_type_id,
            "litert.lm.proto.Text2ImageSessionConfig");
  EXPECT_EQ(reg.metadata_proto_type_id, "litert.lm.proto.Text2ImageMetadata");

  ASSERT_OK_AND_ASSIGN(std::string inferred,
                       OmniRegistry::Get().FindModelNameForPath(
                           "/tmp/z_image_turbo_fp16.litertlm"));
  ASSERT_OK_AND_ASSIGN(OmniRegistry::RegisteredModel inferred_reg,
                       OmniRegistry::Get().GetRegisteredModel(inferred));
  EXPECT_EQ(inferred_reg.canonical_model_name, "z-image-turbo");
}

TEST(ZImageTurboEngineTest, GeneratesImageEndToEndWithAll13Shards) {
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteZImageTurboBundle("z_image_turbo_e2e_13shards.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = std::string(kNoCacheDir);
  settings.num_threads = 1;

  lm::proto::Text2ImageSessionConfig session_cfg;
  session_cfg.set_num_inference_steps(2);
  session_cfg.set_seed(42);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniEngine> engine,
      OmniEngine::Create("z-image-turbo", settings, &session_cfg));

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       engine->CreateSession(std::move(input_source)));

  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "a red fox"}));
  raw_input->Finish();

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session->ProcessNext());
  const auto* img = std::get_if<ImageOutput>(&out);
  ASSERT_NE(img, nullptr);
  EXPECT_EQ(img->width, 16);
  EXPECT_EQ(img->height, 16);
  EXPECT_EQ(img->channels, 3);
  ASSERT_EQ(img->rgb_data.size(), 16u * 16u * 3u);
  // Default test bundle emits `vae_rgb_value = 0.5f` -> pixel value ~191.
  EXPECT_NEAR(img->rgb_data[0], 191, 1);

  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(ZImageTurboEngineTest, SupportsFewerMainBlockShards) {
  ZImageTurboTestBundleOptions bundle_opts;
  bundle_opts.config = MakeTinyZImageTurboConfig();
  bundle_opts.num_main_shards = 2;
  bundle_opts.vae_rgb_value = -0.5f;
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteZImageTurboBundle("z_image_turbo_2shards.litertlm", bundle_opts));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = std::string(kNoCacheDir);
  settings.num_threads = 1;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniEngine> engine,
                       OmniEngine::Create("z-image-turbo", settings));
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       engine->CreateSession(std::move(input_source)));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "cat"}));
  raw_input->Finish();

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session->ProcessNext());
  const auto* img = std::get_if<ImageOutput>(&out);
  ASSERT_NE(img, nullptr);
  ASSERT_EQ(img->rgb_data.size(), 16u * 16u * 3u);
  // -0.5f -> (-0.5 * 0.5 + 0.5) * 255 = 63.75 -> 64.
  EXPECT_NEAR(img->rgb_data[0], 64, 1);
}

TEST(ZImageTurboEngineTest, CallerMetadataOverrideUnpacksCustomModelCleanly) {
  ZImageTurboModelConfig cfg = MakeTinyZImageTurboConfig();
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteZImageTurboBundle("z_image_override.litertlm"));

  // Override metadata via caller-provided `Text2ImageMetadata` packing a
  // `ZImageTurbo` proto with updated default_steps and shift.
  cfg.steps = 3;
  cfg.shift = 2.0f;
  ASSERT_OK_AND_ASSIGN(lm::proto::Text2ImageMetadata caller_meta,
                       BuildDefaultZImageTurboMetadata(cfg));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = std::string(kNoCacheDir);
  settings.num_threads = 1;

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniEngine> engine,
      OmniEngine::Create("z-image-turbo", settings,
                         /*session_config=*/nullptr, &caller_meta));

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       engine->CreateSession(std::move(input_source)));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "sunset"}));
  raw_input->Finish();

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
}

TEST(ZImageTurboEngineTest, RejectsWrongImageGenModelTypeArm) {
  lm::proto::Text2ImageMetadata wrong_arm_meta;
  wrong_arm_meta.mutable_image_gen_model_type()->mutable_bonsai_flux2();
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteZImageTurboBundle("z_image_wrong_arm.litertlm", &wrong_arm_meta));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = std::string(kNoCacheDir);
  settings.num_threads = 1;

  EXPECT_THAT(OmniEngine::Create("z-image-turbo", settings),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ZImageTurboEngineTest, RejectsWrongCustomModelTypeUrl) {
  lm::proto::Text2ImageMetadata wrong_any_meta;
  lm::proto::Text2ImageSessionConfig unrelated_proto;
  unrelated_proto.set_width(256);
  ASSERT_TRUE(wrong_any_meta.mutable_image_gen_model_type()
                  ->mutable_custom_model()
                  ->PackFrom(unrelated_proto));
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteZImageTurboBundle("z_image_wrong_any.litertlm", &wrong_any_meta));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = std::string(kNoCacheDir);
  settings.num_threads = 1;

  EXPECT_THAT(OmniEngine::Create("z-image-turbo", settings),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ZImageTurboEngineTest, RejectsNonSquareOrMismatchedSessionDimensions) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteZImageTurboBundle("z_image_dim_check.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = std::string(kNoCacheDir);
  settings.num_threads = 1;

  lm::proto::Text2ImageSessionConfig non_square;
  non_square.set_width(16);
  non_square.set_height(32);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniEngine> non_square_engine,
      OmniEngine::Create("z-image-turbo", settings, &non_square));
  EXPECT_THAT(
      non_square_engine->CreateSession(std::make_unique<PushInputSource>()),
      StatusIs(absl::StatusCode::kInvalidArgument));

  lm::proto::Text2ImageSessionConfig wrong_size;
  wrong_size.set_width(32);
  wrong_size.set_height(32);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniEngine> wrong_size_engine,
      OmniEngine::Create("z-image-turbo", settings, &wrong_size));
  EXPECT_THAT(
      wrong_size_engine->CreateSession(std::make_unique<PushInputSource>()),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ZImageTurboEngineTest, EnforcesSingleActiveSessionLease) {
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteZImageTurboBundle("z_image_session_lease.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = std::string(kNoCacheDir);
  settings.num_threads = 1;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniEngine> engine,
                       OmniEngine::Create("z-image-turbo", settings));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSession> first_session,
      engine->CreateSession(std::make_unique<PushInputSource>()));
  EXPECT_THAT(engine->CreateSession(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Another Z-Image-Turbo session is already "
                                 "active on this engine.")));

  // Releasing the first session allows creating a subsequent session cleanly.
  first_session.reset();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSession> second_session,
      engine->CreateSession(std::make_unique<PushInputSource>()));
  EXPECT_NE(second_session, nullptr);
}

}  // namespace
}  // namespace litert::omni::text2image
