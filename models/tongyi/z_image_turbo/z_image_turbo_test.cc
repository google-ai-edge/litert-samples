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

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/tongyi/z_image_turbo/z_image_turbo_factory.h"
#include "models/tongyi/z_image_turbo/z_image_turbo_model_config.h"
#include "models/tongyi/z_image_turbo/z_image_turbo_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_macros.h"
#include "omni/base/io_types.h"
#include "omni/base/model_cache.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;

absl::StatusOr<Environment> CreateTestEnvironment() {
  LITERT_ASSIGN_OR_RETURN(Environment env,
                          Environment::Create(EnvironmentOptions(
                              absl::Span<const EnvironmentOptions::Option>{})));
  return env;
}

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
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Releasing the first session allows creating a subsequent session cleanly.
  first_session.reset();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSession> second_session,
      engine->CreateSession(std::make_unique<PushInputSource>()));
  EXPECT_NE(second_session, nullptr);
}

}  // namespace
}  // namespace litert::omni::text2image
